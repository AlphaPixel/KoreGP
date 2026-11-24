/* KGP_DEBUG=1, the validation mode (kgp_validate.c): the mistakes it exists
 * for are refused when recorded or submitted, and a correct command buffer
 * is not. Without the mode every refused case here would reach the GPU.
 * Nothing needs results from the GPU, so it checks the same on the shim and
 * on hardware.
 */
#include "kgp_test.h"

#include "saxpy_spv.h"

#define COUNT 256
#define TEX_W 64
#define TEX_H 64

struct saxpy_root {
   uint64_t x;
   uint64_t y;
   float a;
   uint32_t count;
};

static KgpQueue queue;
static KgpPipeline pipeline;

static KgpCommandBuffer
start(void)
{
   KgpCommandBuffer cb = kgpStartCommandRecording(queue);
   CHECK(cb);
   kgpSetPipeline(cb, pipeline);
   return cb;
}

static bool
submit(KgpCommandBuffer cb)
{
   return kgpSubmit(queue, &cb, 1, NULL, 0);
}

int
main(void)
{
   setenv("KGP_DEBUG", "1", 1);
   KgpDevice dev = kgpCreateDevice(0);
   CHECK(dev);
   queue = kgpGetQueue(dev);
   pipeline = kgpCreateComputePipeline(
      dev, (KgpByteSpan) { saxpy_spv, sizeof(saxpy_spv) }, NULL);
   CHECK(pipeline);

   const size_t bytes = COUNT * sizeof(float);
   float *x = kgpMalloc(dev, bytes, 16, KGP_MEMORY_DEFAULT);
   float *y = kgpMalloc(dev, bytes, 16, KGP_MEMORY_READBACK);
   struct saxpy_root *root =
      kgpMalloc(dev, sizeof(*root), 16, KGP_MEMORY_DEFAULT);
   struct saxpy_root *doomed =
      kgpMalloc(dev, sizeof(*root), 16, KGP_MEMORY_DEFAULT);
   const uint32_t heap_entries = 8;
   void *heap = kgpMalloc(dev, heap_entries * sizeof(KgpDescriptor256), 32,
                          KGP_MEMORY_DEFAULT);
   CHECK(x && y && root && doomed && heap);
   for (unsigned i = 0; i < COUNT; i++) {
      x[i] = i;
      y[i] = 1;
   }
   void *x_gpu = kgpHostToDevicePointer(dev, x);
   void *y_gpu = kgpHostToDevicePointer(dev, y);
   void *root_gpu = kgpHostToDevicePointer(dev, root);
   void *doomed_gpu = kgpHostToDevicePointer(dev, doomed);
   void *heap_gpu = kgpHostToDevicePointer(dev, heap);
   *root = (struct saxpy_root) {
      .x = (uintptr_t)x_gpu, .y = (uintptr_t)y_gpu, .a = 2, .count = COUNT,
   };
   *doomed = *root;

   const KgpTextureDesc tex_desc = {
      .format = KGP_FORMAT_RGBA8_UNORM, .width = TEX_W, .height = TEX_H,
   };
   KgpSizeAlign sa;
   CHECK(kgpTextureSizeAlign(dev, &tex_desc, &sa));
   void *tex_mem = kgpMalloc(dev, sa.size, sa.align, KGP_MEMORY_GPU);
   void *small_mem = kgpMalloc(dev, sa.size / 2, sa.align, KGP_MEMORY_GPU);
   uint32_t *texels = kgpMalloc(dev, TEX_W * TEX_H * 4, 16,
                                KGP_MEMORY_READBACK);
   CHECK(tex_mem && small_mem && texels);
   void *texels_gpu = kgpHostToDevicePointer(dev, texels);
   KgpTexture tex = kgpCreateTexture(dev, &tex_desc, tex_mem);
   CHECK(tex);

   const KgpUvec3 groups = { COUNT / 64, 1, 1 };
   KgpCommandBuffer cb;

   /* Correct use of everything checked below goes through. */
   KgpSemaphore done = kgpCreateSemaphore(dev, 0);
   CHECK(done);
   cb = start();
   kgpSetActiveTextureHeapPtr(cb, heap_gpu, heap_entries);
   kgpDispatch(cb, root_gpu, groups);
   kgpBarrier(cb, KGP_STAGE_COMPUTE, KGP_STAGE_TRANSFER, KGP_HAZARD_NONE);
   kgpMemCpy(cb, x_gpu, y_gpu, bytes);
   kgpCopyFromTexture(cb, tex, 0, 0, texels_gpu, TEX_W * 4);
   CHECK(kgpSubmit(queue, &cb, 1, done, 1));
   CHECK(kgpWaitSemaphore(dev, done, 1, KGP_TEST_TIMEOUT_NS));
   if (!kgp_test_offline())
      CHECK(y[3] == 7 && x[3] == 7);    /* 2 * 3 + 1, then copied */

   fprintf(stderr, "(the errors logged below are the point of this test)\n");

   /* A CPU pointer where a GPU pointer belongs. */
   cb = start();
   kgpDispatch(cb, root, groups);
   CHECK(!submit(cb));

   /* A root pointer that is not aligned. */
   cb = start();
   kgpDispatch(cb, (char *)root_gpu + 2, groups);
   CHECK(!submit(cb));

   /* One byte past the end of an allocation, on either side of a copy. */
   cb = start();
   kgpMemCpy(cb, x_gpu, y_gpu, bytes + 1);
   CHECK(!submit(cb));
   cb = start();
   kgpMemCpy(cb, (char *)x_gpu + 1, y_gpu, bytes);
   CHECK(!submit(cb));

   /* A heap said to have more entries than its memory holds. */
   cb = start();
   kgpSetActiveTextureHeapPtr(cb, heap_gpu, heap_entries + 1);
   CHECK(!submit(cb));
   cb = start();
   kgpSetActiveSamplerHeapPtr(cb, heap_gpu, heap_entries + 1);
   CHECK(!submit(cb));

   /* Barriers between nothing, or between stages that do not exist. */
   cb = start();
   kgpBarrier(cb, (KGP_STAGE)0, KGP_STAGE_COMPUTE, KGP_HAZARD_NONE);
   CHECK(!submit(cb));
   cb = start();
   kgpBarrier(cb, KGP_STAGE_COMPUTE, (KGP_STAGE)(1 << 10), KGP_HAZARD_NONE);
   CHECK(!submit(cb));

   /* A texture read back with rows that overrun the destination, and a
    * texture over memory too small for it.
    */
   cb = start();
   kgpCopyFromTexture(cb, tex, 0, 0, texels_gpu, TEX_W * 4 + 4);
   CHECK(!submit(cb));
   CHECK(!kgpCreateTexture(dev, &tex_desc, small_mem));

   /* Memory freed between recording and submit: nothing wrong when it was
    * recorded, so it is the submit that refuses.
    */
   cb = start();
   kgpDispatch(cb, doomed_gpu, groups);
   kgpFree(dev, doomed);
   CHECK(!submit(cb));

   /* And used after it was freed. */
   cb = start();
   kgpDispatch(cb, doomed_gpu, groups);
   CHECK(!submit(cb));

   kgpDestroySemaphore(dev, done);
   kgpDestroyTexture(dev, tex);
   kgpDestroyPipeline(dev, pipeline);
   kgpFree(dev, texels);
   kgpFree(dev, small_mem);
   kgpFree(dev, tex_mem);
   kgpFree(dev, heap);
   kgpFree(dev, root);
   kgpFree(dev, y);
   kgpFree(dev, x);
   /* The poisoned command buffers are reclaimed here. */
   kgpDestroyDevice(dev);

   printf("validate: every mistake was refused, correct use was not\n");
   return 0;
}
