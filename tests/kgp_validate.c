/* KGP_DEBUG=1, the validation mode (kgp_validate.c): the mistakes it exists
 * for are refused when recorded or submitted, and a correct command buffer
 * is not. Without the mode every refused case here would reach the GPU.
 * Nothing needs results from the GPU, so it checks the same on the shim and
 * on hardware.
 */
#include "kgp_test.h"

#include "heap_fetch_spv.h"
#include "saxpy_spv.h"

#include <string.h>

#define COUNT 256
#define TEX_W 64
#define TEX_H 64

struct saxpy_root {
   uint64_t x;
   uint64_t y;
   float a;
   uint32_t count;
};

/* heap_fetch.comp */
struct fetch_root {
   uint64_t dst;
   uint32_t texture_index;
   uint32_t width;
   uint32_t height;
   uint32_t layer;
};

#define TEXEL 0xff336699u       /* what the texture is filled with */
#define UNTOUCHED 0xdeadbeefu   /* what the fetch destinations start as */
#define HEAP_USED 4u            /* entries the heap is said to have */
#define HEAP_INSIDE 1u
#define HEAP_OUTSIDE 5u         /* a valid descriptor, past that count */

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

   /* The shader side: a heap index past the entries the command buffer
    * gave the heap reads zero, though a valid descriptor lies there; one
    * inside reads the texture.
    */
   KgpPipeline fetch = kgpCreateComputePipeline(
      dev, (KgpByteSpan) { heap_fetch_spv, sizeof(heap_fetch_spv) }, NULL);
   uint32_t *inside = kgpMalloc(dev, TEX_W * TEX_H * 4, 16,
                                KGP_MEMORY_READBACK);
   uint32_t *outside = kgpMalloc(dev, TEX_W * TEX_H * 4, 16,
                                 KGP_MEMORY_READBACK);
   struct fetch_root *fetch_roots =
      kgpMalloc(dev, 2 * sizeof(*fetch_roots), 16, KGP_MEMORY_DEFAULT);
   CHECK(fetch && inside && outside && fetch_roots);
   _Static_assert(HEAP_OUTSIDE < 8, "inside the heap's memory");
   for (unsigned i = 0; i < TEX_W * TEX_H; i++) {
      texels[i] = TEXEL;
      inside[i] = outside[i] = UNTOUCHED;
   }
   memset(heap, 0, heap_entries * sizeof(KgpDescriptor256));
   KgpDescriptor256 *entries = heap;
   CHECK(kgpTextureViewDescriptor(tex, NULL, &entries[HEAP_INSIDE]));
   CHECK(kgpTextureViewDescriptor(tex, NULL, &entries[HEAP_OUTSIDE]));
   fetch_roots[0] = (struct fetch_root) {
      .dst = (uintptr_t)kgpHostToDevicePointer(dev, inside),
      .texture_index = HEAP_INSIDE, .width = TEX_W, .height = TEX_H,
   };
   fetch_roots[1] = fetch_roots[0];
   fetch_roots[1].dst = (uintptr_t)kgpHostToDevicePointer(dev, outside);
   fetch_roots[1].texture_index = HEAP_OUTSIDE;
   void *fetch_roots_gpu = kgpHostToDevicePointer(dev, fetch_roots);

   cb = kgpStartCommandRecording(queue);
   CHECK(cb);
   kgpCopyToTexture(cb, tex, 0, 0, texels_gpu, TEX_W * 4);
   kgpBarrier(cb, KGP_STAGE_TRANSFER, KGP_STAGE_COMPUTE, KGP_HAZARD_NONE);
   kgpSetActiveTextureHeapPtr(cb, heap_gpu, HEAP_USED);
   kgpSetPipeline(cb, fetch);
   for (unsigned i = 0; i < 2; i++) {
      kgpDispatch(cb, (char *)fetch_roots_gpu + i * sizeof(*fetch_roots),
                  (KgpUvec3) { TEX_W / 8, TEX_H / 8, 1 });
   }
   CHECK(kgpSubmit(queue, &cb, 1, done, 2));
   CHECK(kgpWaitSemaphore(dev, done, 2, KGP_TEST_TIMEOUT_NS));
   for (unsigned i = 0; !kgp_test_offline() && i < TEX_W * TEX_H; i++) {
      if (inside[i] != TEXEL || outside[i] != 0) {
         fprintf(stderr, "validate: texel %u read %08x inside the heap "
                 "(expected %08x) and %08x outside (expected 0)\n",
                 i, inside[i], TEXEL, outside[i]);
         return 1;
      }
   }

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
   kgpDestroyPipeline(dev, fetch);
   kgpFree(dev, fetch_roots);
   kgpFree(dev, outside);
   kgpFree(dev, inside);
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
