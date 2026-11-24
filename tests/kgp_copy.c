/* GPU-side copies (plan N2, task T7), chained so each depends on the last:
 *
 *    CPU-filled memory --kgpMemCpy--> GPU-only memory
 *       --kgpCopyToTexture--> a texture in GPU-only memory
 *          --kgpCopyFromTexture--> readback memory      (checked)
 *          --texelFetch in a shader--> readback memory  (checked)
 *
 * The shader read is what shows the copy engine wrote the layout the texture
 * unit expects; the copy back alone would only show the engine agrees with
 * itself.
 *
 *    kgp_copy            the hardware's tiled layout
 *    kgp_copy linear     KGP_TEXTURE_LINEAR
 *
 * Vulkan twin: vk_baseline copy [linear], on NVK's transfer queue.
 */
#include "kgp_test.h"

#include "copy_pattern.h"
#include "heap_fetch_spv.h"

#include <string.h>

#define HEAP_ENTRIES 4u
#define SLOT 2u

struct fetch_root {
   uint64_t dst;
   uint32_t texture_index;
   uint32_t width;
   uint32_t height;
   uint32_t pad[11];
};
_Static_assert(sizeof(struct fetch_root) == 64, "valid under root v1 too");

int
main(int argc, char **argv)
{
   const bool linear = argc > 1 && strcmp(argv[1], "linear") == 0;
   const char *mode = linear ? "linear" : "tiled";

   KgpDevice dev = kgpCreateDevice(0);
   CHECK(dev);
   KgpQueue queue = kgpGetQueue(dev);

   const KgpTextureDesc desc = {
      .format = KGP_FORMAT_RGBA8_UNORM,
      .width = COPY_W,
      .height = COPY_H,
      .flags = linear ? KGP_TEXTURE_LINEAR : 0,
   };
   KgpSizeAlign sa;
   CHECK(kgpTextureSizeAlign(dev, &desc, &sa));

   /* Only the first and the last two are CPU-visible. */
   uint32_t *src = kgpMalloc(dev, COPY_BYTES, 64, KGP_MEMORY_DEFAULT);
   void *mid_gpu = kgpMalloc(dev, COPY_BYTES, 64, KGP_MEMORY_GPU);
   void *tex_gpu = kgpMalloc(dev, sa.size, sa.align, KGP_MEMORY_GPU);
   uint32_t *out = kgpMalloc(dev, COPY_OUT_BYTES, 64, KGP_MEMORY_READBACK);
   uint32_t *fetched = kgpMalloc(dev, COPY_BYTES, 64, KGP_MEMORY_READBACK);
   CHECK(src && mid_gpu && tex_gpu && out && fetched);
   copy_fill(src);
   memset(out, 0, COPY_OUT_BYTES);
   memset(fetched, 0, COPY_BYTES);

   KgpTexture tex = kgpCreateTexture(dev, &desc, tex_gpu);
   CHECK(tex);

   KgpDescriptor256 *heap =
      kgpMalloc(dev, HEAP_ENTRIES * sizeof(*heap), 64, KGP_MEMORY_DEFAULT);
   struct fetch_root *root =
      kgpMalloc(dev, sizeof(*root), 64, KGP_MEMORY_DEFAULT);
   CHECK(heap && root);
   memset(heap, 0, HEAP_ENTRIES * sizeof(*heap));
   CHECK(kgpTextureViewDescriptor(tex, NULL, &heap[SLOT]));
   *root = (struct fetch_root) {
      .dst = (uintptr_t)kgpHostToDevicePointer(dev, fetched),
      .texture_index = SLOT, .width = COPY_W, .height = COPY_H,
   };

   KgpPipeline fetch = kgpCreateComputePipeline(
      dev, (KgpByteSpan) { heap_fetch_spv, sizeof(heap_fetch_spv) }, NULL);
   KgpSemaphore done = kgpCreateSemaphore(dev, 0);
   CHECK(fetch && done);

   KgpCommandBuffer cb = kgpStartCommandRecording(queue);
   CHECK(cb);

   kgpMemCpy(cb, mid_gpu, kgpHostToDevicePointer(dev, src), COPY_BYTES);
   kgpBarrier(cb, KGP_STAGE_TRANSFER, KGP_STAGE_TRANSFER, KGP_HAZARD_NONE);
   kgpCopyToTexture(cb, tex, 0, 0, mid_gpu, COPY_W * 4);
   kgpBarrier(cb, KGP_STAGE_TRANSFER, KGP_STAGE_TRANSFER, KGP_HAZARD_NONE);
   kgpCopyFromTexture(cb, tex, 0, 0, kgpHostToDevicePointer(dev, out),
                      COPY_OUT_ROW_TEXELS * 4);
   kgpBarrier(cb, KGP_STAGE_TRANSFER, KGP_STAGE_COMPUTE, KGP_HAZARD_NONE);

   kgpSetActiveTextureHeapPtr(cb, kgpHostToDevicePointer(dev, heap),
                              HEAP_ENTRIES);
   kgpSetPipeline(cb, fetch);
   kgpDispatch(cb, kgpHostToDevicePointer(dev, root),
               (KgpUvec3) { (COPY_W + 7) / 8, (COPY_H + 7) / 8, 1 });

   CHECK(kgpSubmit(queue, &cb, 1, done, 1));
   CHECK(kgpWaitSemaphore(dev, done, 1, KGP_TEST_TIMEOUT_NS));

   /* Bad arguments fail the command buffer, not the process. */
   fprintf(stderr, "(the errors logged by this test are expected)\n");
   KgpCommandBuffer bad = kgpStartCommandRecording(queue);
   CHECK(bad);
   kgpCopyToTexture(bad, tex, 1, 0, mid_gpu, COPY_W * 4);    /* no such mip */
   CHECK(!kgpSubmit(queue, &bad, 1, NULL, 0));
   bad = kgpStartCommandRecording(queue);
   CHECK(bad);
   kgpCopyFromTexture(bad, tex, 0, 0, mid_gpu, COPY_W * 4 - 4);   /* short row */
   CHECK(!kgpSubmit(queue, &bad, 1, NULL, 0));
   bad = kgpStartCommandRecording(queue);
   CHECK(bad);
   kgpMemCpy(bad, NULL, mid_gpu, 16);
   CHECK(!kgpSubmit(queue, &bad, 1, NULL, 0));

   int ret = 0;
   if (kgp_test_offline()) {
      printf("copy: %s recorded and submitted (offline, results not checked)\n",
             mode);
   } else {
      ret |= copy_check("copied back", out, COPY_OUT_ROW_TEXELS);
      ret |= copy_check("shader-fetched", fetched, COPY_W);
      if (ret == 0)
         printf("copy: %s: memory, to-texture and from-texture copies of %u "
                "texels correct, and a shader reads the same\n",
                mode, COPY_W * COPY_H);
   }

   kgpDestroySemaphore(dev, done);
   kgpDestroyPipeline(dev, fetch);
   kgpDestroyTexture(dev, tex);
   kgpFree(dev, root);
   kgpFree(dev, heap);
   kgpFree(dev, fetched);
   kgpFree(dev, out);
   kgpFree(dev, tex_gpu);
   kgpFree(dev, mid_gpu);
   kgpFree(dev, src);
   kgpDestroyDevice(dev);
   return ret;
}
