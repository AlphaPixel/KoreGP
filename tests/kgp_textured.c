/* A pixel shader reads the texture heap (plan N3, task R8): a texture
 * uploaded from the CPU is drawn 1:1 over a render target by a full-target
 * quad whose pixel shader fetches heap entry SLOT, and the target is read
 * back and compared with what was uploaded.
 *
 * What it adds to kgp_triangle: the texture pool bound on the 3D engine,
 * NAK's bindless texture fetch in a pixel shader, gl_FragCoord.
 * No Vulkan twin: the heap has no Vulkan spelling (as for kgp_heap).
 */
#include "kgp_test.h"

#include "copy_pattern.h"
#include "tex_ps_spv.h"
#include "tri_scene.h"
#include "tri_vs_spv.h"

#include <string.h>

#define SPAN(a) ((KgpByteSpan) { a, sizeof(a) })
#define HEAP_ENTRIES 4u
#define SLOT 3u

struct tex_ps_root {
   uint32_t texture_index;
   uint32_t pad[15];
};

int
main(void)
{
   KgpDevice dev = kgpCreateDevice(0);
   CHECK(dev);
   KgpQueue queue = kgpGetQueue(dev);

   const KgpTextureDesc desc = {
      .format = KGP_FORMAT_RGBA8_UNORM,
      .width = COPY_W,
      .height = COPY_H,
   };
   KgpSizeAlign sa;
   CHECK(kgpTextureSizeAlign(dev, &desc, &sa));

   /* Source: CPU-mapped, filled with the copy tests' pattern. */
   void *src_mem = kgpMalloc(dev, sa.size, sa.align, KGP_MEMORY_DEFAULT);
   void *dst_gpu = kgpMalloc(dev, sa.size, sa.align, KGP_MEMORY_GPU);
   uint32_t *out = kgpMalloc(dev, COPY_BYTES, 64, KGP_MEMORY_READBACK);
   CHECK(src_mem && dst_gpu && out);
   memset(out, 0, COPY_BYTES);
   KgpTexture src = kgpCreateTexture(dev, &desc,
                                     kgpHostToDevicePointer(dev, src_mem));
   KgpTexture dst = kgpCreateTexture(dev, &desc, dst_gpu);
   CHECK(src && dst);
   uint32_t *pixels = malloc(COPY_BYTES);
   CHECK(pixels);
   copy_fill(pixels);
   CHECK(kgpWriteTexture(src, src_mem, 0, 0, pixels, COPY_W * 4));
   free(pixels);

   KgpDescriptor256 *heap =
      kgpMalloc(dev, HEAP_ENTRIES * sizeof(*heap), 64, KGP_MEMORY_DEFAULT);
   CHECK(heap);
   memset(heap, 0, HEAP_ENTRIES * sizeof(*heap));
   CHECK(kgpTextureViewDescriptor(src, NULL, &heap[SLOT]));

   /* One quad over the whole target, white, one instance. */
   struct tri_vertex *vertices =
      kgpMalloc(dev, 4 * sizeof(*vertices), 64, KGP_MEMORY_DEFAULT);
   uint32_t *indices = kgpMalloc(dev, 6 * sizeof(*indices), 64,
                                 KGP_MEMORY_DEFAULT);
   struct tri_vs_root *vs_root =
      kgpMalloc(dev, sizeof(*vs_root), 64, KGP_MEMORY_DEFAULT);
   struct tex_ps_root *ps_root =
      kgpMalloc(dev, sizeof(*ps_root), 64, KGP_MEMORY_DEFAULT);
   CHECK(vertices && indices && vs_root && ps_root);
   tri_fill(vertices, indices);
   static const float corners[4][2] = {
      { -1.0f, -1.0f }, { 1.0f, -1.0f }, { 1.0f, 1.0f }, { -1.0f, 1.0f },
   };
   for (unsigned i = 0; i < 4; i++) {
      vertices[i] = (struct tri_vertex) {
         .pos = { corners[i][0], corners[i][1] },
         .color = { 1.0f, 1.0f, 1.0f, 1.0f },
      };
   }
   *vs_root = (struct tri_vs_root) {
      .vertices = (uintptr_t)kgpHostToDevicePointer(dev, vertices),
   };
   *ps_root = (struct tex_ps_root) { .texture_index = SLOT };

   const KgpRasterDesc raster = {
      .colorTargetCount = 1,
      .colorFormats = { KGP_FORMAT_RGBA8_UNORM },
   };
   KgpPipeline pipeline = kgpCreateGraphicsPipeline(
      dev, SPAN(tri_vs_spv), SPAN(tex_ps_spv), &raster, NULL);
   KgpSemaphore done = kgpCreateSemaphore(dev, 0);
   CHECK(pipeline && done);

   KgpCommandBuffer cb = kgpStartCommandRecording(queue);
   CHECK(cb);
   kgpSetActiveTextureHeapPtr(cb, kgpHostToDevicePointer(dev, heap),
                              HEAP_ENTRIES);
   kgpBeginRenderPass(cb, &(KgpRenderPassDesc) {
      .colorTargetCount = 1,
      .colorTargets = { { .texture = dst, .load = KGP_LOAD_CLEAR } },
   });
   kgpSetPipeline(cb, pipeline);
   kgpDrawIndexedInstanced(cb, kgpHostToDevicePointer(dev, vs_root),
                           kgpHostToDevicePointer(dev, ps_root),
                           kgpHostToDevicePointer(dev, indices), 6, 1);
   kgpEndRenderPass(cb);
   kgpBarrier(cb, KGP_STAGE_RASTER_COLOR_OUT, KGP_STAGE_TRANSFER,
              KGP_HAZARD_NONE);
   kgpCopyFromTexture(cb, dst, 0, 0, kgpHostToDevicePointer(dev, out),
                      COPY_W * 4);
   CHECK(kgpSubmit(queue, &cb, 1, done, 1));
   CHECK(kgpWaitSemaphore(dev, done, 1, KGP_TEST_TIMEOUT_NS));

   /* A pipeline made for another format is refused at the draw. */
   fprintf(stderr, "(the errors logged by this test are expected)\n");
   KgpRasterDesc other = raster;
   other.colorFormats[0] = KGP_FORMAT_RGBA16_FLOAT;
   KgpPipeline wrong = kgpCreateGraphicsPipeline(
      dev, SPAN(tri_vs_spv), SPAN(tex_ps_spv), &other, NULL);
   CHECK(wrong);
   KgpCommandBuffer bad = kgpStartCommandRecording(queue);
   CHECK(bad);
   kgpBeginRenderPass(bad, &(KgpRenderPassDesc) {
      .colorTargetCount = 1,
      .colorTargets = { { .texture = dst } },
   });
   kgpSetPipeline(bad, wrong);
   kgpDrawIndexedInstanced(bad, kgpHostToDevicePointer(dev, vs_root),
                           kgpHostToDevicePointer(dev, ps_root),
                           kgpHostToDevicePointer(dev, indices), 6, 1);
   CHECK(!kgpSubmit(queue, &bad, 1, NULL, 0));

   int ret = 0;
   if (kgp_test_offline()) {
      printf("textured: recorded and submitted (offline, results not "
             "checked)\n");
   } else {
      ret = copy_check("drawn from the heap", out, COPY_W);
      if (ret == 0)
         printf("textured: a pixel shader fetched %u texels through the "
                "heap correctly\n", COPY_W * COPY_H);
   }

   kgpDestroySemaphore(dev, done);
   kgpDestroyPipeline(dev, wrong);
   kgpDestroyPipeline(dev, pipeline);
   kgpDestroyTexture(dev, dst);
   kgpDestroyTexture(dev, src);
   kgpFree(dev, ps_root);
   kgpFree(dev, vs_root);
   kgpFree(dev, indices);
   kgpFree(dev, vertices);
   kgpFree(dev, heap);
   kgpFree(dev, out);
   kgpFree(dev, dst_gpu);
   kgpFree(dev, src_mem);
   kgpDestroyDevice(dev);
   return ret;
}
