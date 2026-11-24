/* Presentation (R10): the triangle scene drawn straight into swapchain
 * textures and presented.
 *
 *    kgp_present            headless swapchain, 64x64, three frames
 *    kgp_present display    the first connected display at its own mode,
 *                           for a couple of seconds; needs a window opened
 *                           with modesetting on (w0.sh present)
 *
 * The first frame is read back from the swapchain texture and checked, so
 * the test also says whether memory shared in from the hidden Vulkan device
 * is the memory KoreGP rendered to. What the monitor shows it cannot check:
 * two red rectangles on black, side by side.
 */
#include "kgp_test.h"
#include "tri_scene.h"

#include "tri_ps_spv.h"
#include "tri_vs_spv.h"

#include <string.h>

#define SPAN(a) ((KgpByteSpan) { a, sizeof(a) })

int
main(int argc, char **argv)
{
   const bool display = argc > 1 && strcmp(argv[1], "display") == 0;
   const unsigned frames = display ? 120 : 3;

   KgpDevice dev = kgpCreateDevice(0);
   CHECK(dev);
   KgpQueue queue = kgpGetQueue(dev);

   KgpSwapchainInfo info;
   KgpSwapchain swapchain = kgpCreateSwapchain(
      dev, &(KgpSwapchainDesc) {
         .target = display ? KGP_PRESENT_DISPLAY : KGP_PRESENT_HEADLESS,
         .width = TRI_W, .height = TRI_H,
      }, &info);
   CHECK(swapchain);
   CHECK(info.format == KGP_FORMAT_BGRA8_UNORM && info.width && info.height);

   struct tri_vertex *vertices =
      kgpMalloc(dev, TRI_VERTEX_COUNT * sizeof(*vertices), 64,
                KGP_MEMORY_DEFAULT);
   uint32_t *indices =
      kgpMalloc(dev, TRI_INDEX_COUNT * sizeof(*indices), 64,
                KGP_MEMORY_DEFAULT);
   struct tri_vs_root *vs_root =
      kgpMalloc(dev, sizeof(*vs_root), 64, KGP_MEMORY_DEFAULT);
   struct tri_ps_root *ps_root =
      kgpMalloc(dev, sizeof(*ps_root), 64, KGP_MEMORY_DEFAULT);
   uint32_t *out = kgpMalloc(dev, (size_t)info.width * info.height * 4, 64,
                             KGP_MEMORY_READBACK);
   CHECK(vertices && indices && vs_root && ps_root && out);
   tri_fill(vertices, indices);
   *vs_root = (struct tri_vs_root) {
      .vertices = (uintptr_t)kgpHostToDevicePointer(dev, vertices),
      .instance_step = TRI_INSTANCE_STEP,
   };
   *ps_root = (struct tri_ps_root) { .tint = TRI_TINT_RGBA };
   memset(out, 0, (size_t)info.width * info.height * 4);

   const KgpRasterDesc raster = {
      .colorTargetCount = 1,
      .colorFormats = { info.format },
   };
   KgpPipeline pipeline = kgpCreateGraphicsPipeline(
      dev, SPAN(tri_vs_spv), SPAN(tri_ps_spv), &raster, NULL);
   KgpSemaphore done = kgpCreateSemaphore(dev, 0);
   CHECK(pipeline && done);

   for (unsigned f = 0; f < frames; f++) {
      KgpTexture target = kgpAcquireSwapchainTexture(swapchain);
      CHECK(target);
      const KgpRenderPassDesc pass = {
         .colorTargetCount = 1,
         .colorTargets = { {
            .texture = target,
            .load = KGP_LOAD_CLEAR,
            .clearColor = TRI_CLEAR_RGBA,
         } },
      };
      KgpCommandBuffer cb = kgpStartCommandRecording(queue);
      CHECK(cb);
      kgpBeginRenderPass(cb, &pass);
      kgpSetPipeline(cb, pipeline);
      kgpDrawIndexedInstanced(cb, kgpHostToDevicePointer(dev, vs_root),
                              kgpHostToDevicePointer(dev, ps_root),
                              kgpHostToDevicePointer(dev, indices),
                              TRI_INDEX_COUNT, TRI_INSTANCE_COUNT);
      kgpEndRenderPass(cb);
      if (f == 0) {
         kgpBarrier(cb, KGP_STAGE_RASTER_COLOR_OUT, KGP_STAGE_TRANSFER,
                    KGP_HAZARD_NONE);
         kgpCopyFromTexture(cb, target, 0, 0,
                            kgpHostToDevicePointer(dev, out),
                            (size_t)info.width * 4);
      }
      CHECK(kgpSubmit(queue, &cb, 1, done, f + 1));
      /* kgpPresent does not wait for the rendering. */
      CHECK(kgpWaitSemaphore(dev, done, f + 1, KGP_TEST_TIMEOUT_NS));
      CHECK(kgpPresent(swapchain));
   }

   fprintf(stderr, "(the errors logged by this test are expected)\n");
   CHECK(!kgpPresent(swapchain));                     /* nothing acquired */
   CHECK(kgpAcquireSwapchainTexture(swapchain));
   CHECK(!kgpAcquireSwapchainTexture(swapchain));     /* twice */
   CHECK(!kgpCreateSwapchain(dev, &(KgpSwapchainDesc) {
            .target = KGP_PRESENT_HEADLESS }, &info));   /* no size */

   int ret = 0;
   if (kgp_test_offline()) {
      printf("present: %u frames recorded and presented (offline, results "
             "not checked)\n", frames);
   } else {
      /* The scene is defined on a 64x64 grid: sample each cell's centre.
       * The texture is BGRA, the scene's colours are RGBA.
       */
      for (uint32_t gy = 0; gy < TRI_H && ret == 0; gy++) {
         for (uint32_t gx = 0; gx < TRI_W; gx++) {
            const uint32_t x = (2 * gx + 1) * info.width / (2 * TRI_W);
            const uint32_t y = (2 * gy + 1) * info.height / (2 * TRI_H);
            const uint32_t bgra = out[(size_t)y * info.width + x];
            const uint32_t got = (bgra & 0xff00ff00u) |
                                 ((bgra >> 16) & 0xffu) |
                                 ((bgra & 0xffu) << 16);
            const uint32_t want = tri_expected(gx, gy, TRI_SCENE_PLAIN);
            if (got != want) {
               fprintf(stderr, "present: pixel (%u, %u) is %08x, expected "
                       "%08x\n", x, y, got, want);
               ret = 1;
               break;
            }
         }
      }
      if (ret == 0)
         printf("present: %ux%u swapchain texture correct, %u frames "
                "presented%s\n", info.width, info.height, frames,
                display ? " on the display" : " (headless)");
   }

   kgpDestroySemaphore(dev, done);
   kgpDestroyPipeline(dev, pipeline);
   kgpDestroySwapchain(swapchain);
   kgpFree(dev, out);
   kgpFree(dev, ps_root);
   kgpFree(dev, vs_root);
   kgpFree(dev, indices);
   kgpFree(dev, vertices);
   kgpDestroyDevice(dev);
   return ret;
}
