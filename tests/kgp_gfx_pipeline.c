/* Graphics pipelines (plan N3, task R2): a vertex and a pixel shader go
 * through SPIR-V -> NIR -> cross-stage linking -> NAK and are uploaded.
 * Nothing is drawn yet, so this needs no GPU work and checks the same thing
 * offline and on hardware: creation succeeds for a good pair and fails,
 * with an error and not a crash, for bad input.
 */
#include "kgp_test.h"

#include "bad_ps_push_spv.h"
#include "saxpy_spv.h"
#include "tri_ps_spv.h"
#include "tri_vs_spv.h"

#define SPAN(a) ((KgpByteSpan) { a, sizeof(a) })

int
main(void)
{
   KgpDevice dev = kgpCreateDevice(0);
   CHECK(dev);
   KgpQueue queue = kgpGetQueue(dev);

   const KgpRasterDesc raster = {
      .topology = KGP_TOPOLOGY_TRIANGLE_LIST,
      .cull = KGP_CULL_NONE,
      .colorTargetCount = 1,
      .colorFormats = { KGP_FORMAT_RGBA8_UNORM },
   };

   KgpPipeline pipeline = kgpCreateGraphicsPipeline(
      dev, SPAN(tri_vs_spv), SPAN(tri_ps_spv), &raster, NULL);
   CHECK(pipeline);

   fprintf(stderr, "(the errors logged by this test are expected)\n");

   /* Stages in the wrong slots, a compute shader, no raster desc. */
   CHECK(!kgpCreateGraphicsPipeline(dev, SPAN(tri_ps_spv), SPAN(tri_vs_spv),
                                    &raster, NULL));
   CHECK(!kgpCreateGraphicsPipeline(dev, SPAN(saxpy_spv), SPAN(tri_ps_spv),
                                    &raster, NULL));
   CHECK(!kgpCreateGraphicsPipeline(dev, SPAN(tri_vs_spv), SPAN(tri_ps_spv),
                                    NULL, NULL));

   /* A pixel shader reading the vertex stage's root pointer. */
   CHECK(!kgpCreateGraphicsPipeline(dev, SPAN(tri_vs_spv),
                                    SPAN(bad_ps_push_spv), &raster, NULL));

   /* Raster descs that are not implemented or not meaningful. */
   KgpRasterDesc bad = raster;
   bad.colorTargetCount = 0;
   CHECK(!kgpCreateGraphicsPipeline(dev, SPAN(tri_vs_spv), SPAN(tri_ps_spv),
                                    &bad, NULL));
   bad = raster;
   bad.depthFormat = KGP_FORMAT_R32_FLOAT;
   CHECK(!kgpCreateGraphicsPipeline(dev, SPAN(tri_vs_spv), SPAN(tri_ps_spv),
                                    &bad, NULL));

   /* A graphics pipeline cannot be dispatched. */
   KgpCommandBuffer cb = kgpStartCommandRecording(queue);
   CHECK(cb);
   kgpSetPipeline(cb, pipeline);
   kgpDispatch(cb, NULL, (KgpUvec3) { 1, 1, 1 });
   CHECK(!kgpSubmit(queue, &cb, 1, NULL, 0));

   kgpDestroyPipeline(dev, pipeline);
   kgpDestroyDevice(dev);
   printf("gfx_pipeline: a vertex + pixel pipeline compiles, links and "
          "uploads; bad input is refused\n");
   return 0;
}
