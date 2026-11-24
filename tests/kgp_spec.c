/* Specialization constants: the same shader as two pipelines, one with the
 * module's defaults and one with all three constants replaced.
 */
#include "kgp_test.h"

#include "spec_spv.h"
#include "tri_ps_spv.h"
#include "tri_vs_spv.h"

#include <string.h>

int
main(void)
{
   KgpDevice dev = kgpCreateDevice(0);
   CHECK(dev);
   KgpQueue queue = kgpGetQueue(dev);
   const KgpByteSpan spv = { spec_spv, sizeof(spec_spv) };

   float *out = kgpMalloc(dev, 2 * sizeof(float), 16, KGP_MEMORY_READBACK);
   CHECK(out);
   out[0] = out[1] = 0.0f;

   const float bias = 2.25f;
   uint32_t bias_bits;
   memcpy(&bias_bits, &bias, sizeof(bias_bits));
   const KgpSpecConstant values[] = {
      { .id = 9, .value = 1 },           /* NEGATE */
      { .id = 3, .value = 4 },           /* SCALE */
      { .id = 5, .value = bias_bits },   /* BIAS */
   };
   KgpPipeline defaults = kgpCreateComputePipeline(dev, spv, NULL);
   KgpPipeline special = kgpCreateComputePipeline(
      dev, spv, &(KgpSpecialization) { values, 3 });
   CHECK(defaults && special);

   fprintf(stderr, "(the errors logged by this test are expected)\n");
   const KgpSpecConstant unknown[] = { { .id = 3, .value = 2 }, { .id = 77 } };
   CHECK(!kgpCreateComputePipeline(dev, spv,
                                   &(KgpSpecialization) { unknown, 2 }));
   CHECK(!kgpCreateComputePipeline(dev, spv,
                                   &(KgpSpecialization) { NULL, 1 }));
   /* Raster: one list for both stages, checked against both. */
   const KgpRasterDesc raster = {
      .colorTargetCount = 1, .colorFormats = { KGP_FORMAT_RGBA8_UNORM },
   };
   CHECK(!kgpCreateGraphicsPipeline(
            dev, (KgpByteSpan) { tri_vs_spv, sizeof(tri_vs_spv) },
            (KgpByteSpan) { tri_ps_spv, sizeof(tri_ps_spv) }, &raster,
            &(KgpSpecialization) { unknown, 2 }));

   KgpSemaphore done = kgpCreateSemaphore(dev, 0);
   CHECK(done);
   KgpCommandBuffer cb = kgpStartCommandRecording(queue);
   CHECK(cb);
   kgpSetPipeline(cb, defaults);
   kgpDispatch(cb, kgpHostToDevicePointer(dev, &out[0]), (KgpUvec3) { 1, 1, 1 });
   kgpSetPipeline(cb, special);
   kgpDispatch(cb, kgpHostToDevicePointer(dev, &out[1]), (KgpUvec3) { 1, 1, 1 });
   CHECK(kgpSubmit(queue, &cb, 1, done, 1));
   CHECK(kgpWaitSemaphore(dev, done, 1, KGP_TEST_TIMEOUT_NS));

   int ret = 0;
   if (kgp_test_offline()) {
      printf("spec: recorded and submitted (offline, results not checked)\n");
   } else if (out[0] != 10.5f || out[1] != -42.25f) {
      fprintf(stderr, "spec: got %f and %f, expected 10.5 and -42.25\n",
              out[0], out[1]);
      ret = 1;
   } else {
      printf("spec: defaults and three replaced constants both correct\n");
   }

   kgpDestroySemaphore(dev, done);
   kgpDestroyPipeline(dev, special);
   kgpDestroyPipeline(dev, defaults);
   kgpFree(dev, out);
   kgpDestroyDevice(dev);
   return ret;
}
