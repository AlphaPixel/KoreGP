/* A shader that needs shader-local memory, a constant-data buffer and
 * gl_NumWorkGroups (shaders/scratch.comp). Run twice from one pipeline, in
 * two command buffers submitted together, to cover multi-buffer submits.
 */
#include "kgp_test.h"

#include "scratch_oracle.h"
#include "scratch_spv.h"

int
main(void)
{
   KgpDevice dev = kgpCreateDevice(0);
   CHECK(dev);
   KgpQueue queue = kgpGetQueue(dev);

   uint32_t *src = kgpMalloc(dev, SCRATCH_COUNT * sizeof(uint32_t), 16,
                             KGP_MEMORY_DEFAULT);
   uint32_t *dst[2];
   struct scratch_root *roots =
      kgpMalloc(dev, 2 * sizeof(*roots), 16, KGP_MEMORY_DEFAULT);
   CHECK(src && roots);
   struct scratch_root *roots_gpu = kgpHostToDevicePointer(dev, roots);

   for (uint32_t i = 0; i < SCRATCH_COUNT; i++)
      src[i] = scratch_input(i);

   for (unsigned r = 0; r < 2; r++) {
      dst[r] = kgpMalloc(dev, SCRATCH_COUNT * sizeof(uint32_t), 16,
                         KGP_MEMORY_READBACK);
      CHECK(dst[r]);
      roots[r] = (struct scratch_root) {
         .src = (uintptr_t)kgpHostToDevicePointer(dev, src),
         .dst = (uintptr_t)kgpHostToDevicePointer(dev, dst[r]),
         .count = SCRATCH_COUNT,
      };
   }

   KgpPipeline pipeline = kgpCreateComputePipeline(
      dev, (KgpByteSpan) { scratch_spv, sizeof(scratch_spv) }, NULL);
   CHECK(pipeline);

   KgpSemaphore done = kgpCreateSemaphore(dev, 5);
   CHECK(done);
   /* Already at 5: waiting for an earlier point returns at once. */
   CHECK(kgpWaitSemaphore(dev, done, 3, 0));

   KgpCommandBuffer cbs[2];
   for (unsigned r = 0; r < 2; r++) {
      cbs[r] = kgpStartCommandRecording(queue);
      CHECK(cbs[r]);
      kgpSetPipeline(cbs[r], pipeline);
      kgpDispatch(cbs[r], &roots_gpu[r], (KgpUvec3) { SCRATCH_GROUPS, 1, 1 });
   }
   CHECK(kgpSubmit(queue, cbs, 2, done, 6));
   CHECK(kgpWaitSemaphore(dev, done, 6, KGP_TEST_TIMEOUT_NS));

   if (kgp_test_offline()) {
      printf("scratch: recorded and submitted (offline, results not checked)\n");
   } else {
      for (unsigned r = 0; r < 2; r++) {
         for (uint32_t i = 0; i < SCRATCH_COUNT; i++) {
            const uint32_t expect = scratch_expected(i, SCRATCH_GROUPS);
            if (dst[r][i] != expect) {
               fprintf(stderr, "scratch: dst%u[%u] = %u, expected %u\n",
                       r, i, dst[r][i], expect);
               return 1;
            }
         }
      }
      printf("scratch: 2 x %u elements correct\n", SCRATCH_COUNT);
   }

   kgpDestroySemaphore(dev, done);
   kgpDestroyPipeline(dev, pipeline);
   kgpFree(dev, roots);
   kgpFree(dev, dst[1]);
   kgpFree(dev, dst[0]);
   kgpFree(dev, src);
   kgpDestroyDevice(dev);
   return 0;
}
