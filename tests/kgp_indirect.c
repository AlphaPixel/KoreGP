/* GPU-driven dispatch: one compute shader works out how many groups saxpy
 * needs and writes the counts to memory; saxpy is then dispatched
 * indirectly from that memory, in the same command buffer. If the counts
 * do not arrive, saxpy covers the wrong number of elements and the check
 * fails.
 */
#include "kgp_test.h"

#include "genargs_spv.h"
#include "saxpy_spv.h"

#define COUNT 100000u

struct saxpy_root {
   uint64_t x;
   uint64_t y;
   float a;
   uint32_t count;
};

/* Matches GenArgsRoot in shaders/genargs.comp (scalar layout). */
struct genargs_root {
   uint64_t args;
   uint32_t count;
   uint32_t pad;
};

int
main(void)
{
   KgpDevice dev = kgpCreateDevice(0);
   CHECK(dev);
   KgpQueue queue = kgpGetQueue(dev);

   float *x = kgpMalloc(dev, COUNT * sizeof(float), 16, KGP_MEMORY_DEFAULT);
   float *y = kgpMalloc(dev, COUNT * sizeof(float), 16, KGP_MEMORY_READBACK);
   struct saxpy_root *saxpy_root =
      kgpMalloc(dev, sizeof(*saxpy_root), 16, KGP_MEMORY_DEFAULT);
   struct genargs_root *genargs_root =
      kgpMalloc(dev, sizeof(*genargs_root), 16, KGP_MEMORY_DEFAULT);
   /* Written by the GPU, read by the GPU; the CPU only looks afterwards. */
   KgpUvec3 *args = kgpMalloc(dev, sizeof(*args), 16, KGP_MEMORY_READBACK);
   CHECK(x && y && saxpy_root && genargs_root && args);
   *args = (KgpUvec3) { 0, 0, 0 };

   const float a = 2.5f;
   for (uint32_t i = 0; i < COUNT; i++) {
      x[i] = (float)i;
      y[i] = 1.0f + (float)(i % 7);
   }
   *saxpy_root = (struct saxpy_root) {
      .x = (uintptr_t)kgpHostToDevicePointer(dev, x),
      .y = (uintptr_t)kgpHostToDevicePointer(dev, y),
      .a = a,
      .count = COUNT,
   };
   *genargs_root = (struct genargs_root) {
      .args = (uintptr_t)kgpHostToDevicePointer(dev, args),
      .count = COUNT,
   };

   KgpPipeline genargs = kgpCreateComputePipeline(
      dev, (KgpByteSpan) { genargs_spv, sizeof(genargs_spv) }, NULL);
   KgpPipeline saxpy = kgpCreateComputePipeline(
      dev, (KgpByteSpan) { saxpy_spv, sizeof(saxpy_spv) }, NULL);
   KgpSemaphore done = kgpCreateSemaphore(dev, 0);
   CHECK(genargs && saxpy && done);

   KgpCommandBuffer cb = kgpStartCommandRecording(queue);
   CHECK(cb);
   kgpSetPipeline(cb, genargs);
   kgpDispatch(cb, kgpHostToDevicePointer(dev, genargs_root),
               (KgpUvec3) { 1, 1, 1 });
   kgpBarrier(cb, KGP_STAGE_COMPUTE, KGP_STAGE_COMPUTE,
              KGP_HAZARD_DRAW_ARGUMENTS);
   kgpSetPipeline(cb, saxpy);
   kgpDispatchIndirect(cb, kgpHostToDevicePointer(dev, saxpy_root),
                       kgpHostToDevicePointer(dev, args));
   CHECK(kgpSubmit(queue, &cb, 1, done, 1));
   CHECK(kgpWaitSemaphore(dev, done, 1, KGP_TEST_TIMEOUT_NS));

   if (kgp_test_offline()) {
      printf("indirect: recorded and submitted (offline, results not checked)\n");
   } else {
      CHECK(args->x == (COUNT + 63) / 64 && args->y == 1 && args->z == 1);
      for (uint32_t i = 0; i < COUNT; i++) {
         const float expect = a * (float)i + (1.0f + (float)(i % 7));
         if (y[i] != expect) {
            fprintf(stderr, "indirect: y[%u] = %f, expected %f\n",
                    i, y[i], expect);
            return 1;
         }
      }
      printf("indirect: %u groups from the GPU, %u elements correct\n",
             args->x, COUNT);
   }

   kgpDestroySemaphore(dev, done);
   kgpDestroyPipeline(dev, saxpy);
   kgpDestroyPipeline(dev, genargs);
   kgpFree(dev, args);
   kgpFree(dev, genargs_root);
   kgpFree(dev, saxpy_root);
   kgpFree(dev, y);
   kgpFree(dev, x);
   kgpDestroyDevice(dev);
   return 0;
}
