/* GPU time: wall clock from submit to the semaphore, best and median of
 * several runs. Hardware only (on the DRM shim nothing executes and the
 * numbers mean nothing).
 *
 *    bench_gpu [mode] [elements] [dispatches]
 *
 *    saxpy   (default) y = a*x + y over `elements` (default 16M) floats.
 *            Memory-bound: measures the card's memory traffic.
 *    loop    64 iterations of arithmetic on eight root fields per element
 *            (default 1M), one store. Arithmetic-bound.
 *
 * `dispatches` (default 1) records that many identical dispatches in the one
 * command buffer, each over elements / dispatches elements, so the total
 * work stays the same and the per-dispatch cost shows.
 *
 * The arrays are GPU-only memory and are not initialized: the work costs the same whatever the bits are. Nothing is
 * checked; correctness is the tests' job.
 */
#include "kgp_test.h"

#include "rootloop_spv.h"
#include "saxpy_spv.h"

#include <string.h>
#include <time.h>

#define RUNS 21

/* Both padded to 64 bytes; the structs sit alone in their allocation. */
struct saxpy_root {
   uint64_t x;
   uint64_t y;
   float a;
   uint32_t count;
   uint32_t pad[10];
};
struct loop_root {
   uint64_t dst;
   uint32_t count;
   uint32_t iters;
   float c[8];
   uint32_t pad[4];
};
_Static_assert(sizeof(struct saxpy_root) == 64 &&
               sizeof(struct loop_root) == 64, "root structs are 64 bytes");

static double
now_ms(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static int
cmp_double(const void *a, const void *b)
{
   const double x = *(const double *)a, y = *(const double *)b;
   return x < y ? -1 : x > y;
}

int
main(int argc, char **argv)
{
   const bool loop = argc > 1 && strcmp(argv[1], "loop") == 0;
   const uint32_t elements =
      argc > 2 ? strtoul(argv[2], NULL, 0) : loop ? 1u << 20 : 1u << 24;
   const uint32_t dispatches = argc > 3 ? strtoul(argv[3], NULL, 0) : 1;
   CHECK(dispatches > 0 && dispatches <= 65536 && elements >= dispatches);
   const uint32_t count = elements / dispatches;

   KgpDevice dev = kgpCreateDevice(0);
   CHECK(dev);
   KgpQueue queue = kgpGetQueue(dev);

   void *x = kgpMalloc(dev, (size_t)count * sizeof(float), 4096, KGP_MEMORY_GPU);
   void *y = kgpMalloc(dev, (size_t)count * sizeof(float), 4096, KGP_MEMORY_GPU);
   void *root = kgpMalloc(dev, 64, 64, KGP_MEMORY_DEFAULT);
   CHECK(x && y && root);
   if (loop) {
      *(struct loop_root *)root = (struct loop_root) {
         .dst = (uintptr_t)y, .count = count, .iters = 64,
         /* Keeps x bounded: no denormals or infinities to skew timing. */
         .c = { 0.5f, 0.25f, -0.5f, 0.125f, 0.75f, -0.25f, -0.75f, 0.5f },
      };
   } else {
      *(struct saxpy_root *)root = (struct saxpy_root) {
         .x = (uintptr_t)x, .y = (uintptr_t)y, .a = 0.5f, .count = count,
      };
   }
   void *root_gpu = kgpHostToDevicePointer(dev, root);

   KgpPipeline pipeline = kgpCreateComputePipeline(
      dev, loop ? (KgpByteSpan) { rootloop_spv, sizeof(rootloop_spv) }
                : (KgpByteSpan) { saxpy_spv, sizeof(saxpy_spv) }, NULL);
   KgpSemaphore done = kgpCreateSemaphore(dev, 0);
   CHECK(pipeline && done);

   double ms[RUNS];
   for (uint32_t i = 0; i < RUNS; i++) {
      KgpCommandBuffer cb = kgpStartCommandRecording(queue);
      kgpSetPipeline(cb, pipeline);
      for (uint32_t d = 0; d < dispatches; d++)
         kgpDispatch(cb, root_gpu, (KgpUvec3) { (count + 63) / 64, 1, 1 });

      const double t0 = now_ms();
      CHECK(kgpSubmit(queue, &cb, 1, done, i + 1));
      CHECK(kgpWaitSemaphore(dev, done, i + 1, 6 * KGP_TEST_TIMEOUT_NS));
      ms[i] = now_ms() - t0;
   }

   qsort(ms, RUNS, sizeof(ms[0]), cmp_double);
   const char *debug = getenv("KGP_DEBUG");
   printf("%s %u elements in %u dispatch%s [KGP_DEBUG=%s]: "
          "best %.3f ms, median %.3f ms\n", loop ? "loop" : "saxpy",
          count * dispatches, dispatches, dispatches == 1 ? "" : "es",
          debug ? debug : "", ms[0], ms[RUNS / 2]);

   kgpDestroySemaphore(dev, done);
   kgpDestroyPipeline(dev, pipeline);
   kgpFree(dev, root);
   kgpFree(dev, y);
   kgpFree(dev, x);
   kgpDestroyDevice(dev);
   return 0;
}
