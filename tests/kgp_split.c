/* Split barriers: kgpSignalAfter and kgpWaitBefore on 64-bit values in GPU
 * memory.
 *
 *  1. One command buffer: a dispatch, then signals (SET; MAX below and
 *     above what is there), a wait on a value signalled just before, and a
 *     second dispatch behind it.
 *  2. The CPU releases the GPU: a command buffer that starts with a wait on
 *     a value nothing has written must not finish until the CPU writes it.
 *
 * No Vulkan twin and no comparison with NVK: Vulkan has no spelling for
 * these and NVK does not use the channel's semaphore methods
 * (src/koregp/kgp_barrier.c). On the DRM shim this only records and
 * submits.
 *
 * "never" is not a test but an observation: what the kernel does with a
 * wait nothing satisfies for longer than its job timeout. It prints what
 * it sees and exits 0 unless it cannot even set up; the hardware suite
 * runs it last of the correctness steps and then checks that a new
 * process still works.
 */
#include "kgp_test.h"

#include <string.h>
#include <time.h>

#include "saxpy_spv.h"

#define COUNT 4096u

/* Matches SaxpyRoot in shaders/saxpy.comp (scalar layout). */
struct saxpy_root {
   uint64_t x;
   uint64_t y;
   float a;
   uint32_t count;
};

static int
check_y(const float *y, float a, unsigned times)
{
   for (uint32_t i = 0; i < COUNT; i++) {
      const float expect = (float)times * a * (float)i + 1.0f;
      if (y[i] != expect) {
         fprintf(stderr, "split: y[%u] = %f after %u dispatches, expected %f\n",
                 i, y[i], times, expect);
         return 1;
      }
   }
   return 0;
}

static double
now_s(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Waits on `done` reaching `value` for up to `seconds` and says what
 * happened.
 */
static bool
observe(KgpDevice dev, KgpSemaphore done, uint64_t value, unsigned seconds,
        const char *what)
{
   const double start = now_s();
   const bool ok = kgpWaitSemaphore(dev, done, value,
                                    (uint64_t)seconds * 1000 * 1000 * 1000);
   printf("never: %s: %s after %.2f s\n", what,
          ok ? "completed" : "not completed", now_s() - start);
   fflush(stdout);
   return ok;
}

static int
run_never(KgpDevice dev, KgpQueue queue, KgpPipeline pipeline,
          KgpSemaphore done, void *root_gpu, uint64_t *sem, uint64_t *sem_gpu,
          KgpUvec3 groups, bool offline)
{
   KgpCommandBuffer cb = kgpStartCommandRecording(queue);
   CHECK(cb);
   kgpWaitBefore(cb, KGP_STAGE_COMPUTE, &sem_gpu[3], 1, KGP_WAIT_EQUAL,
                 KGP_HAZARD_NONE);
   kgpSetPipeline(cb, pipeline);
   kgpDispatch(cb, root_gpu, groups);
   const bool submitted = kgpSubmit(queue, &cb, 1, done, 1);
   printf("never: submit of the waiting command buffer: %s\n",
          submitted ? "accepted" : "refused");
   if (offline || !submitted) {
      printf("never: recorded and submitted (offline, nothing observed)\n");
      return 0;
   }

   /* nouveau's job timeout is 10 s. */
   if (!observe(dev, done, 1, 15, "the waiting command buffer, value unwritten")) {
      sem[3] = 1;
      observe(dev, done, 1, 3, "the same, after the CPU wrote the value");
   }

   /* Is the queue still usable? */
   cb = kgpStartCommandRecording(queue);
   if (cb == NULL) {
      printf("never: no new command buffer on this device\n");
      return 0;
   }
   kgpSetPipeline(cb, pipeline);
   kgpDispatch(cb, root_gpu, groups);
   const bool again = kgpSubmit(queue, &cb, 1, done, 2);
   printf("never: a later submit on the same device: %s\n",
          again ? "accepted" : "refused");
   if (again)
      observe(dev, done, 2, 3, "the later command buffer");
   return 0;
}

int
main(int argc, char **argv)
{
   KgpDevice dev = kgpCreateDevice(0);
   CHECK(dev);
   KgpQueue queue = kgpGetQueue(dev);
   const bool offline = kgp_test_offline();

   float *x = kgpMalloc(dev, COUNT * sizeof(float), 16, KGP_MEMORY_DEFAULT);
   float *y = kgpMalloc(dev, COUNT * sizeof(float), 16, KGP_MEMORY_READBACK);
   struct saxpy_root *root =
      kgpMalloc(dev, sizeof(*root), 16, KGP_MEMORY_DEFAULT);
   uint64_t *sem = kgpMalloc(dev, 4 * sizeof(uint64_t), 64, KGP_MEMORY_DEFAULT);
   CHECK(x && y && root && sem);
   uint64_t *sem_gpu = kgpHostToDevicePointer(dev, sem);
   void *root_gpu = kgpHostToDevicePointer(dev, root);
   CHECK(sem_gpu && root_gpu);

   const float a = 2.0f;
   for (uint32_t i = 0; i < COUNT; i++) {
      x[i] = (float)i;
      y[i] = 1.0f;
   }
   *root = (struct saxpy_root) {
      .x = (uintptr_t)kgpHostToDevicePointer(dev, x),
      .y = (uintptr_t)kgpHostToDevicePointer(dev, y),
      .a = a,
      .count = COUNT,
   };
   for (unsigned i = 0; i < 4; i++)
      sem[i] = 0;

   KgpPipeline pipeline = kgpCreateComputePipeline(
      dev, (KgpByteSpan) { saxpy_spv, sizeof(saxpy_spv) }, NULL);
   KgpSemaphore done = kgpCreateSemaphore(dev, 0);
   CHECK(pipeline && done);
   const KgpUvec3 groups = { (COUNT + 63) / 64, 1, 1 };

   if (argc > 1 && strcmp(argv[1], "never") == 0) {
      /* No teardown: what destroying a device in this state does is not
       * what is being observed, and the process is about to exit.
       */
      return run_never(dev, queue, pipeline, done, root_gpu, sem, sem_gpu,
                       groups, offline);
   }

   /* 1 */
   KgpCommandBuffer cb = kgpStartCommandRecording(queue);
   CHECK(cb);
   kgpSetPipeline(cb, pipeline);
   kgpDispatch(cb, root_gpu, groups);
   kgpSignalAfter(cb, KGP_STAGE_COMPUTE, &sem_gpu[0], 7, KGP_SIGNAL_SET);
   kgpSignalAfter(cb, KGP_STAGE_COMPUTE, &sem_gpu[1], 7, KGP_SIGNAL_SET);
   kgpSignalAfter(cb, KGP_STAGE_COMPUTE, &sem_gpu[1], 5, KGP_SIGNAL_MAX);
   kgpSignalAfter(cb, KGP_STAGE_COMPUTE, &sem_gpu[2], 7, KGP_SIGNAL_SET);
   kgpSignalAfter(cb, KGP_STAGE_COMPUTE, &sem_gpu[2], 9, KGP_SIGNAL_MAX);
   kgpWaitBefore(cb, KGP_STAGE_COMPUTE, &sem_gpu[2], 8,
                 KGP_WAIT_GREATER_EQUAL, KGP_HAZARD_NONE);
   kgpDispatch(cb, root_gpu, groups);
   CHECK(kgpSubmit(queue, &cb, 1, done, 1));
   CHECK(kgpWaitSemaphore(dev, done, 1, KGP_TEST_TIMEOUT_NS));
   if (!offline) {
      if (sem[0] != 7 || sem[1] != 7 || sem[2] != 9) {
         fprintf(stderr, "split: signalled values are %llu, %llu, %llu; "
                 "expected 7, 7, 9\n", (unsigned long long)sem[0],
                 (unsigned long long)sem[1], (unsigned long long)sem[2]);
         return 1;
      }
      if (check_y(y, a, 2))
         return 1;
   }

   /* 2 */
   cb = kgpStartCommandRecording(queue);
   CHECK(cb);
   kgpWaitBefore(cb, KGP_STAGE_COMPUTE, &sem_gpu[3], 1, KGP_WAIT_EQUAL,
                 KGP_HAZARD_NONE);
   kgpSetPipeline(cb, pipeline);
   kgpDispatch(cb, root_gpu, groups);
   kgpSignalAfter(cb, KGP_STAGE_COMPUTE, &sem_gpu[0], 8, KGP_SIGNAL_SET);
   CHECK(kgpSubmit(queue, &cb, 1, done, 2));
   if (!offline) {
      const bool early =
         kgpWaitSemaphore(dev, done, 2, 200ull * 1000 * 1000);
      if (early || sem[0] != 7) {
         /* Let the GPU go before reporting. */
         sem[3] = 1;
         fprintf(stderr, "split: the command buffer ran before the CPU "
                 "wrote the value it waits for\n");
         return 1;
      }
   }
   sem[3] = 1;
   CHECK(kgpWaitSemaphore(dev, done, 2, KGP_TEST_TIMEOUT_NS));
   if (!offline) {
      if (sem[0] != 8) {
         fprintf(stderr, "split: no signal after the released dispatch\n");
         return 1;
      }
      if (check_y(y, a, 3))
         return 1;
   }

   /* Misuse fails the command buffer, not the process. */
   fprintf(stderr, "(the errors logged by this test are expected)\n");
   cb = kgpStartCommandRecording(queue);
   CHECK(cb);
   kgpSignalAfter(cb, KGP_STAGE_COMPUTE, (char *)sem_gpu + 4, 1,
                  KGP_SIGNAL_SET);
   CHECK(!kgpSubmit(queue, &cb, 1, NULL, 0));
   cb = kgpStartCommandRecording(queue);
   CHECK(cb);
   kgpWaitBefore(cb, KGP_STAGE_COMPUTE, NULL, 1, KGP_WAIT_EQUAL,
                 KGP_HAZARD_NONE);
   CHECK(!kgpSubmit(queue, &cb, 1, NULL, 0));

   if (offline) {
      printf("split: recorded and submitted (offline, results not checked)\n");
   } else {
      printf("split: signals, a wait on a signalled value and a wait the "
             "CPU released, all correct\n");
   }

   kgpDestroySemaphore(dev, done);
   kgpDestroyPipeline(dev, pipeline);
   kgpFree(dev, sem);
   kgpFree(dev, root);
   kgpFree(dev, y);
   kgpFree(dev, x);
   kgpDestroyDevice(dev);
   return 0;
}
