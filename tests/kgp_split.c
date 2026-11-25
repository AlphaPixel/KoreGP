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
 */
#include "kgp_test.h"

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

int
main(void)
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
