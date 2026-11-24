/* CPU cost of recording and submitting compute dispatches, KoreGP against
 * Vulkan on the same driver, shader and machine (the proposal's M0 test).
 *
 *    bench_dispatch kgp|vk [command buffers] [dispatches per command buffer]
 *
 * One API per process so `perf stat -e instructions` can wrap a run.
 * Timed with the thread's CPU clock: recording (begin .. last dispatch,
 * plus vkEndCommandBuffer) and submission separately. Waiting for the GPU
 * is not timed. The shader is saxpy over zero elements, so GPU time is
 * negligible and the same for both.
 *
 * On the DRM shim the submit ioctl is a no-op, so the record figure is
 * still meaningful offline; the submit figure is not.
 */
#include "vk_compute.h"

#include "saxpy_spv.h"

#include <inttypes.h>
#include <string.h>
#include <time.h>

#define WARMUP 16

struct saxpy_root {
   uint64_t x;
   uint64_t y;
   float a;
   uint32_t count;
};

static uint64_t
cpu_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
   return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static void
report(const char *api, uint32_t cbs, uint32_t n,
       uint64_t record_ns, uint64_t submit_ns)
{
   printf("%-4s %u command buffers x %u dispatches: "
          "record %.1f ns/dispatch, submit %.2f us/command buffer\n",
          api, cbs, n,
          (double)record_ns / ((double)cbs * n),
          (double)submit_ns / cbs / 1000.0);
}

static void
bench_kgp(uint32_t cbs, uint32_t n)
{
   KgpDevice dev = kgpCreateDevice(0);
   CHECK(dev);
   KgpQueue queue = kgpGetQueue(dev);

   struct saxpy_root *root =
      kgpMalloc(dev, sizeof(*root), 16, KGP_MEMORY_DEFAULT);
   CHECK(root);
   memset(root, 0, sizeof(*root));
   void *root_gpu = kgpHostToDevicePointer(dev, root);

   KgpPipeline pipeline = kgpCreateComputePipeline(
      dev, (KgpByteSpan) { saxpy_spv, sizeof(saxpy_spv) }, NULL);
   KgpSemaphore done = kgpCreateSemaphore(dev, 0);
   CHECK(pipeline && done);

   uint64_t record_ns = 0, submit_ns = 0;
   for (uint32_t i = 0; i < WARMUP + cbs; i++) {
      const uint64_t t0 = cpu_ns();
      KgpCommandBuffer cb = kgpStartCommandRecording(queue);
      kgpSetPipeline(cb, pipeline);
      for (uint32_t j = 0; j < n; j++)
         kgpDispatch(cb, root_gpu, (KgpUvec3) { 1, 1, 1 });
      const uint64_t t1 = cpu_ns();
      CHECK(kgpSubmit(queue, &cb, 1, done, i + 1));
      const uint64_t t2 = cpu_ns();

      CHECK(kgpWaitSemaphore(dev, done, i + 1, KGP_TEST_TIMEOUT_NS));
      if (i >= WARMUP) {
         record_ns += t1 - t0;
         submit_ns += t2 - t1;
      }
   }
   report("kgp", cbs, n, record_ns, submit_ns);

   kgpDestroySemaphore(dev, done);
   kgpDestroyPipeline(dev, pipeline);
   kgpFree(dev, root);
   kgpDestroyDevice(dev);
}

static void
bench_vk(uint32_t cbs, uint32_t n)
{
   struct vkc c;
   vkc_init(&c, false);

   struct vkc_buffer root = vkc_buffer_create(&c, sizeof(struct saxpy_root));
   memset(root.map, 0, sizeof(struct saxpy_root));

   VkPipeline pipeline = vkc_pipeline_create(&c, saxpy_spv, sizeof(saxpy_spv));
   VkCommandBuffer cmd = vkc_cmd_alloc(&c);

   uint64_t record_ns = 0, submit_ns = 0;
   for (uint32_t i = 0; i < WARMUP + cbs; i++) {
      const uint64_t t0 = cpu_ns();
      vkc_cmd_begin(cmd);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
      for (uint32_t j = 0; j < n; j++)
         vkc_cmd_dispatch(&c, cmd, root.addr, 1, 1, 1);
      VK_OK(vkEndCommandBuffer(cmd));
      const uint64_t t1 = cpu_ns();
      VK_OK(vkQueueSubmit(c.queue, 1, &(VkSubmitInfo) {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1,
         .pCommandBuffers = &cmd,
      }, VK_NULL_HANDLE));
      const uint64_t t2 = cpu_ns();

      VK_OK(vkQueueWaitIdle(c.queue));
      if (i >= WARMUP) {
         record_ns += t1 - t0;
         submit_ns += t2 - t1;
      }
   }
   report("vk", cbs, n, record_ns, submit_ns);

   vkDestroyPipeline(c.dev, pipeline, NULL);
   vkc_buffer_destroy(&c, &root);
   vkc_finish(&c);
}

int
main(int argc, char **argv)
{
   const uint32_t cbs = argc > 2 ? strtoul(argv[2], NULL, 0) : 1000;
   const uint32_t n = argc > 3 ? strtoul(argv[3], NULL, 0) : 100;

   if (argc >= 2 && cbs > 0 && n > 0 && strcmp(argv[1], "kgp") == 0) {
      bench_kgp(cbs, n);
   } else if (argc >= 2 && cbs > 0 && n > 0 && strcmp(argv[1], "vk") == 0) {
      bench_vk(cbs, n);
   } else {
      fprintf(stderr, "usage: %s kgp|vk [command buffers] "
                      "[dispatches per command buffer]\n", argv[0]);
      return 2;
   }
   return 0;
}
