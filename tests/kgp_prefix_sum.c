/* Inclusive prefix sum by Hillis-Steele scan: log2(n) dispatches, each
 * reading what the previous one wrote, with a barrier between them. All
 * root structs are written up front; recording is pointer + group count.
 */
#include "kgp_test.h"

#include "prefix_sum_spv.h"

#define COUNT 70000u
#define MAX_PASSES 32

/* Matches ScanRoot in shaders/prefix_sum.comp (scalar layout), padded so
 * every element of an array of them is 64-byte aligned.
 */
struct scan_root {
   uint64_t src;
   uint64_t dst;
   uint32_t offset;
   uint32_t count;
   uint32_t pad[10];
};
_Static_assert(sizeof(struct scan_root) == 64, "root structs are 64 bytes");

static uint32_t
input_value(uint32_t i)
{
   return (i * 2654435761u) >> 20;   /* small pseudo-random values */
}

int
main(void)
{
   KgpDevice dev = kgpCreateDevice(0);
   CHECK(dev);
   KgpQueue queue = kgpGetQueue(dev);

   /* Two ping-pong buffers in one allocation, to exercise interior
    * pointers: the second starts right after the first.
    */
   uint32_t *bufs = kgpMalloc(dev, 2 * COUNT * sizeof(uint32_t), 64,
                              KGP_MEMORY_READBACK);
   struct scan_root *roots = kgpMalloc(dev, MAX_PASSES * sizeof(*roots), 64,
                                       KGP_MEMORY_DEFAULT);
   CHECK(bufs && roots);

   const uint64_t buf_gpu[2] = {
      (uintptr_t)kgpHostToDevicePointer(dev, bufs),
      (uintptr_t)kgpHostToDevicePointer(dev, bufs + COUNT),
   };
   struct scan_root *roots_gpu = kgpHostToDevicePointer(dev, roots);
   CHECK(buf_gpu[0] && buf_gpu[1] == buf_gpu[0] + COUNT * sizeof(uint32_t));
   CHECK(roots_gpu);

   for (uint32_t i = 0; i < COUNT; i++)
      bufs[i] = input_value(i);

   KgpPipeline pipeline = kgpCreateComputePipeline(
      dev, (KgpByteSpan) { prefix_sum_spv, sizeof(prefix_sum_spv) }, NULL);
   CHECK(pipeline);

   KgpSemaphore done = kgpCreateSemaphore(dev, 0);
   CHECK(done);

   KgpCommandBuffer cb = kgpStartCommandRecording(queue);
   CHECK(cb);
   kgpSetPipeline(cb, pipeline);

   uint32_t passes = 0;
   for (uint32_t offset = 1; offset < COUNT; offset *= 2, passes++) {
      CHECK(passes < MAX_PASSES);
      roots[passes] = (struct scan_root) {
         .src = buf_gpu[passes & 1],
         .dst = buf_gpu[(passes + 1) & 1],
         .offset = offset,
         .count = COUNT,
      };
      if (passes > 0)
         kgpBarrier(cb, KGP_STAGE_COMPUTE, KGP_STAGE_COMPUTE, KGP_HAZARD_NONE);
      kgpDispatch(cb, &roots_gpu[passes],
                  (KgpUvec3) { (COUNT + 63) / 64, 1, 1 });
   }
   CHECK(kgpSubmit(queue, &cb, 1, done, 1));
   CHECK(kgpWaitSemaphore(dev, done, 1, KGP_TEST_TIMEOUT_NS));

   if (kgp_test_offline()) {
      printf("prefix_sum: %u passes recorded and submitted "
             "(offline, results not checked)\n", passes);
   } else {
      const uint32_t *result = bufs + (passes & 1) * COUNT;
      uint32_t sum = 0;
      for (uint32_t i = 0; i < COUNT; i++) {
         sum += input_value(i);
         if (result[i] != sum) {
            fprintf(stderr, "prefix_sum: [%u] = %u, expected %u\n",
                    i, result[i], sum);
            return 1;
         }
      }
      printf("prefix_sum: %u elements correct after %u passes\n",
             COUNT, passes);
   }

   kgpDestroySemaphore(dev, done);
   kgpDestroyPipeline(dev, pipeline);
   kgpFree(dev, roots);
   kgpFree(dev, bufs);
   kgpDestroyDevice(dev);
   return 0;
}
