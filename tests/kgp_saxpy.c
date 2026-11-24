/* y = a*x + y through one compute dispatch, checked against the CPU. */
#include "kgp_test.h"

#include "saxpy_spv.h"

#define COUNT 100000u   /* not a multiple of the 64-wide workgroup */

/* Matches SaxpyRoot in shaders/saxpy.comp (scalar layout). */
struct saxpy_root {
   uint64_t x;
   uint64_t y;
   float a;
   uint32_t count;
};

int
main(void)
{
   KgpDevice dev = kgpCreateDevice(0);
   CHECK(dev);
   KgpQueue queue = kgpGetQueue(dev);

   float *x = kgpMalloc(dev, COUNT * sizeof(float), 16, KGP_MEMORY_DEFAULT);
   float *y = kgpMalloc(dev, COUNT * sizeof(float), 16, KGP_MEMORY_READBACK);
   struct saxpy_root *root =
      kgpMalloc(dev, sizeof(*root), 16, KGP_MEMORY_DEFAULT);
   CHECK(x && y && root);

   const float a = 2.5f;
   for (uint32_t i = 0; i < COUNT; i++) {
      x[i] = (float)i;
      y[i] = 1.0f + (float)(i % 7);
   }
   const struct saxpy_root root_data = {
      .x = (uintptr_t)kgpHostToDevicePointer(dev, x),
      .y = (uintptr_t)kgpHostToDevicePointer(dev, y),
      .a = a,
      .count = COUNT,
   };
   CHECK(root_data.x && root_data.y);
   *root = root_data;

   /* An interior pointer converts too, and an unknown one does not. */
   CHECK((uintptr_t)kgpHostToDevicePointer(dev, &x[10]) ==
         root_data.x + 10 * sizeof(float));
   CHECK(kgpHostToDevicePointer(dev, &dev) == NULL);

   KgpPipeline pipeline = kgpCreateComputePipeline(
      dev, (KgpByteSpan) { saxpy_spv, sizeof(saxpy_spv) }, NULL);
   CHECK(pipeline);

   KgpSemaphore done = kgpCreateSemaphore(dev, 0);
   CHECK(done);

   KgpCommandBuffer cb = kgpStartCommandRecording(queue);
   CHECK(cb);
   kgpSetPipeline(cb, pipeline);
   kgpDispatch(cb, kgpHostToDevicePointer(dev, root),
               (KgpUvec3) { (COUNT + 63) / 64, 1, 1 });
   CHECK(kgpSubmit(queue, &cb, 1, done, 1));
   CHECK(kgpWaitSemaphore(dev, done, 1, KGP_TEST_TIMEOUT_NS));

   if (kgp_test_offline()) {
      printf("saxpy: recorded and submitted (offline, results not checked)\n");
   } else {
      for (uint32_t i = 0; i < COUNT; i++) {
         const float expect = a * (float)i + (1.0f + (float)(i % 7));
         if (y[i] != expect) {
            fprintf(stderr, "saxpy: y[%u] = %f, expected %f\n",
                    i, y[i], expect);
            return 1;
         }
      }
      printf("saxpy: %u elements correct\n", COUNT);
   }

   kgpDestroySemaphore(dev, done);
   kgpDestroyPipeline(dev, pipeline);
   kgpFree(dev, root);
   kgpFree(dev, y);
   kgpFree(dev, x);
   kgpDestroyDevice(dev);
   return 0;
}
