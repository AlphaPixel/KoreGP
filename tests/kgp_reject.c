/* Things KoreGP must refuse, cleanly: every one of these returns an error
 * instead of crashing or reaching the GPU. Nothing here needs results from
 * the GPU, so it checks the same on the shim and on hardware.
 */
#include "kgp_test.h"

#include "bad_push_spv.h"
#include "bad_ubo_spv.h"
#include "saxpy_spv.h"

int
main(void)
{
   KgpDevice dev = kgpCreateDevice(0);
   CHECK(dev);
   KgpQueue queue = kgpGetQueue(dev);

   fprintf(stderr, "(the errors logged below are the point of this test)\n");

   /* No such device. */
   CHECK(kgpCreateDevice(1000) == NULL);

   /* Memory. */
   CHECK(kgpMalloc(dev, 0, 16, KGP_MEMORY_DEFAULT) == NULL);
   CHECK(kgpMalloc(dev, 64, 24, KGP_MEMORY_DEFAULT) == NULL);
   CHECK(kgpMalloc(dev, 64, 16, (KGP_MEMORY)99) == NULL);
   uint32_t *mem = kgpMalloc(dev, 4096, 16, KGP_MEMORY_DEFAULT);
   CHECK(mem);
   kgpFree(dev, mem + 1);            /* interior pointer: refused, no crash */
   kgpFree(dev, &dev);               /* not ours at all */
   kgpFree(dev, NULL);               /* allowed, like free() */
   CHECK(kgpHostToDevicePointer(dev, mem) != NULL);   /* still live */

   /* GPU-only memory: the returned pointer is the GPU pointer. */
   void *gpu_only = kgpMalloc(dev, 1 << 20, 4096, KGP_MEMORY_GPU);
   CHECK(gpu_only);
   CHECK(kgpHostToDevicePointer(dev, gpu_only) == gpu_only);
   kgpFree(dev, gpu_only);
   CHECK(kgpHostToDevicePointer(dev, gpu_only) == NULL);

   /* Pipelines. */
   static const uint32_t junk[8] = { 0xdeadbeef };
   CHECK(!kgpCreateComputePipeline(dev, (KgpByteSpan) { junk, sizeof(junk) },
                                   NULL));
   CHECK(!kgpCreateComputePipeline(dev, (KgpByteSpan) { saxpy_spv, 10 }, NULL));
   CHECK(!kgpCreateComputePipeline(
            dev, (KgpByteSpan) { bad_push_spv, sizeof(bad_push_spv) }, NULL));
   CHECK(!kgpCreateComputePipeline(
            dev, (KgpByteSpan) { bad_ubo_spv, sizeof(bad_ubo_spv) }, NULL));
   CHECK(!kgpCreateComputePipeline(
            dev, (KgpByteSpan) { saxpy_spv, sizeof(saxpy_spv) },
            &(KgpSpecialization) { NULL, 1 }));

   KgpPipeline pipeline = kgpCreateComputePipeline(
      dev, (KgpByteSpan) { saxpy_spv, sizeof(saxpy_spv) }, NULL);
   CHECK(pipeline);
   KgpSemaphore sem = kgpCreateSemaphore(dev, 0);
   CHECK(sem);

   /* A semaphore that is never signalled times out. Not on the shim, whose
    * waits always succeed.
    */
   if (!kgp_test_offline())
      CHECK(!kgpWaitSemaphore(dev, sem, 1, 1000 * 1000));

   /* Dispatch with no pipeline: the buffer is poisoned, submit refuses. */
   KgpCommandBuffer cb = kgpStartCommandRecording(queue);
   CHECK(cb);
   kgpDispatch(cb, NULL, (KgpUvec3) { 1, 1, 1 });
   CHECK(!kgpSubmit(queue, &cb, 1, NULL, 0));

   /* Group counts the hardware cannot express. */
   cb = kgpStartCommandRecording(queue);
   CHECK(cb);
   kgpSetPipeline(cb, pipeline);
   kgpDispatch(cb, NULL, (KgpUvec3) { 1, 1 << 20, 1 });
   CHECK(!kgpSubmit(queue, &cb, 1, NULL, 0));

   /* A timeline semaphore cannot be signalled to 0. */
   cb = kgpStartCommandRecording(queue);
   CHECK(cb);
   CHECK(!kgpSubmit(queue, &cb, 1, sem, 0));

   /* That refusal left the buffer usable; a second submit of it does not.
    * (Only this simple case is caught: nothing has polled for completion
    * in between, so the first submit has not yet retired the buffer.)
    */
   CHECK(kgpSubmit(queue, &cb, 1, sem, 1));
   CHECK(!kgpSubmit(queue, &cb, 1, sem, 2));
   CHECK(kgpWaitSemaphore(dev, sem, 1, KGP_TEST_TIMEOUT_NS));

   /* An empty submit is legal: it signals once earlier work is done. */
   CHECK(kgpSubmit(queue, NULL, 0, sem, 2));
   CHECK(kgpWaitSemaphore(dev, sem, 2, KGP_TEST_TIMEOUT_NS));

   kgpDestroySemaphore(dev, sem);
   kgpDestroyPipeline(dev, pipeline);
   kgpFree(dev, mem);
   /* The two poisoned command buffers are reclaimed here. */
   kgpDestroyDevice(dev);

   printf("reject: every invalid request was refused\n");
   return 0;
}
