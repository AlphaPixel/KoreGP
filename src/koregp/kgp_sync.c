/*
 * Timeline semaphores and submission.
 *
 * A KgpSemaphore is a DRM timeline syncobj and nothing else. kgpSubmit is
 * nvkmd_ctx_exec (one {va, length} per pushbuf range, no buffer list) plus
 * the signals. Ancestor: nvk_queue.c:nvk_queue_submit_exec with the
 * vk_queue_submit plumbing, descriptor-table flushes and upload queue cut.
 *
 * SPDX-License-Identifier: MIT
 */
#include "kgp_private.h"

#include "util/os_time.h"
#include "util/stack_array.h"
#include "vk_drm_syncobj.h"

#include <stdlib.h>
#include <xf86drm.h>

struct KgpSemaphore_T {
   uint32_t syncobj;
};

KGP_API KgpSemaphore
kgpCreateSemaphore(KgpDevice dev, uint64_t initialValue)
{
   struct KgpSemaphore_T *sem = calloc(1, sizeof(*sem));
   if (sem == NULL)
      return NULL;

   if (drmSyncobjCreate(dev->drm_fd, 0, &sem->syncobj) != 0 ||
       (initialValue > 0 &&
        drmSyncobjTimelineSignal(dev->drm_fd, &sem->syncobj,
                                 &initialValue, 1) != 0)) {
      kgp_err("kgpCreateSemaphore: %m");
      if (sem->syncobj)
         drmSyncobjDestroy(dev->drm_fd, sem->syncobj);
      free(sem);
      return NULL;
   }

   return sem;
}

KGP_API void
kgpDestroySemaphore(KgpDevice dev, KgpSemaphore sem)
{
   if (sem == NULL)
      return;
   drmSyncobjDestroy(dev->drm_fd, sem->syncobj);
   free(sem);
}

KGP_API bool
kgpWaitSemaphore(KgpDevice dev, KgpSemaphore sem, uint64_t value,
                 uint64_t timeoutNs)
{
   return drmSyncobjTimelineWait(dev->drm_fd, &sem->syncobj, &value, 1,
                                 os_time_get_absolute_timeout(timeoutNs),
                                 DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT,
                                 NULL) == 0;
}

KGP_API bool
kgpSubmit(KgpQueue queue, const KgpCommandBuffer *cbs, uint32_t n,
          KgpSemaphore signal, uint64_t value)
{
   struct KgpDevice_T *dev = queue->dev;

   if (signal != NULL && value == 0) {
      kgp_err("kgpSubmit: a timeline semaphore cannot be signalled to 0");
      return false;
   }

   uint32_t exec_count = 0;
   for (uint32_t i = 0; i < n; i++) {
      if (cbs[i]->failed) {
         kgp_err("kgpSubmit: command buffer %u failed during recording", i);
         return false;
      }
      if (cbs[i]->seq != 0) {
         kgp_err("kgpSubmit: command buffer %u was already submitted; "
                 "command buffers are single-use", i);
         return false;
      }
      if (unlikely(cbs[i]->validate) && !kgp_cb_validate_submit(cbs[i], i))
         return false;
      kgp_cb_flush_push(cbs[i]);
      exec_count += util_dynarray_num_elements(&cbs[i]->execs,
                                               struct nvkmd_ctx_exec);
   }

   simple_mtx_lock(&queue->submit_mutex);

   if (dev->lost) {
      simple_mtx_unlock(&queue->submit_mutex);
      kgp_err("kgpSubmit: the channel was lost by an earlier submit");
      return false;
   }

   struct KgpCommandBuffer_T *slm_cb = kgp_queue_slm_update(queue);
   if (slm_cb != NULL) {
      kgp_cb_flush_push(slm_cb);
      exec_count += util_dynarray_num_elements(&slm_cb->execs,
                                               struct nvkmd_ctx_exec);
   }

   STACK_ARRAY(struct nvkmd_ctx_exec, execs, exec_count);
   uint32_t e = 0;
   if (slm_cb != NULL) {
      util_dynarray_foreach(&slm_cb->execs, struct nvkmd_ctx_exec, exec)
         execs[e++] = *exec;
   }
   for (uint32_t i = 0; i < n; i++) {
      util_dynarray_foreach(&cbs[i]->execs, struct nvkmd_ctx_exec, exec)
         execs[e++] = *exec;
   }
   assert(e == exec_count);

   /* Everything in this submit retires at the next device timeline point. */
   simple_mtx_lock(&dev->mutex);
   const uint64_t seq = ++dev->submit_seq;
   if (slm_cb != NULL) {
      slm_cb->seq = seq;
      list_del(&slm_cb->link);
      list_addtail(&slm_cb->link, &queue->pending);
   }
   for (uint32_t i = 0; i < n; i++) {
      cbs[i]->seq = seq;
      list_del(&cbs[i]->link);
      list_addtail(&cbs[i]->link, &queue->pending);
   }
   simple_mtx_unlock(&dev->mutex);

   /* nvkmd takes vk_sync pointers but only reads the syncobj handle out of
    * them (nvkmd_nouveau_ctx_add_sync), so these never touch a vk_device.
    */
   struct vk_drm_syncobj syncs[2] = {
      {
         .base = { .type = dev->pdev->sync_types[0],
                   .flags = VK_SYNC_IS_TIMELINE },
         .syncobj = dev->timeline,
      },
      {
         .base = { .type = dev->pdev->sync_types[0],
                   .flags = VK_SYNC_IS_TIMELINE },
         .syncobj = signal ? signal->syncobj : 0,
      },
   };
   const struct vk_sync_signal signals[2] = {
      { .sync = &syncs[0].base, .stage_mask = ~0ull, .signal_value = seq },
      { .sync = &syncs[1].base, .stage_mask = ~0ull, .signal_value = value },
   };

   VkResult result = nvkmd_ctx_exec(queue->ctx, KGP_LOG_OBJ, exec_count, execs);
   if (result == VK_SUCCESS) {
      /* Implies a flush: this is the DRM_NOUVEAU_EXEC ioctl. */
      result = nvkmd_ctx_signal(queue->ctx, KGP_LOG_OBJ,
                                signal ? 2 : 1, signals);
   }

   STACK_ARRAY_FINISH(execs);

   if (result != VK_SUCCESS) {
      dev->lost = true;
      kgp_err("kgpSubmit: submission failed; the device is lost");
   }

   simple_mtx_unlock(&queue->submit_mutex);
   return result == VK_SUCCESS;
}
