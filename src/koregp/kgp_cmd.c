/*
 * Queue setup, command recording, dispatch.
 *
 * Ancestors, all in src/nouveau/vulkan:
 *   nvk_cmd_buffer.c   nvk_cmd_buffer_{new,flush}_push, _upload_alloc
 *   nvk_cmd_dispatch.c nvk_push_dispatch_state_init, _begin_compute,
 *                      nvk_cmd_upload_qmd, nvk_CmdDispatchBase
 *   nvk_queue.c        nvk_queue_slm_state_update
 * Cut: vk_command_buffer, descriptor state, the root descriptor table,
 * queries, conditional rendering, pre-Turing branches.
 *
 * SPDX-License-Identifier: MIT
 */
#include "kgp_private.h"

#include "cla0c0.h"
#include "clc5c0.h"
#include "clcbc0.h"
#include "nv_push_cl9097.h"
#include "nv_push_cl90c0.h"
#include "nv_push_cla0c0.h"
#include "nv_push_clb1c0.h"
#include "nv_push_clc3c0.h"
#include "nv_push_clc6c0.h"

#include <stdlib.h>

#define KGP_SUBC_MASK (BITFIELD_BIT(SUBC_NV9097) | BITFIELD_BIT(SUBC_NV90C0) | \
                       BITFIELD_BIT(SUBC_NV90B5))

/* Largest single kgp_cb_push request (the channel init). Also the size of
 * the runout buffer recording falls into when a chunk allocation fails
 * (ancestor: push_runout); such a command buffer is marked failed and never
 * submitted.
 */
#define KGP_MAX_PUSH_DW KGP_INIT_PUSH_DW
static uint32_t kgp_push_runout[KGP_MAX_PUSH_DW];

/* Chunks ------------------------------------------------------------------- */

static struct kgp_chunk *
kgp_cb_new_chunk(struct KgpCommandBuffer_T *cb, enum kgp_chunk_kind kind)
{
   struct KgpQueue_T *queue = cb->queue;
   struct KgpDevice_T *dev = queue->dev;
   struct kgp_chunk *chunk = NULL;

   simple_mtx_lock(&dev->mutex);
   /* ponytail: one syncobj query whenever no chunk is free. If per-buffer
    * cost ever matters, learn completion from kgpWaitSemaphore instead.
    */
   if (list_is_empty(&queue->free_chunks[kind]))
      kgp_device_poll_locked(dev);
   if (!list_is_empty(&queue->free_chunks[kind])) {
      chunk = list_first_entry(&queue->free_chunks[kind],
                               struct kgp_chunk, link);
      list_del(&chunk->link);
   }
   simple_mtx_unlock(&dev->mutex);

   if (chunk == NULL) {
      chunk = calloc(1, sizeof(*chunk));
      /* Pushbufs are read once by the command streamer, which pulls across
       * PCI fine; uploads are read by the compute engine. Same split as NVK.
       */
      const enum nvkmd_mem_flags flags =
         kind == KGP_CHUNK_PUSH ? NVKMD_MEM_GART : NVKMD_MEM_LOCAL;
      if (chunk == NULL ||
          nvkmd_dev_alloc_mapped_mem(dev->nvkmd, KGP_LOG_OBJ,
                                     KGP_CHUNK_SIZE_B, 0, flags,
                                     NVKMD_MEM_MAP_WR,
                                     &chunk->mem) != VK_SUCCESS) {
         free(chunk);
         cb->failed = true;
         return NULL;
      }
   }

   list_addtail(&chunk->link, &cb->chunks[kind]);
   return chunk;
}

/* `incomplete`: the range ends inside a method whose data is in the next
 * range (kgp_cb_push_indirect); the kernel keeps the two in one submit.
 */
static void
kgp_cb_flush_push_as(struct KgpCommandBuffer_T *cb, bool incomplete)
{
   if (cb->push_chunk != NULL && nv_push_dw_count(&cb->push) > 0) {
      const struct nvkmd_mem *mem = cb->push_chunk->mem;
      const struct nvkmd_ctx_exec exec = {
         .addr = mem->va->addr +
                 ((char *)cb->push.start - (char *)mem->map),
         .size_B = nv_push_dw_count(&cb->push) * 4,
         .incomplete = incomplete,
      };
      util_dynarray_append(&cb->execs, exec);
   }
   cb->push.start = cb->push.end;
}

void
kgp_cb_flush_push(struct KgpCommandBuffer_T *cb)
{
   kgp_cb_flush_push_as(cb, false);
}

static void
kgp_cb_new_push(struct KgpCommandBuffer_T *cb)
{
   kgp_cb_flush_push(cb);

   cb->push_chunk = kgp_cb_new_chunk(cb, KGP_CHUNK_PUSH);
   if (cb->push_chunk != NULL) {
      nv_push_init(&cb->push, cb->push_chunk->mem->map,
                   KGP_CHUNK_SIZE_B / 4, KGP_SUBC_MASK);
   } else {
      nv_push_init(&cb->push, kgp_push_runout, KGP_MAX_PUSH_DW, KGP_SUBC_MASK);
   }
}

struct nv_push *
kgp_cb_push(struct KgpCommandBuffer_T *cb, uint32_t dw_count)
{
   assert(dw_count <= KGP_MAX_PUSH_DW);
   if (unlikely(cb->push.end + dw_count > cb->push.limit))
      kgp_cb_new_push(cb);
   return &cb->push;
}

static bool
kgp_cb_upload_alloc(struct KgpCommandBuffer_T *cb,
                    uint32_t size_B, uint32_t align_B,
                    uint64_t *addr_out, void **map_out)
{
   assert(size_B <= KGP_CHUNK_SIZE_B);

   uint32_t offset = align(cb->upload_offset, align_B);
   if (cb->upload_chunk == NULL || offset + size_B > KGP_CHUNK_SIZE_B) {
      cb->upload_chunk = kgp_cb_new_chunk(cb, KGP_CHUNK_UPLOAD);
      if (cb->upload_chunk == NULL)
         return false;
      offset = 0;
   }

   *addr_out = cb->upload_chunk->mem->va->addr + offset;
   *map_out = (char *)cb->upload_chunk->mem->map + offset;
   cb->upload_offset = offset + size_B;
   return true;
}

/* Command buffer lifetime -------------------------------------------------- */

static struct KgpCommandBuffer_T *
kgp_cb_create(struct KgpQueue_T *queue)
{
   struct KgpDevice_T *dev = queue->dev;

   struct KgpCommandBuffer_T *cb = calloc(1, sizeof(*cb));
   if (cb == NULL)
      return NULL;

   cb->queue = queue;
   cb->execs = UTIL_DYNARRAY_INIT;
   cb->refs = UTIL_DYNARRAY_INIT;
   cb->validate = dev->debug & KGP_DEBUG_VALIDATE;
   for (unsigned k = 0; k < KGP_CHUNK_KINDS; k++)
      list_inithead(&cb->chunks[k]);

   simple_mtx_lock(&dev->mutex);
   list_addtail(&cb->link, &queue->recording);
   simple_mtx_unlock(&dev->mutex);

   kgp_cb_new_push(cb);
   return cb;
}

/* Takes the command buffer off whichever queue list it is on and returns
 * its chunks to the free lists.
 */
static void
kgp_cb_destroy_locked(struct KgpCommandBuffer_T *cb)
{
   struct KgpQueue_T *queue = cb->queue;

   list_del(&cb->link);
   for (unsigned k = 0; k < KGP_CHUNK_KINDS; k++)
      list_splicetail(&cb->chunks[k], &queue->free_chunks[k]);
   util_dynarray_fini(&cb->execs);
   util_dynarray_fini(&cb->refs);
   free(cb);
}

void
kgp_queue_reap_locked(struct KgpQueue_T *queue)
{
   list_for_each_entry_safe(struct KgpCommandBuffer_T, cb,
                            &queue->pending, link) {
      if (cb->seq > queue->dev->done_seq)
         break;
      kgp_cb_destroy_locked(cb);
   }
}

/* Channel state ------------------------------------------------------------ */

/* Ancestor: nvk_push_dispatch_state_init, Volta+ branches only. */
static void
kgp_push_dispatch_state_init(const struct nv_device_info *info,
                             struct nv_push *p)
{
   P_MTHD(p, NV90C0, SET_OBJECT);
   P_NV90C0_SET_OBJECT(p, {
      .class_id = info->cls_compute,
      .engine_id = 0,
   });

   /* The windows are VA ranges the hardware decodes as shared / local
    * memory. NVK places them at the top of the low 4 GiB and does not
    * reserve them in the VA allocator (its own TODO); same here.
    * The shared memory window needs 4 GiB alignment on Hopper+.
    */
   uint64_t temp = info->cls_compute >= HOPPER_COMPUTE_A ? 1ULL << 32
                                                         : 0xfeULL << 24;
   P_MTHD(p, NVC3C0, SET_SHADER_SHARED_MEMORY_WINDOW_A);
   P_NVC3C0_SET_SHADER_SHARED_MEMORY_WINDOW_A(p, temp >> 32);
   P_NVC3C0_SET_SHADER_SHARED_MEMORY_WINDOW_B(p, temp & 0xffffffff);

   temp = 0xffULL << 24;
   P_MTHD(p, NVC3C0, SET_SHADER_LOCAL_MEMORY_WINDOW_A);
   P_NVC3C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A(p, temp >> 32);
   P_NVC3C0_SET_SHADER_LOCAL_MEMORY_WINDOW_B(p, temp & 0xffffffff);

   /* No texture or sampler heap yet (N2). NVK never leaves the pools
    * unprogrammed: with nothing to bind it sets a NULL address and a count
    * of one. Ancestor: nvk_queue_heap_state_bind_{image,sampler}_heap.
    */
   P_MTHD(p, NVA0C0, SET_TEX_HEADER_POOL_A);
   P_NVA0C0_SET_TEX_HEADER_POOL_A(p, 0);
   P_NVA0C0_SET_TEX_HEADER_POOL_B(p, 0);
   P_NVA0C0_SET_TEX_HEADER_POOL_C(p, 0);
   P_IMMD(p, NVA0C0, INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI, {
      .lines = LINES_ALL
   });

   P_MTHD(p, NVA0C0, SET_TEX_SAMPLER_POOL_A);
   P_NVA0C0_SET_TEX_SAMPLER_POOL_A(p, 0);
   P_NVA0C0_SET_TEX_SAMPLER_POOL_B(p, 0);
   P_NVA0C0_SET_TEX_SAMPLER_POOL_C(p, 0);
   P_IMMD(p, NVA0C0, INVALIDATE_SAMPLER_CACHE_NO_WFI, {
      .lines = LINES_ALL
   });
}

/* If shader-local memory grew since this queue last programmed it, returns
 * a command buffer that reprograms it, to be submitted ahead of the caller's.
 * Ancestor: nvk_queue_slm_state_update. Called with submit_mutex held.
 */
struct KgpCommandBuffer_T *
kgp_queue_slm_update(struct KgpQueue_T *queue)
{
   struct KgpDevice_T *dev = queue->dev;

   simple_mtx_lock(&dev->mutex);
   const struct nvkmd_mem *mem = dev->slm_mem;
   const uint64_t slm_per_tpc = dev->slm_bytes_per_tpc;
   const uint64_t slm_per_warp = dev->slm_bytes_per_warp;
   simple_mtx_unlock(&dev->mutex);

   if (slm_per_tpc == queue->slm_bytes_per_tpc)
      return NULL;

   struct KgpCommandBuffer_T *cb = kgp_cb_create(queue);
   if (cb == NULL)
      return NULL;

   const uint64_t slm_addr = mem->va->addr;
   assert(!(slm_per_tpc & 0x7fff));

   struct nv_push *p = kgp_cb_push(cb, 13);
   P_MTHD(p, NVA0C0, SET_SHADER_LOCAL_MEMORY_A);
   P_NVA0C0_SET_SHADER_LOCAL_MEMORY_A(p, slm_addr >> 32);
   P_NVA0C0_SET_SHADER_LOCAL_MEMORY_B(p, slm_addr);

   P_MTHD(p, NVA0C0, SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A);
   P_NVA0C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A(p, slm_per_tpc >> 32);
   P_NVA0C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_B(p, slm_per_tpc);
   P_NVA0C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C(p, 0xff);

   /* The 3D engine has its own registers for the same area. */
   const uint64_t slm_size = mem->size_B;
   P_MTHD(p, NV9097, SET_SHADER_LOCAL_MEMORY_A);
   P_NV9097_SET_SHADER_LOCAL_MEMORY_A(p, slm_addr >> 32);
   P_NV9097_SET_SHADER_LOCAL_MEMORY_B(p, slm_addr);
   P_NV9097_SET_SHADER_LOCAL_MEMORY_C(p, slm_size >> 32);
   P_NV9097_SET_SHADER_LOCAL_MEMORY_D(p, slm_size);
   P_NV9097_SET_SHADER_LOCAL_MEMORY_E(p, slm_per_warp);

   queue->slm_bytes_per_tpc = slm_per_tpc;
   return cb;
}

bool
kgp_queue_init(struct KgpDevice_T *dev)
{
   struct KgpQueue_T *queue = &dev->queue;

   queue->dev = dev;
   simple_mtx_init(&queue->submit_mutex, mtx_plain);
   for (unsigned k = 0; k < KGP_CHUNK_KINDS; k++)
      list_inithead(&queue->free_chunks[k]);
   list_inithead(&queue->recording);
   list_inithead(&queue->pending);

   /* The engine set of NVK's main queue. The copy engine needs no
    * SET_OBJECT: the kernel binds its class to subchannel 4 when the channel
    * is made (NVK pushes none either).
    */
   const enum nvkmd_engines engines =
      NVKMD_ENGINE_3D | NVKMD_ENGINE_COMPUTE | NVKMD_ENGINE_COPY;

   if (nvkmd_dev_create_ctx(dev->nvkmd, KGP_LOG_OBJ, engines,
                            &queue->ctx) != VK_SUCCESS)
      goto fail_mutex;

   if (nvkmd_dev_alloc_mem(dev->nvkmd, KGP_LOG_OBJ, KGP_DRAW_CB0_MEM_SIZE_B,
                           0, NVKMD_MEM_LOCAL,
                           &queue->draw_cb0) != VK_SUCCESS ||
       nvkmd_dev_alloc_mapped_mem(dev->nvkmd, KGP_LOG_OBJ, 0x1000, 0,
                                  NVKMD_MEM_LOCAL, NVKMD_MEM_MAP_WR,
                                  &queue->zero_page) != VK_SUCCESS)
      goto fail_ctx;
   memset(queue->zero_page->map, 0, 0x1000);

   /* Ancestor: nvk_queue_init_context_state. 3D first, then compute, in one
    * push, before anything else runs on the channel.
    */
   KgpCommandBuffer cb = kgp_cb_create(queue);
   if (cb == NULL)
      goto fail_ctx;
   struct nv_push *p = kgp_cb_push(cb, KGP_INIT_PUSH_DW);
   uint32_t mme_end = 0;
   if (!kgp_push_draw_state_init(kgp_dev_info(dev), p,
                                 queue->draw_cb0->va->addr,
                                 queue->zero_page->va->addr, &mme_end))
      goto fail_ctx;

   /* No texture or sampler heap yet on the 3D engine either: a NULL
    * address, as for compute (kgp_push_dispatch_state_init) and as NVK's
    * first submit does with its own pools.
    */
   P_MTHD(p, NV9097, SET_TEX_HEADER_POOL_A);
   P_NV9097_SET_TEX_HEADER_POOL_A(p, 0);
   P_NV9097_SET_TEX_HEADER_POOL_B(p, 0);
   P_NV9097_SET_TEX_HEADER_POOL_C(p, 0);
   P_IMMD(p, NV9097, INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI, {
      .lines = LINES_ALL
   });

   P_MTHD(p, NV9097, SET_TEX_SAMPLER_POOL_A);
   P_NV9097_SET_TEX_SAMPLER_POOL_A(p, 0);
   P_NV9097_SET_TEX_SAMPLER_POOL_B(p, 0);
   P_NV9097_SET_TEX_SAMPLER_POOL_C(p, 0);
   P_IMMD(p, NV9097, INVALIDATE_SAMPLER_CACHE_NO_WFI, {
      .lines = LINES_ALL
   });

   kgp_push_own_macros(kgp_dev_info(dev), p, mme_end);

   kgp_push_dispatch_state_init(kgp_dev_info(dev), p);
   if (!kgpSubmit(queue, &cb, 1, NULL, 0))
      goto fail_ctx;

   return true;

fail_ctx:
   kgp_queue_finish(dev);
   return false;
fail_mutex:
   simple_mtx_destroy(&queue->submit_mutex);
   return false;
}

void
kgp_queue_finish(struct KgpDevice_T *dev)
{
   struct KgpQueue_T *queue = &dev->queue;

   simple_mtx_lock(&dev->mutex);
   /* Recorded but never submitted, or submitted on a channel that died. */
   list_for_each_entry_safe(struct KgpCommandBuffer_T, cb,
                            &queue->recording, link)
      kgp_cb_destroy_locked(cb);
   list_for_each_entry_safe(struct KgpCommandBuffer_T, cb,
                            &queue->pending, link)
      kgp_cb_destroy_locked(cb);
   simple_mtx_unlock(&dev->mutex);

   for (unsigned k = 0; k < KGP_CHUNK_KINDS; k++) {
      list_for_each_entry_safe(struct kgp_chunk, chunk,
                               &queue->free_chunks[k], link) {
         nvkmd_mem_unref(chunk->mem);
         free(chunk);
      }
   }

   if (queue->zero_page)
      nvkmd_mem_unref(queue->zero_page);
   if (queue->draw_cb0)
      nvkmd_mem_unref(queue->draw_cb0);
   nvkmd_ctx_destroy(queue->ctx);
   simple_mtx_destroy(&queue->submit_mutex);
}

/* Public recording API ----------------------------------------------------- */

KGP_API KgpCommandBuffer
kgpStartCommandRecording(KgpQueue queue)
{
   struct KgpCommandBuffer_T *cb = kgp_cb_create(queue);
   if (cb == NULL)
      return NULL;

   /* Ancestor: nvk_cmd_buffer_begin_compute. Upload chunks are recycled, so
    * a QMD address seen before may now hold a different QMD: the scheduler's
    * cache has to be dropped at the top of every command buffer.
    */
   struct nv_push *p = kgp_cb_push(cb, 6);
   P_IMMD(p, NVB1C0, INVALIDATE_SKED_CACHES, 0);
   P_IMMD(p, NVA0C0, INVALIDATE_SAMPLER_CACHE_NO_WFI, {
      .lines = LINES_ALL,
   });
   P_IMMD(p, NVA0C0, INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI, {
      .lines = LINES_ALL,
   });

   return cb;
}

KGP_API void
kgpSetPipeline(KgpCommandBuffer cb, KgpPipeline pipeline)
{
   cb->pipeline = pipeline;
   /* A graphics pipeline's methods go out with the next draw. */
   cb->gfx_dirty = true;
}

/* The proposal's set-state commands: the state goes out with the next draw,
 * in the place NVK's dynamic state would (kgp_cb_flush_gfx_state).
 */
KGP_API void
kgpSetDepthStencilState(KgpCommandBuffer cb, KgpDepthStencilState state)
{
   cb->ds_dirty |= cb->ds_state != state;
   cb->ds_state = state;
}

KGP_API void
kgpSetBlendState(KgpCommandBuffer cb, KgpBlendState state)
{
   cb->blend_dirty |= cb->blend_state != state;
   cb->blend_state = state;
}

/* Ancestor: nvk_queue_heap_state_bind_image_heap. NVK points the pool at
 * its own descriptor table once per queue; here it is the caller's memory.
 */
KGP_API void
kgpSetActiveTextureHeapPtr(KgpCommandBuffer cb, void *heapGpu, uint32_t count)
{
   uint64_t addr = (uintptr_t)heapGpu;

   if (unlikely(count > KGP_MAX_HEAP_TEXTURES ||
                addr % sizeof(KgpDescriptor256) != 0)) {
      kgp_err("kgpSetActiveTextureHeapPtr: at most %u entries, at a "
              "32-byte aligned GPU pointer", KGP_MAX_HEAP_TEXTURES);
      cb->failed = true;
      return;
   }
   if (!kgp_check_ptr(cb, "kgpSetActiveTextureHeapPtr: the heap", heapGpu,
                      (uint64_t)count * sizeof(KgpDescriptor256), 1))
      return;

   /* We can't actually set a size of 0 but we can set a NULL address */
   if (addr == 0 || count == 0) {
      addr = 0;
      count = 1;
   }

   /* Each engine has its own pool registers; NVK programs both too. */
   struct nv_push *p = kgp_cb_push(cb, 12);
   P_MTHD(p, NVA0C0, SET_TEX_HEADER_POOL_A);
   P_NVA0C0_SET_TEX_HEADER_POOL_A(p, addr >> 32);
   P_NVA0C0_SET_TEX_HEADER_POOL_B(p, addr);
   P_NVA0C0_SET_TEX_HEADER_POOL_C(p, count - 1);
   P_IMMD(p, NVA0C0, INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI, {
      .lines = LINES_ALL
   });

   P_MTHD(p, NV9097, SET_TEX_HEADER_POOL_A);
   P_NV9097_SET_TEX_HEADER_POOL_A(p, addr >> 32);
   P_NV9097_SET_TEX_HEADER_POOL_B(p, addr);
   P_NV9097_SET_TEX_HEADER_POOL_C(p, count - 1);
   P_IMMD(p, NV9097, INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI, {
      .lines = LINES_ALL
   });
}

/* Ancestor: nvk_queue_heap_state_bind_sampler_heap. */
KGP_API void
kgpSetActiveSamplerHeapPtr(KgpCommandBuffer cb, void *heapGpu, uint32_t count)
{
   uint64_t addr = (uintptr_t)heapGpu;

   if (unlikely(count > KGP_MAX_HEAP_SAMPLERS ||
                addr % sizeof(KgpDescriptor256) != 0)) {
      kgp_err("kgpSetActiveSamplerHeapPtr: at most %u entries, at a "
              "32-byte aligned GPU pointer", KGP_MAX_HEAP_SAMPLERS);
      cb->failed = true;
      return;
   }
   if (!kgp_check_ptr(cb, "kgpSetActiveSamplerHeapPtr: the heap", heapGpu,
                      (uint64_t)count * sizeof(KgpDescriptor256), 1))
      return;

   /* We can't actually set a size of 0 but we can set a NULL address */
   if (addr == 0 || count == 0) {
      addr = 0;
      count = 1;
   }

   struct nv_push *p = kgp_cb_push(cb, 12);
   P_MTHD(p, NVA0C0, SET_TEX_SAMPLER_POOL_A);
   P_NVA0C0_SET_TEX_SAMPLER_POOL_A(p, addr >> 32);
   P_NVA0C0_SET_TEX_SAMPLER_POOL_B(p, addr);
   P_NVA0C0_SET_TEX_SAMPLER_POOL_C(p, count - 1);
   P_IMMD(p, NVA0C0, INVALIDATE_SAMPLER_CACHE_NO_WFI, {
      .lines = LINES_ALL
   });

   P_MTHD(p, NV9097, SET_TEX_SAMPLER_POOL_A);
   P_NV9097_SET_TEX_SAMPLER_POOL_A(p, addr >> 32);
   P_NV9097_SET_TEX_SAMPLER_POOL_B(p, addr);
   P_NV9097_SET_TEX_SAMPLER_POOL_C(p, count - 1);
   P_IMMD(p, NV9097, INVALIDATE_SAMPLER_CACHE_NO_WFI, {
      .lines = LINES_ALL
   });
}

/* The pushbuf continues in memory the caller owns: the GPU's command
 * fetcher reads `size_B` bytes of method data straight from `addr`.
 * Ancestor: nvk_cmd_buffer_push_indirect.
 */
static void
kgp_cb_push_indirect(struct KgpCommandBuffer_T *cb, uint64_t addr,
                     uint32_t size_B)
{
   kgp_cb_flush_push_as(cb, true /* incomplete */);

   const struct nvkmd_ctx_exec exec = {
      .addr = addr,
      .size_B = size_B,
      .no_prefetch = true,
   };
   util_dynarray_append(&cb->execs, exec);
}

/* GPU-side copy of `size_B` bytes from caller memory, using the compute
 * class's inline-to-memory upload with the inline data taken from `src`.
 * Ancestor: nvk_cmd_compute_indirect_copy.
 */
static void
kgp_cb_indirect_copy(struct KgpCommandBuffer_T *cb, uint64_t dst_addr,
                     uint64_t src_addr, uint32_t size_B)
{
   struct nv_push *p = kgp_cb_push(cb, 7);
   P_MTHD(p, NVA0C0, LINE_LENGTH_IN);
   P_NVA0C0_LINE_LENGTH_IN(p, size_B);
   P_NVA0C0_LINE_COUNT(p, 1);
   P_NVA0C0_OFFSET_OUT_UPPER(p, dst_addr >> 32);
   P_NVA0C0_OFFSET_OUT(p, dst_addr);
   P_1INC(p, NVA0C0, LAUNCH_DMA);
   P_NVA0C0_LAUNCH_DMA(p, {
      .dst_memory_layout = DST_MEMORY_LAYOUT_PITCH,
      .completion_type = COMPLETION_TYPE_FLUSH_ONLY,
   });
   nv_push_update_count(p, DIV_ROUND_UP(size_B, 4));
   kgp_cb_push_indirect(cb, src_addr, ALIGN_POT(size_B, 4));
}

/* Ancestors: nvk_cmd_upload_qmd + nvk_CmdDispatchBase, and
 * nvk_CmdDispatchIndirect2KHR when `groups_gpu` is set (the group counts
 * are then three uint32_t the GPU reads from that address).
 */
static void
kgp_cb_dispatch(struct KgpCommandBuffer_T *cb, void *dataGpu,
                KgpUvec3 groups, uint64_t groups_gpu)
{
   const struct nv_device_info *info = kgp_dev_info(cb->queue->dev);
   const struct KgpPipeline_T *pipeline = cb->pipeline;

   if (unlikely(pipeline == NULL || pipeline->graphics)) {
      kgp_err("dispatch: no compute pipeline set");
      cb->failed = true;
      return;
   }
   if (unlikely(groups.y > pipeline->max_groups_y ||
                groups.z > pipeline->max_groups_z)) {
      kgp_err("dispatch: group count (%u, %u, %u) exceeds the hardware "
              "limit of %u in Y or %u in Z", groups.x, groups.y, groups.z,
              pipeline->max_groups_y, pipeline->max_groups_z);
      cb->failed = true;
      return;
   }
   if (unlikely(pipeline->user_root_size_B > 0 &&
                (uintptr_t)dataGpu % KGP_CBUF_ALIGN_B != 0)) {
      kgp_err("dispatch: under root convention v1 the root pointer must be "
              "%u-byte aligned", KGP_CBUF_ALIGN_B);
      cb->failed = true;
      return;
   }
   if (unlikely(groups_gpu != 0 &&
                (groups_gpu % 4 != 0 || !pipeline->qmd_patchable))) {
      kgp_err("dispatch: indirect group counts must be 4-byte aligned%s",
              pipeline->qmd_patchable ? "" :
              " (and this chip's QMD layout is not supported for indirect)");
      cb->failed = true;
      return;
   }
   if (!kgp_check_ptr(cb, "dispatch: the root pointer", dataGpu, 1,
                      KGP_ROOT_ALIGN_B) ||
       !kgp_check_ptr(cb, "dispatch: the indirect group counts",
                      (void *)(uintptr_t)groups_gpu, 3 * sizeof(uint32_t), 4))
      return;

   /* One upload per dispatch: the QMD, then the root record (cbuf 0) right
    * behind it. NVK uploads a root descriptor table of several hundred
    * bytes here; ours is 64.
    */
   const uint32_t qmd_size_B = pipeline->qmd_size_B;
   const uint32_t root_offset = align(qmd_size_B, KGP_CBUF_ALIGN_B);
   uint64_t qmd_addr;
   void *map;
   if (!kgp_cb_upload_alloc(cb, root_offset + sizeof(struct kgp_root_record),
                            NAK_QMD_ALIGN_B, &qmd_addr, &map))
      return;
   const uint64_t root_addr = qmd_addr + root_offset;

   /* Built on the stack and copied: the upload map is write-combined. */
   uint32_t qmd[NAK_MAX_QMD_DWORDS];
   kgp_pipeline_fill_qmd(info, pipeline, qmd, root_addr, (uintptr_t)dataGpu,
                         groups.x, groups.y, groups.z);
   memcpy(map, qmd, qmd_size_B);

   const struct kgp_root_record root = {
      .root = (uintptr_t)dataGpu,
      .group_count = { groups.x, groups.y, groups.z },
   };
   memcpy((char *)map + root_offset, &root, sizeof(root));

   if (groups_gpu != 0) {
      /* The counts were written as zero above; have the GPU overwrite them
       * in both places: where the shader reads gl_NumWorkGroups, and the
       * three consecutive dwords of the QMD.
       */
      kgp_cb_indirect_copy(
         cb, root_addr + offsetof(struct kgp_root_record, group_count),
         groups_gpu, 3 * sizeof(uint32_t));
      kgp_cb_indirect_copy(cb, qmd_addr + pipeline->qmd_size_dw * 4,
                           groups_gpu, 3 * sizeof(uint32_t));
   }

   struct nv_push *p = kgp_cb_push(cb, 4);
   P_MTHD(p, NVA0C0, SEND_PCAS_A);
   P_NVA0C0_SEND_PCAS_A(p, qmd_addr >> 8);

   if (info->cls_compute <= TURING_COMPUTE_A) {
      P_IMMD(p, NVA0C0, SEND_SIGNALING_PCAS_B, {
         .invalidate = INVALIDATE_TRUE,
         .schedule = SCHEDULE_TRUE
      });
   } else {
      P_IMMD(p, NVC6C0, SEND_SIGNALING_PCAS2_B,
             PCAS_ACTION_INVALIDATE_COPY_SCHEDULE);
   }
}

KGP_API void
kgpDispatch(KgpCommandBuffer cb, void *dataGpu, KgpUvec3 groups)
{
   kgp_cb_dispatch(cb, dataGpu, groups, 0);
}

KGP_API void
kgpDispatchIndirect(KgpCommandBuffer cb, void *dataGpu, void *groupsGpu)
{
   if (unlikely(groupsGpu == NULL)) {
      kgp_err("kgpDispatchIndirect: groupsGpu is NULL");
      cb->failed = true;
      return;
   }
   kgp_cb_dispatch(cb, dataGpu, (KgpUvec3) { 0, 0, 0 }, (uintptr_t)groupsGpu);
}
