/*
 * KGP_DEBUG=1: the validation mode of the proposal's section 7.
 *
 * The proposal interposes a dispatch table. KoreGP has none: its entry
 * points are plain exported functions. So each one that takes a GPU pointer
 * asks kgp_check_ptr (kgp_private.h), which is one branch on a flag of the
 * command buffer when the mode is off.
 *
 * Checked:
 *  - every GPU pointer argument lies, with the bytes the call will touch
 *    where that is known at record time, inside live kgpMalloc memory;
 *  - root pointers are KGP_ROOT_ALIGN_B aligned;
 *  - a heap's `count` entries fit in its allocation;
 *  - kgpBarrier's masks name stages and hazards that exist, and a stage on
 *    each side;
 *  - at kgpSubmit, that nothing a command buffer was recorded against has
 *    been freed since.
 *
 * Heap indices are the shaders' to compute, so that check is compiled into
 * the shaders of pipelines created under validation (kgp_nir_lower_heap.c):
 * an index past the entries the command buffer gave the heap reads zero,
 * and a write through it is dropped.
 *
 * Not checked, because nothing here sees it: pointers stored in memory
 * (root structs, descriptors in a heap), anything else the GPU computes
 * (indirect counts), and handles (a destroyed texture or pipeline).
 *
 * SPDX-License-Identifier: MIT
 */
#include "kgp_private.h"

#include <inttypes.h>

/* One allocation a command buffer was recorded against. The id tells a
 * freed allocation from a later one at the same address.
 */
struct kgp_ref {
   uint64_t addr;
   uint64_t id;
};

bool
kgp_validate_range(struct KgpDevice_T *dev, struct util_dynarray *refs,
                   const char *what, uint64_t addr, uint64_t size_B,
                   uint32_t align_B)
{
   /* Where NULL is not allowed the entry point says so itself. */
   if (addr == 0)
      return true;

   if (addr % align_B != 0) {
      kgp_err("validation: %s 0x%" PRIx64 " is not %u-byte aligned", what,
              addr, align_B);
      return false;
   }

   simple_mtx_lock(&dev->mutex);
   const struct kgp_alloc *a = kgp_alloc_at_gpu_locked(dev, addr);
   const uint64_t offset_B = a ? addr - a->mem->va->addr : 0;
   const uint64_t req_B = a ? a->req_B : 0;
   const uint64_t id = a ? a->id : 0;
   simple_mtx_unlock(&dev->mutex);

   if (a == NULL) {
      kgp_err("validation: %s 0x%" PRIx64 " is not in live kgpMalloc memory "
              "(freed, never allocated, or a CPU pointer that did not go "
              "through kgpHostToDevicePointer)", what, addr);
      return false;
   }
   size_B = MAX2(size_B, 1);
   if (offset_B >= req_B || size_B > req_B - offset_B) {
      kgp_err("validation: %s 0x%" PRIx64 " with %" PRIu64 " bytes runs past "
              "the end of its allocation (%" PRIu64 " bytes, pointer at "
              "offset %" PRIu64 ")", what, addr, size_B, req_B, offset_B);
      return false;
   }

   if (refs != NULL) {
      const unsigned n = util_dynarray_num_elements(refs, struct kgp_ref);
      if (n == 0 || util_dynarray_element(refs, struct kgp_ref, n - 1)->id != id) {
         const struct kgp_ref ref = { .addr = addr, .id = id };
         util_dynarray_append(refs, ref);
      }
   }
   return true;
}

bool
kgp_cb_validate_ptr(struct KgpCommandBuffer_T *cb, const char *what,
                    uint64_t addr, uint64_t size_B, uint32_t align_B)
{
   if (kgp_validate_range(cb->queue->dev, &cb->refs, what, addr, size_B,
                          align_B))
      return true;
   cb->failed = true;
   return false;
}

bool
kgp_cb_validate_texture(struct KgpCommandBuffer_T *cb, const char *what,
                        const struct KgpTexture_T *tex)
{
   /* A swapchain's texture is over the swapchain's memory, not the caller's. */
   if (tex == NULL || tex->owned_mem != NULL)
      return true;
   return kgp_cb_validate_ptr(cb, what, tex->mem_addr, tex->image.size_B, 1);
}

bool
kgp_cb_validate_barrier(struct KgpCommandBuffer_T *cb, uint32_t src,
                        uint32_t dst, uint32_t hazards)
{
   const uint32_t stages = KGP_STAGE_RASTER_DEPTH_OUT * 2 - 1;
   const uint32_t known_hazards = KGP_HAZARD_DEPTH_STENCIL * 2 - 1;

   if (src != 0 && dst != 0 && !((src | dst) & ~stages) &&
       !(hazards & ~known_hazards))
      return true;

   kgp_err("validation: kgpBarrier(0x%x, 0x%x, 0x%x): each side needs at "
           "least one stage, and only stages and hazards that exist", src,
           dst, hazards);
   cb->failed = true;
   return false;
}

/* Has anything this command buffer was recorded against been freed? */
bool
kgp_cb_validate_submit(struct KgpCommandBuffer_T *cb, uint32_t index)
{
   struct KgpDevice_T *dev = cb->queue->dev;
   uint64_t freed = 0;

   simple_mtx_lock(&dev->mutex);
   util_dynarray_foreach(&cb->refs, struct kgp_ref, ref) {
      const struct kgp_alloc *a = kgp_alloc_at_gpu_locked(dev, ref->addr);
      if (a == NULL || a->id != ref->id) {
         freed = ref->addr;
         break;
      }
   }
   simple_mtx_unlock(&dev->mutex);

   if (freed == 0)
      return true;
   kgp_err("validation: kgpSubmit: command buffer %u uses memory at 0x%"
           PRIx64 " that was freed after it was recorded", index, freed);
   return false;
}
