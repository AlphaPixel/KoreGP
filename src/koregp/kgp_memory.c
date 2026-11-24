/*
 * kgpMalloc / kgpFree / kgpHostToDevicePointer.
 *
 * nvkmd does the real work (GEM object + VA + bind in one call) and is
 * called, not copied. What is ours: the pointer-keyed lookup the proposal
 * asks for, and deferring destruction against the device timeline.
 *
 * SPDX-License-Identifier: MIT
 */
#include "kgp_private.h"

#include "util/u_math.h"

#include <stdlib.h>

/* rb_tree comparison convention: negative means "go left". */
static int
kgp_alloc_cmp_insert(const struct rb_node *tree_node,
                     const struct rb_node *new_node)
{
   const struct kgp_alloc *t = rb_node_data(struct kgp_alloc, tree_node, node);
   const struct kgp_alloc *n = rb_node_data(struct kgp_alloc, new_node, node);
   return n->key < t->key ? -1 : n->key > t->key;
}

static int
kgp_alloc_cmp_search(const struct rb_node *tree_node, const void *key)
{
   const struct kgp_alloc *t = rb_node_data(struct kgp_alloc, tree_node, node);
   const uintptr_t k = (uintptr_t)key;
   if (k < t->key)
      return -1;
   return k - t->key >= t->size_B;
}

static struct kgp_alloc *
kgp_alloc_lookup_locked(struct KgpDevice_T *dev, unsigned tree, const void *ptr)
{
   struct rb_node *n =
      rb_tree_search(&dev->allocs[tree], ptr, kgp_alloc_cmp_search);
   return n ? rb_node_data(struct kgp_alloc, n, node) : NULL;
}

static void
kgp_alloc_destroy(struct kgp_alloc *a)
{
   nvkmd_mem_unref(a->mem);
   free(a);
}

/* Memory with no entry in the pointer trees (shader code). */
struct kgp_alloc *
kgp_alloc_create_mapped(struct KgpDevice_T *dev, uint64_t size_B,
                        enum nvkmd_mem_flags flags,
                        enum nvkmd_mem_map_flags map_flags)
{
   struct kgp_alloc *a = calloc(1, sizeof(*a));
   if (a == NULL)
      return NULL;

   if (nvkmd_dev_alloc_mapped_mem(dev->nvkmd, KGP_LOG_OBJ, size_B, 0,
                                  flags, map_flags, &a->mem) != VK_SUCCESS) {
      free(a);
      return NULL;
   }
   a->size_B = a->mem->size_B;
   return a;
}

/* Destroys the allocation once everything submitted so far has finished. */
void
kgp_alloc_free_deferred(struct KgpDevice_T *dev, struct kgp_alloc *a)
{
   simple_mtx_lock(&dev->mutex);
   a->free_seq = dev->submit_seq;
   list_addtail(&a->link, &dev->pending_free);
   kgp_device_poll_locked(dev);
   kgp_memory_reap_locked(dev);
   simple_mtx_unlock(&dev->mutex);
}

KGP_API void *
kgpMalloc(KgpDevice dev, size_t bytes, size_t align, KGP_MEMORY type)
{
   if (bytes == 0 || !util_is_power_of_two_or_zero64(align)) {
      kgp_err("kgpMalloc: size must be non-zero and align a power of two");
      return NULL;
   }

   struct kgp_alloc *a = calloc(1, sizeof(*a));
   if (a == NULL)
      return NULL;

   VkResult result;
   switch (type) {
   case KGP_MEMORY_DEFAULT:
      result = nvkmd_dev_alloc_mapped_mem(dev->nvkmd, KGP_LOG_OBJ, bytes, align,
                                          NVKMD_MEM_LOCAL, NVKMD_MEM_MAP_RDWR,
                                          &a->mem);
      break;
   case KGP_MEMORY_READBACK:
      result = nvkmd_dev_alloc_mapped_mem(dev->nvkmd, KGP_LOG_OBJ, bytes, align,
                                          NVKMD_MEM_GART, NVKMD_MEM_MAP_RDWR,
                                          &a->mem);
      break;
   case KGP_MEMORY_GPU:
      result = nvkmd_dev_alloc_mem(dev->nvkmd, KGP_LOG_OBJ, bytes, align,
                                   NVKMD_MEM_VRAM, &a->mem);
      break;
   default:
      kgp_err("kgpMalloc: unknown memory type %d", type);
      result = VK_ERROR_UNKNOWN;
      break;
   }
   if (result != VK_SUCCESS) {
      free(a);
      return NULL;
   }

   const bool gpu_only = type == KGP_MEMORY_GPU;
   a->key = gpu_only ? a->mem->va->addr : (uintptr_t)a->mem->map;
   a->size_B = a->mem->size_B;
   a->req_B = bytes;

   simple_mtx_lock(&dev->mutex);
   a->id = ++dev->next_alloc_id;
   rb_tree_insert(&dev->allocs[gpu_only], &a->node, kgp_alloc_cmp_insert);
   simple_mtx_unlock(&dev->mutex);

   return (void *)a->key;
}

KGP_API void
kgpFree(KgpDevice dev, void *ptr)
{
   if (ptr == NULL)
      return;

   simple_mtx_lock(&dev->mutex);

   /* CPU pointers and GPU-only pointers are different address spaces that
    * share one void*. Look in the CPU tree first; it is the common case.
    */
   unsigned tree = 0;
   struct kgp_alloc *a = kgp_alloc_lookup_locked(dev, 0, ptr);
   if (a == NULL) {
      tree = 1;
      a = kgp_alloc_lookup_locked(dev, 1, ptr);
   }
   if (a == NULL || a->key != (uintptr_t)ptr) {
      simple_mtx_unlock(&dev->mutex);
      kgp_err("kgpFree(%p): not a pointer returned by kgpMalloc", ptr);
      return;
   }
   rb_tree_remove(&dev->allocs[tree], &a->node);
   simple_mtx_unlock(&dev->mutex);

   /* Anything submitted so far may still read or write this memory. */
   kgp_alloc_free_deferred(dev, a);
}

KGP_API void *
kgpHostToDevicePointer(KgpDevice dev, void *cpuPtr)
{
   void *gpu = NULL;

   simple_mtx_lock(&dev->mutex);
   struct kgp_alloc *a = kgp_alloc_lookup_locked(dev, 0, cpuPtr);
   if (a != NULL)
      gpu = (void *)(a->mem->va->addr + ((uintptr_t)cpuPtr - a->key));
   else if (kgp_alloc_lookup_locked(dev, 1, cpuPtr) != NULL)
      gpu = cpuPtr;   /* already a GPU pointer */
   simple_mtx_unlock(&dev->mutex);

   return gpu;
}

/* The memory object behind a KGP_MEMORY_GPU pointer, and the offset into it. */
struct nvkmd_mem *
kgp_gpu_only_mem(struct KgpDevice_T *dev, uint64_t addr, uint64_t *offset_B_out)
{
   struct nvkmd_mem *mem = NULL;

   simple_mtx_lock(&dev->mutex);
   struct kgp_alloc *a = kgp_alloc_lookup_locked(dev, 1, (void *)(uintptr_t)addr);
   if (a != NULL) {
      mem = a->mem;
      *offset_B_out = addr - a->key;
   }
   simple_mtx_unlock(&dev->mutex);
   return mem;
}

/* The live allocation holding GPU address `addr`: validation only.
 * ponytail: mapped memory is keyed by CPU pointer, so this walks all of
 * it; a tree keyed by VA when a debug run is slow because of it.
 */
const struct kgp_alloc *
kgp_alloc_at_gpu_locked(struct KgpDevice_T *dev, uint64_t addr)
{
   const struct kgp_alloc *a =
      kgp_alloc_lookup_locked(dev, 1, (void *)(uintptr_t)addr);
   if (a != NULL)
      return a;

   rb_tree_foreach(struct kgp_alloc, m, &dev->allocs[0], node) {
      if (addr - m->mem->va->addr < m->size_B)
         return m;
   }
   return NULL;
}

void
kgp_memory_reap_locked(struct KgpDevice_T *dev)
{
   /* The list is in free order, so free_seq is non-decreasing. */
   list_for_each_entry_safe(struct kgp_alloc, a, &dev->pending_free, link) {
      if (a->free_seq > dev->done_seq)
         break;
      list_del(&a->link);
      kgp_alloc_destroy(a);
   }
}

void
kgp_memory_finish(struct KgpDevice_T *dev)
{
   assert(list_is_empty(&dev->pending_free));

   for (unsigned tree = 0; tree < 2; tree++) {
      rb_tree_foreach_safe(struct kgp_alloc, a, &dev->allocs[tree], node) {
         rb_tree_remove(&dev->allocs[tree], &a->node);
         kgp_alloc_destroy(a);
      }
   }

   if (dev->slm_mem)
      nvkmd_mem_unref(dev->slm_mem);
   util_dynarray_foreach(&dev->old_slm_mems, struct nvkmd_mem *, mem)
      nvkmd_mem_unref(*mem);
   util_dynarray_fini(&dev->old_slm_mems);
}
