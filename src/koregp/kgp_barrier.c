/*
 * kgpBarrier: stage pair + hazards -> cache flush/invalidate methods.
 *
 * Ancestors: nvk_cmd_buffer.c:nvk_cmd_flush_wait_dep (producer side) and
 * nvk_cmd_invalidate_deps (consumer side). NVK first reduces a Vulkan
 * dependency (stage masks, access masks, per-resource lists) to a handful
 * of nvk_barrier flags; here the stage pair selects those flags directly.
 * With no access masks to narrow it, a shader stage is assumed to have
 * written, and to be about to read, anything a shader can.
 *
 * One place departs from NVK's mapping. NVK's main queue copies with
 * internal shaders, so there a transfer stage flushes and invalidates like a
 * shader stage. KoreGP's transfer stage is the copy engine (kgp_copy.c): as
 * a producer it needs only to finish, which is what NVK does on its
 * transfer-only queue; as a consumer it reads memory directly and has no
 * cache to invalidate.
 *
 * A stage names an engine: compute, the 3D engine (vertex and pixel
 * shaders, and the render targets they write), or the copy engine. Each has
 * its own caches, so producer and consumer are handled per engine.
 *
 * SPDX-License-Identifier: MIT
 */
#include "kgp_private.h"

#include "nv_push_cl9097.h"
#include "nv_push_cl90b5.h"
#include "nv_push_cla097.h"
#include "nv_push_cla0c0.h"
#include "nv_push_clb1c0.h"
#include "nv_push_clc597.h"

#include "clc36f.h"

#define KGP_RASTER_SHADER_STAGES \
   (KGP_STAGE_VERTEX_SHADER | KGP_STAGE_PIXEL_SHADER)
#define KGP_RASTER_OUT_STAGES \
   (KGP_STAGE_RASTER_COLOR_OUT | KGP_STAGE_RASTER_DEPTH_OUT)

/* Producer: finish, and get what it wrote out of the write-back caches. */
static void
kgp_barrier_producer(struct nv_push *p, KGP_STAGE src)
{
   if (src & KGP_STAGE_TRANSFER) {
      /* NVK_BARRIER_WFI as NVK spells it on the copy subchannel: an empty
       * transfer that is not pipelined, so it waits for those before it.
       * (Moving on to another subchannel would wait too; a second copy
       * would not.)
       */
      P_MTHD(p, NV90B5, LINE_LENGTH_IN);
      P_NV90B5_LINE_LENGTH_IN(p, 0);
      P_NV90B5_LINE_COUNT(p, 0);

      P_IMMD(p, NV90B5, LAUNCH_DMA, {
         .data_transfer_type = DATA_TRANSFER_TYPE_NON_PIPELINED,
         .multi_line_enable = false,
         .flush_enable = FLUSH_ENABLE_TRUE,
         /* Note: FLUSH_TYPE=SYS implicitly for NVC3B5+ */
         .src_memory_layout = SRC_MEMORY_LAYOUT_PITCH,
         .dst_memory_layout = DST_MEMORY_LAYOUT_PITCH,
         .remap_enable = REMAP_ENABLE_TRUE,
      });
   }
   if (src & KGP_STAGE_COMPUTE) {
      /* NVK_BARRIER_FLUSH_SHADER_DATA. This is also implicitly a WFI. */
      P_IMMD(p, NVA0C0, INVALIDATE_SHADER_CACHES, {
         .data = DATA_TRUE,
         .flush_data = FLUSH_DATA_TRUE,
      });
   }
   if (src & KGP_RASTER_SHADER_STAGES) {
      /* The same for memory a vertex or pixel shader wrote. */
      P_IMMD(p, NVA097, INVALIDATE_SHADER_CACHES, {
         .data = DATA_TRUE,
         .flush_data = FLUSH_DATA_TRUE,
      });
   } else if (src & KGP_RASTER_OUT_STAGES) {
      /* NVK_BARRIER_WFI on the 3D engine: render targets are written. */
      P_IMMD(p, NV9097, WAIT_FOR_IDLE, 0);
   }
   if (!(src & (KGP_STAGE_TRANSFER | KGP_STAGE_COMPUTE |
                KGP_RASTER_SHADER_STAGES | KGP_RASTER_OUT_STAGES))) {
      /* NVK_BARRIER_WFI */
      P_IMMD(p, NVA0C0, WAIT_FOR_IDLE, 0);
   }
}

static void
kgp_barrier_consumer(struct nv_push *p, KGP_STAGE dst, KGP_HAZARD hazards)
{
   /* Which engine the consumer is: compute, the 3D engine's shaders, both.
    * The copy engine as a consumer needs nothing (see the header comment).
    */
   const bool dst_gfx = dst & KGP_RASTER_SHADER_STAGES;
   const bool dst_cs = dst & KGP_STAGE_COMPUTE;
   const bool reads_args = hazards & KGP_HAZARD_DRAW_ARGUMENTS;
   /* Arguments of an indirect dispatch, or of nothing in particular. */
   const bool reads_args_cs = reads_args && (dst_cs || !dst_gfx);

   /* Consumer: drop what it may have cached of the producer's targets. */
   if (dst_cs) {
      /* NVK_BARRIER_INVALIDATE_TEX_DATA */
      P_IMMD(p, NVA0C0, INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI, {
         .lines = LINES_ALL,
      });
   }
   if (dst_cs || reads_args_cs) {
      /* NVK_BARRIER_INVALIDATE_SHADER_DATA for memory a shader reads.
       *
       * NVK_BARRIER_INVALIDATE_CONSTANT only with indirect arguments, as in
       * NVK: an indirect dispatch has the GPU rewrite the group counts
       * inside a constant buffer (the root record). Otherwise not: the
       * root record is the only constant buffer and is written fresh per
       * dispatch, exactly like NVK's root table, and the user's root
       * struct is read with global loads (delta D7).
       */
      P_IMMD(p, NVA0C0, INVALIDATE_SHADER_CACHES_NO_WFI, {
         .global_data = dst_cs,
         .constant = reads_args,
      });
   }
   if (dst_gfx) {
      /* The 3D engine's shaders have their own caches of both. */
      P_IMMD(p, NV9097, INVALIDATE_TEXTURE_DATA_CACHE_NO_WFI, {
         .lines = LINES_ALL,
      });
      P_IMMD(p, NVA097, INVALIDATE_SHADER_CACHES_NO_WFI, {
         .global_data = true,
         .constant = reads_args,
      });
   }

   if (reads_args_cs) {
      /* NVK_BARRIER_INVALIDATE_MME_DATA: indirect arguments are read by the
       * command fetcher, which runs ahead of execution. SET_REFERENCE makes
       * the host side wait for everything before it, so the fetcher sees
       * what the producer wrote.
       */
      __push_immd(p, SUBC_NV90C0, NV906F_SET_REFERENCE, 0);
      /* NVK_BARRIER_INVALIDATE_QMD_DATA */
      P_IMMD(p, NVB1C0, INVALIDATE_SKED_CACHES, 0);
   }
   if (reads_args && dst_gfx) {
      /* The same for an indirect draw, whose arguments the macro processor
       * reads: NVK's form, the reference on the 3D subchannel and a memory
       * barrier for the macro's reads.
       */
      __push_immd(p, SUBC_NV9097, NV906F_SET_REFERENCE, 0);
      P_IMMD(p, NVC597, MME_DMA_SYSMEMBAR, 0);
   }

   if (hazards & KGP_HAZARD_DESCRIPTORS) {
      if (dst_cs || !dst_gfx) {
         P_IMMD(p, NVA0C0, INVALIDATE_SAMPLER_CACHE_NO_WFI, {
            .lines = LINES_ALL,
         });
         P_IMMD(p, NVA0C0, INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI, {
            .lines = LINES_ALL,
         });
      }
      if (dst_gfx) {
         P_IMMD(p, NV9097, INVALIDATE_SAMPLER_CACHE_NO_WFI, {
            .lines = LINES_ALL,
         });
         P_IMMD(p, NV9097, INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI, {
            .lines = LINES_ALL,
         });
      }
   }

   /* KGP_HAZARD_DEPTH_STENCIL: nothing until there are depth targets. */
}

#define KGP_BARRIER_PUSH_DW 40

KGP_API void
kgpBarrier(KgpCommandBuffer cb, KGP_STAGE src, KGP_STAGE dst,
           KGP_HAZARD hazards)
{
   if (unlikely(cb->validate) &&
       !kgp_cb_validate_barrier(cb, src, dst, hazards))
      return;

   struct nv_push *p = kgp_cb_push(cb, KGP_BARRIER_PUSH_DW);
   kgp_barrier_producer(p, src);
   kgp_barrier_consumer(p, dst, hazards);
}

/* Split barriers ------------------------------------------------------------
 *
 * No ancestor: NVK does not use the channel's semaphore methods. They are
 * host methods (class C36F), valid on any subchannel: an address, a
 * payload and an operation. RELEASE with RELEASE_WFI stores the payload
 * once the engines are idle; ACQUIRE holds the channel until the memory
 * matches, yielding its timeslice meanwhile (ACQUIRE_SWITCH_TSG).
 */
/* NVC36F_SEM_EXECUTE's fields (clc36f.h gives them as bit ranges). */
#define KGP_SEM_OP(x) (NVC36F_SEM_EXECUTE_OPERATION_##x << 0)
#define KGP_SEM_SWITCH_TSG (1u << 12)
#define KGP_SEM_RELEASE_WFI (1u << 20)
#define KGP_SEM_PAYLOAD_64 (1u << 24)
#define KGP_SEM_REDUCTION(x) (NVC36F_SEM_EXECUTE_REDUCTION_##x << 27)
#define KGP_SEM_UNSIGNED (1u << 31)

static bool
kgp_sem_check(struct KgpCommandBuffer_T *cb, const char *fn, void *valueGpu)
{
   const uint64_t addr = (uintptr_t)valueGpu;
   if (unlikely(addr == 0 || addr % 8 != 0 || addr >> 40 != 0)) {
      kgp_err("%s: valueGpu must be an 8-byte aligned GPU pointer below 2^40",
              fn);
      cb->failed = true;
      return false;
   }
   return kgp_check_ptr(cb, fn, valueGpu, 8, 8);
}

static void
kgp_push_sem(struct nv_push *p, uint64_t addr, uint64_t value, uint32_t execute)
{
   __push_mthd(p, SUBC_NV90C0, NVC36F_SEM_ADDR_LO);
   P_INLINE_DATA(p, (uint32_t)addr);          /* OFFSET is bits 31:2 */
   P_INLINE_DATA(p, (uint32_t)(addr >> 32));
   P_INLINE_DATA(p, (uint32_t)value);
   P_INLINE_DATA(p, (uint32_t)(value >> 32));
   P_INLINE_DATA(p, execute);
}

KGP_API void
kgpSignalAfter(KgpCommandBuffer cb, KGP_STAGE src, void *valueGpu,
               uint64_t value, KGP_SIGNAL op)
{
   if (unlikely(op > KGP_SIGNAL_MAX)) {
      kgp_err("kgpSignalAfter: unknown KGP_SIGNAL");
      cb->failed = true;
      return;
   }
   if (!kgp_sem_check(cb, "kgpSignalAfter", valueGpu))
      return;

   struct nv_push *p = kgp_cb_push(cb, KGP_BARRIER_PUSH_DW + 6);
   kgp_barrier_producer(p, src);

   uint32_t execute =
      KGP_SEM_RELEASE_WFI |
      KGP_SEM_PAYLOAD_64;
   if (op == KGP_SIGNAL_MAX) {
      execute |= KGP_SEM_OP(REDUCTION) |
                 KGP_SEM_REDUCTION(IMAX) |
                 KGP_SEM_UNSIGNED;
   } else {
      execute |= KGP_SEM_OP(RELEASE);
   }
   kgp_push_sem(p, (uintptr_t)valueGpu, value, execute);
}

KGP_API void
kgpWaitBefore(KgpCommandBuffer cb, KGP_STAGE dst, void *valueGpu,
              uint64_t value, KGP_WAIT op, KGP_HAZARD hazards)
{
   if (unlikely(op > KGP_WAIT_GREATER_EQUAL)) {
      kgp_err("kgpWaitBefore: unknown KGP_WAIT");
      cb->failed = true;
      return;
   }
   if (!kgp_sem_check(cb, "kgpWaitBefore", valueGpu))
      return;

   struct nv_push *p = kgp_cb_push(cb, KGP_BARRIER_PUSH_DW + 6);
   kgp_push_sem(p, (uintptr_t)valueGpu, value,
                (op == KGP_WAIT_EQUAL
                    ? KGP_SEM_OP(ACQUIRE)
                    : KGP_SEM_OP(ACQ_STRICT_GEQ)) |
                KGP_SEM_SWITCH_TSG |
                KGP_SEM_PAYLOAD_64);
   kgp_barrier_consumer(p, dst, hazards);
}
