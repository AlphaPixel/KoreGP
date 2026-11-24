/*
 * GPU-driven multi-draw with per-draw root data (plan N3, task R9b): one
 * macro call that reads a draw count and that many KgpDrawIndexedArgs from
 * GPU memory and, for draw i, points the two stage roots at
 * vsData + i * vsStride and psData + i * psStride before drawing.
 *
 * This is the first macro that is not NVK's: Vulkan has no per-draw root
 * data, so NVK has nothing to extract whole. It is put together from NVK's
 * parts (all in nvk_cmd_draw.c, Turing branches only):
 *   nvk_mme_draw_indexed_indirect_count   the count read and the loop
 *   nvk_mme_build_draw_indexed            the five arguments
 *   nvk_mme_build_set_draw_params         base vertex, base instance, draw id
 *   nvk_mme_set_cb0_mthd                  a value into its register and into
 *                                         root table 0, only if it changed
 *   nvk_mme_build_draw_indexed_loop       the draw itself
 * Cut: multiview. New: the four root-table words per draw.
 *
 * The offsets into root table 0 are those of struct
 * nvk_root_descriptor_table, which NVK's own draw macros have baked in and
 * kgp_nir_lower_root.c reads back; that header cannot be included here.
 *
 * tests/mme_multi_sim.sh runs this on Mesa's simulator of the macro
 * processor.
 *
 * SPDX-License-Identifier: MIT
 */
#include "kgp_mme.h"
#include "kgp_private.h"

#include "mme_builder.h"
#include "nv_push_cl9097.h"
#include "nv_push_clc597.h"

#define KGP_ROOT0_BASE_VERTEX   0
#define KGP_ROOT0_BASE_INSTANCE 4
#define KGP_ROOT0_DRAW_INDEX    20

static void
kgp_mme_set_root0(struct mme_builder *b, uint16_t offset, uint16_t mthd,
                  struct mme_value val)
{
   struct mme_value old = mme_state(b, mthd);
   mme_if(b, ine, old, val) {
      mme_mthd(b, mthd);
      mme_emit(b, val);

      uint32_t root_table_selector;
      V_NVC597_SET_ROOT_TABLE_SELECTOR(root_table_selector, {
         .root_table = 0,
         .offset = offset,
      });

      mme_mthd(b, NVC597_SET_ROOT_TABLE_SELECTOR);
      mme_emit(b, mme_imm(root_table_selector));
      mme_mthd(b, NVC597_LOAD_ROOT_TABLE);
      mme_emit(b, val);
   }
   mme_free_reg(b, old);
}

void
kgp_mme_draw_indexed_multi(struct mme_builder *b)
{
   /* Every parameter before the first read from memory: what the reads
    * return arrives through the same FIFO.
    */
   struct mme_value64 draw_addr = mme_load_addr64(b);
   struct mme_value64 draw_count_addr = mme_load_addr64(b);
   struct mme_value draw_max = mme_load(b);
   struct mme_value64 vs_root = mme_load_addr64(b);
   struct mme_value vs_stride = mme_load(b);
   struct mme_value64 ps_root = mme_load_addr64(b);
   struct mme_value ps_stride = mme_load(b);

   mme_tu104_read_fifoed(b, draw_count_addr, mme_imm(1));
   mme_free_reg64(b, draw_count_addr);
   struct mme_value draw_count_buf = mme_load(b);

   mme_if(b, ule, draw_count_buf, draw_max) {
      mme_mov_to(b, draw_max, draw_count_buf);
   }
   mme_free_reg(b, draw_count_buf);

   uint32_t stage_roots_selector;
   V_NVC597_SET_ROOT_TABLE_SELECTOR(stage_roots_selector, {
      .root_table = KGP_GFX_ROOT_TABLE,
      .offset = 0,
   });
   uint32_t draw_control_a;
   V_NVC597_SET_DRAW_CONTROL_A(draw_control_a, {
      .primitive_id = PRIMITIVE_ID_FIRST,
      .instance_id = INSTANCE_ID_FIRST,
      .split_mode = SPLIT_MODE_NORMAL_BEGIN_NORMAL_END,
      .instance_iterate_enable = true,
   });

   struct mme_value draw = mme_mov(b, mme_zero());
   mme_while(b, ult, draw, draw_max) {
      /* The two stage roots, where the shaders read them: c[31][0] and
       * c[31][8]. The table offset advances with each word loaded.
       */
      mme_mthd(b, NVC597_SET_ROOT_TABLE_SELECTOR);
      mme_emit(b, mme_imm(stage_roots_selector));
      mme_mthd(b, NVC597_LOAD_ROOT_TABLE);
      mme_emit(b, vs_root.lo);
      mme_mthd(b, NVC597_LOAD_ROOT_TABLE);
      mme_emit(b, vs_root.hi);
      mme_mthd(b, NVC597_LOAD_ROOT_TABLE);
      mme_emit(b, ps_root.lo);
      mme_mthd(b, NVC597_LOAD_ROOT_TABLE);
      mme_emit(b, ps_root.hi);

      /* KgpDrawIndexedArgs, in its order */
      mme_tu104_read_fifoed(b, draw_addr, mme_imm(5));
      struct mme_value index_count = mme_load(b);
      struct mme_value instance_count = mme_load(b);
      struct mme_value first_index = mme_load(b);
      struct mme_value vertex_offset = mme_load(b);
      struct mme_value first_instance = mme_load(b);

      kgp_mme_set_root0(b, KGP_ROOT0_BASE_VERTEX,
         NV9097_SET_MME_SHADOW_SCRATCH(NVK_MME_SCRATCH_CB0_FIRST_VERTEX),
         vertex_offset);
      kgp_mme_set_root0(b, KGP_ROOT0_BASE_INSTANCE,
         NV9097_SET_GLOBAL_BASE_INSTANCE_INDEX, first_instance);
      kgp_mme_set_root0(b, KGP_ROOT0_DRAW_INDEX,
         NV9097_SET_MME_SHADOW_SCRATCH(NVK_MME_SCRATCH_CB0_DRAW_INDEX),
         draw);

      mme_mthd(b, NV9097_SET_GLOBAL_BASE_VERTEX_INDEX);
      mme_emit(b, vertex_offset);
      mme_mthd(b, NV9097_SET_VERTEX_ID_BASE);
      mme_emit(b, vertex_offset);

      mme_free_reg(b, vertex_offset);
      mme_free_reg(b, first_instance);

      mme_mthd(b, NVC597_SET_DRAW_CONTROL_A);
      mme_emit(b, mme_imm(draw_control_a));
      mme_emit(b, instance_count);
      mme_emit(b, first_index);
      mme_emit(b, index_count);

      mme_free_reg(b, instance_count);
      mme_free_reg(b, first_index);
      mme_free_reg(b, index_count);

      mme_add_to(b, draw, draw, mme_imm(1));
      mme_add64_to(b, draw_addr, draw_addr,
                   mme_imm64(sizeof(KgpDrawIndexedArgs)));
      mme_add64_to(b, vs_root, vs_root, mme_value64(vs_stride, mme_zero()));
      mme_add64_to(b, ps_root, ps_root, mme_value64(ps_stride, mme_zero()));
   }
}
