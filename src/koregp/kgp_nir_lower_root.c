/*
 * The root convention (delta D5 in docs/nvk-mapping.md).
 *
 * SPIR-V spells the root pointer as a push-constant block whose first
 * member is a PhysicalStorageBuffer pointer. There are no push constants on
 * this path: kgpDispatch writes a struct kgp_root_record and binds it as
 * constant buffer 0, and this pass turns the shader's push-constant loads
 * into loads from it. The dereference of the pointer itself is an ordinary
 * 64-bit global load that NAK already handles for buffer device address.
 *
 * Ancestor: nvk_nir_lower_descriptors.c, of which only lower_load_constant,
 * lower_load_push_constant and the root-table system values survive.
 *
 * SPDX-License-Identifier: MIT
 */
#include "kgp_private.h"

#include "nir.h"
#include "nir_builder.h"

/* A system value NVK's draw macros keep in hardware root table 0. */
static nir_def *
load_draw_param(nir_builder *b, nir_intrinsic_instr *intrin, uint32_t offset)
{
   return nir_ldc_nv(b, intrin->def.num_components, intrin->def.bit_size,
                     nir_imm_int(b, KGP_GFX_CBUF_DRAW_PARAMS),
                     nir_imm_int(b, 0),
                     .base = offset, .align_mul = 4, .align_offset = 0);
}

static bool
lower_root_intrin(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   const mesa_shader_stage stage = b->shader->info.stage;
   const bool compute = stage == MESA_SHADER_COMPUTE;
   bool *bad_push = data;
   nir_def *val;

   b->cursor = nir_before_instr(&intrin->instr);

   switch (intrin->intrinsic) {
   case nir_intrinsic_load_push_constant: {
      /* Compute: the one root pointer, in the root record (cbuf 0).
       * Raster (delta D10): the vertex stage's pointer at offset 0 and the
       * pixel stage's at offset 8, in the hardware root table where NVK
       * keeps Vulkan push constants.
       */
      const uint32_t first =
         stage == MESA_SHADER_FRAGMENT ? KGP_ROOT_PUSH_SIZE_B : 0;
      const uint32_t base = nir_intrinsic_base(intrin);
      const uint32_t load_B =
         intrin->def.num_components * intrin->def.bit_size / 8;
      if (!nir_src_is_const(intrin->src[0])) {
         *bad_push = true;
      } else {
         const uint64_t offset = nir_src_as_uint(intrin->src[0]) + base;
         if (offset < first || offset + load_B > first + KGP_ROOT_PUSH_SIZE_B)
            *bad_push = true;
      }

      val = nir_ldc_nv(b, intrin->def.num_components, intrin->def.bit_size,
                       nir_imm_int(b, compute ? KGP_CBUF_ROOT
                                              : KGP_GFX_CBUF_ROOT),
                       nir_iadd_imm(b, intrin->src[0].ssa, base),
                       .align_mul = nir_intrinsic_align_mul(intrin),
                       .align_offset = nir_intrinsic_align_offset(intrin));
      break;
   }

   /* Ancestor: lower_sysval_to_root_table, draw.* members. The offsets are
    * those of struct nvk_root_descriptor_table, which the draw macros have
    * baked in.
    */
   case nir_intrinsic_load_base_vertex:
   case nir_intrinsic_load_first_vertex:
      if (compute)
         return false;
      val = load_draw_param(b, intrin, 0);
      break;
   case nir_intrinsic_load_base_instance:
      if (compute)
         return false;
      val = load_draw_param(b, intrin, 4);
      break;
   case nir_intrinsic_load_draw_id:
      if (compute)
         return false;
      val = load_draw_param(b, intrin, 20);
      break;
   case nir_intrinsic_load_view_index:
      if (compute)
         return false;
      val = load_draw_param(b, intrin, 24);
      break;

   case nir_intrinsic_load_num_workgroups:
      if (!compute) {
         /* A mesh shader's: draw.mesh.group_count, which NVK's mesh draw
          * macros write.
          */
         val = load_draw_param(b, intrin, 8);
         break;
      }
      val = nir_ldc_nv(b, 3, 32,
                       nir_imm_int(b, KGP_CBUF_ROOT), nir_imm_int(b, 0),
                       .base = offsetof(struct kgp_root_record, group_count),
                       .align_mul = 4, .align_offset = 0);
      break;

   case nir_intrinsic_load_constant:
      /* nir_opt_large_constants output: lives after the shader code. */
      val = nir_ldc_nv(b, intrin->def.num_components, intrin->def.bit_size,
                       nir_imm_int(b, KGP_CBUF_SHADER_DATA),
                       nir_iadd_imm(b, intrin->src[0].ssa,
                                    nir_intrinsic_base(intrin)),
                       .align_mul = nir_intrinsic_align_mul(intrin),
                       .align_offset = nir_intrinsic_align_offset(intrin));
      break;

   default:
      return false;
   }

   nir_def_rewrite_uses(&intrin->def, val);
   nir_instr_remove(&intrin->instr);
   return true;
}

/* Sets *bad_push_out if the shader reads push constants beyond the root
 * pointer; the caller rejects the shader.
 */
bool
kgp_nir_lower_root(nir_shader *nir, bool *bad_push_out)
{
   return nir_shader_intrinsics_pass(nir, lower_root_intrin,
                                     nir_metadata_control_flow, bad_push_out);
}
