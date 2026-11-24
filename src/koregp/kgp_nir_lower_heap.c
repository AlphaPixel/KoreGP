/*
 * The heap convention: textures are indexed, not bound.
 *
 * SPIR-V declares the texture heap as a runtime array at set 0, binding 0
 * (texture*, or image* for storage access: one descriptor serves both) and
 * the sampler heap at set 0, binding 1. On NVIDIA a shader's texture handle
 * already is "index into the texture header pool | index into the sampler
 * pool << 20", so the array index goes straight into the instruction; there
 * is no descriptor to load and no table to look in.
 *
 * Ancestor: nvk_nir_lower_descriptors.c lower_tex / lower_image_intrin,
 * where the handle is loaded from a descriptor set instead.
 *
 * SPDX-License-Identifier: MIT
 */
#include "kgp_private.h"

#include "nir.h"
#include "nir_builder.h"

/* Ancestor: NVK_IMAGE_DESCRIPTOR_{IMAGE,SAMPLER}_INDEX_MASK; NAK's
 * nak_nir_lower_tex.c hard-codes the same split.
 */
#define KGP_HANDLE_TEXTURE_MASK 0x000fffff
#define KGP_HANDLE_SAMPLER_SHIFT 20

/* Returns the 32-bit heap index a deref selects, or NULL (and sets *bad)
 * if it is not into the heap at `binding`.
 */
static nir_def *
heap_index(nir_builder *b, nir_deref_instr *deref, uint32_t binding, bool *bad)
{
   nir_def *index;
   if (deref->deref_type == nir_deref_type_array) {
      index = nir_u2u32(b, deref->arr.index.ssa);
      deref = nir_deref_instr_parent(deref);
   } else {
      index = nir_imm_int(b, 0);
   }

   if (deref->deref_type != nir_deref_type_var ||
       deref->var->data.descriptor_set != 0 ||
       deref->var->data.binding != binding) {
      *bad = true;
      return NULL;
   }
   return index;
}

static bool
lower_heap_tex(nir_builder *b, nir_tex_instr *tex, bool *bad)
{
   nir_deref_instr *texture =
      nir_steal_tex_deref(tex, nir_tex_src_texture_deref);
   nir_deref_instr *sampler =
      nir_steal_tex_deref(tex, nir_tex_src_sampler_deref);
   if (texture == NULL)
      return false;

   b->cursor = nir_before_instr(&tex->instr);

   nir_def *handle = heap_index(b, texture, KGP_HEAP_BINDING_TEXTURES, bad);
   if (handle == NULL)
      return false;
   handle = nir_iand_imm(b, handle, KGP_HANDLE_TEXTURE_MASK);

   /* Fetches and queries take no sampler on Maxwell and later. */
   if (sampler != NULL && nir_tex_instr_need_sampler(tex)) {
      nir_def *sampler_index =
         heap_index(b, sampler, KGP_HEAP_BINDING_SAMPLERS, bad);
      if (sampler_index == NULL)
         return false;
      handle = nir_ior(b, handle,
                       nir_ishl_imm(b, sampler_index,
                                    KGP_HANDLE_SAMPLER_SHIFT));
   }

   nir_tex_instr_add_src(tex, nir_tex_src_texture_handle, handle);
   return true;
}

static bool
lower_heap_instr(nir_builder *b, nir_instr *instr, void *data)
{
   bool *bad = data;

   if (instr->type == nir_instr_type_tex)
      return lower_heap_tex(b, nir_instr_as_tex(instr), bad);

   if (instr->type != nir_instr_type_intrinsic)
      return false;

   nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
   switch (intrin->intrinsic) {
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_store:
   case nir_intrinsic_image_deref_atomic:
   case nir_intrinsic_image_deref_atomic_swap:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_samples: {
      b->cursor = nir_before_instr(instr);
      nir_def *handle = heap_index(b, nir_src_as_deref(intrin->src[0]),
                                   KGP_HEAP_BINDING_TEXTURES, bad);
      if (handle == NULL)
         return false;
      nir_rewrite_image_intrinsic(intrin, handle,
                                  nir_image_intrinsic_type_bindless);
      return true;
   }
   default:
      return false;
   }
}

/* Sets *bad_out if the shader reaches a texture, image or sampler that is
 * not in a heap; the caller rejects the shader.
 */
bool
kgp_nir_lower_heap(nir_shader *nir, bool *bad_out)
{
   return nir_shader_instructions_pass(nir, lower_heap_instr,
                                       nir_metadata_control_flow, bad_out);
}
