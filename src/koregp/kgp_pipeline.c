/*
 * Compute pipelines: SPIR-V -> NIR -> NAK -> GPU memory.
 *
 * Ancestors:
 *   vulkan/runtime/vk_nir.c  vk_spirv_to_nir            (kgp_spirv_to_nir)
 *   nvk_shader.c             nvk_lower_nir, _late       (kgp_lower_nir)
 *   nvk_shader.c             nvk_compile_nir, _upload   (kgpCreateComputePipeline)
 *   nvk_device.c             nvk_slm_area_ensure        (kgp_device_ensure_slm)
 * Cut: vk_shader, pipeline layouts, descriptor-set lowering, robustness
 * state, every non-compute stage, the shader heap, the pipeline cache.
 *
 * SPDX-License-Identifier: MIT
 */
#include "kgp_private.h"

#include "compiler/spirv/nir_spirv.h"
#include "compiler/spirv/spirv_info.h"
#include "nir.h"
#include "util/u_math.h"

#include <stdlib.h>

#define SPIR_V_MAGIC_NUMBER 0x07230203

/* The hardware may fetch instructions past the end of the shader. NVK
 * over-allocates its shader heap by the same amount.
 */
#define KGP_SHADER_OVERALLOC_B 2048

static void
kgp_spirv_debug(UNUSED void *private_data, enum nir_spirv_debug_level level,
                size_t spirv_offset, const char *message)
{
   if (level == NIR_SPIRV_DEBUG_LEVEL_ERROR)
      kgp_err("SPIR-V offset %zu: %s", spirv_offset, message);
   else if (level == NIR_SPIRV_DEBUG_LEVEL_WARNING)
      mesa_logw("koregp: SPIR-V offset %zu: %s", spirv_offset, message);
}

/* The caller's constants in the form spirv_to_nir takes; freed by the
 * caller. Ancestor: vk_spec_info_to_nir_spirv. False on a bad argument.
 */
static bool
kgp_spec_to_nir(const KgpSpecialization *in,
                struct nir_spirv_specialization **out)
{
   *out = NULL;
   if (in == NULL || in->count == 0)
      return true;
   if (in->constants == NULL)
      return false;
   *out = vtn_alloc_specialization(in->count);
   if (*out == NULL)
      return false;
   for (uint32_t i = 0; i < in->count; i++) {
      /* Eight bytes whatever the constant's type: the parser copies them
       * over a nir_const_value, whose members share their low bytes
       * (little endian), and the caller's value is zero-extended.
       * ponytail: scalar constants only; an array-typed one wants its
       * exact size and asserts in the parser.
       */
      if (!vtn_add_specialization_entry(*out, i, in->constants[i].id,
                                        sizeof(uint64_t),
                                        &in->constants[i].value, false)) {
         vtn_free_specialization(*out);
         *out = NULL;
         return false;
      }
   }
   return true;
}

/* Marks the entries whose id the module decorates a constant with.
 * spirv_to_nir applies the values but does not say which ids it met, and
 * Mesa's own check (spirv_verify_gl_specialization_constants) gives up on
 * Vulkan-flavoured modules.
 */
static void
kgp_spec_mark_defined(const uint32_t *words, size_t word_count,
                      struct nir_spirv_specialization *spec)
{
   for (size_t i = 5; i < word_count;) {
      const uint32_t len = words[i] >> 16, op = words[i] & 0xffff;
      if (len == 0 || len > word_count - i)
         return;   /* malformed: spirv_to_nir reports it */
      if (op == SpvOpDecorate && len == 4 &&
          words[i + 2] == SpvDecorationSpecId) {
         for (uint32_t e = 0; e < spec->num_entries; e++) {
            if (spec->entries[e].id == words[i + 3])
               spec->entries[e].defined_on_module = true;
         }
      }
      i += len;
   }
}

/* After every stage has been parsed: each id must have met a constant. */
static bool
kgp_spec_all_defined(const struct nir_spirv_specialization *spec)
{
   for (uint32_t i = 0; spec != NULL && i < spec->num_entries; i++) {
      if (!spec->entries[i].defined_on_module) {
         kgp_err("no specialization constant with id %u in the pipeline's "
                 "shaders", spec->entries[i].id);
         return false;
      }
   }
   return true;
}

static nir_shader *
kgp_spirv_to_nir(struct KgpDevice_T *dev, KgpByteSpan spirv,
                 mesa_shader_stage stage,
                 struct nir_spirv_specialization *spec)
{
   const uint32_t *words = spirv.data;
   if (spirv.size < 20 || spirv.size % 4 != 0 ||
       words[0] != SPIR_V_MAGIC_NUMBER) {
      kgp_err("not a SPIR-V module");
      return NULL;
   }

   /* What the root convention needs, plus the integer widths. Grows with
    * the API (RuntimeDescriptorArray arrives with the texture heap).
    */
   const struct spirv_capabilities caps = {
      .Shader = true,
      .Int8 = true,
      .Int16 = true,
      .Int64 = true,
      .Float64 = true,
      .Int64Atomics = true,
      .PhysicalStorageBufferAddresses = true,
      .VariablePointers = true,
      .VariablePointersStorageBuffer = true,
      /* The texture heap: runtime arrays indexed by anything */
      .RuntimeDescriptorArray = true,
      .SampledImageArrayDynamicIndexing = true,
      .StorageImageArrayDynamicIndexing = true,
      .ShaderNonUniform = true,
      .SampledImageArrayNonUniformIndexing = true,
      .StorageImageArrayNonUniformIndexing = true,
      .ImageQuery = true,
      .StorageImageExtendedFormats = true,
      .StorageImageReadWithoutFormat = true,
      .StorageImageWriteWithoutFormat = true,
      /* Raster stages */
      .DerivativeControl = true,
      .ClipDistance = true,
      .CullDistance = true,
      .DrawParameters = true,
   };

   const struct spirv_to_nir_options spirv_options = {
      .capabilities = &caps,
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .ssbo_addr_format = nir_address_format_64bit_global_32bit_offset,
      .ubo_addr_format = nir_address_format_64bit_global_32bit_offset,
      .shared_addr_format = nir_address_format_32bit_offset,
      .debug.func = kgp_spirv_debug,
      /* Bad SPIR-V is the caller's error, reported as one: without this an
       * assert-enabled Mesa breaks into the debugger on a parse failure.
       */
      .skip_os_break_in_debug_build = true,
   };

   if (spec != NULL)
      kgp_spec_mark_defined(words, spirv.size / 4, spec);

   nir_shader *nir = spirv_to_nir(words, spirv.size / 4, spec,
                                  stage, "main",
                                  &spirv_options, nak_nir_options(dev->nak));
   if (nir == NULL) {
      kgp_err("SPIR-V has no %s entry point \"main\" or failed to parse",
              _mesa_shader_stage_to_string(stage));
      return NULL;
   }
   nir_validate_shader(nir, "after spirv_to_nir");

   /* We have to lower away local constant initializers right before we
    * inline functions.  That way they get properly initialized at the top
    * of the function and not at the top of its caller.
    */
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_deref);

   /* Pick off the single entrypoint that we want */
   nir_remove_non_cmat_call_entrypoints(nir);

   NIR_PASS(_, nir, nir_lower_variable_initializers, ~0);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_split_per_member_structs);
   NIR_PASS(_, nir, nir_remove_dead_variables,
            nir_var_shader_in | nir_var_shader_out | nir_var_system_value,
            NULL);

   return nir;
}

/* Ancestor: nvk_shader.c:shared_var_info */
static void
shared_var_info(const struct glsl_type *type, unsigned *size, unsigned *align)
{
   assert(glsl_type_is_vector_or_scalar(type));

   uint32_t comp_size = glsl_type_is_boolean(type) ? 4 : glsl_get_bit_size(type) / 8;
   unsigned length = glsl_get_vector_elements(type);
   *size = comp_size * length, *align = comp_size;
}

static bool
kgp_lower_nir(struct KgpDevice_T *dev, nir_shader *nir)
{
   /* KoreGP has no descriptor sets. Buffers are reached through the root
    * pointer; textures by index into the heap (kgp_nir_lower_heap below).
    */
   nir_foreach_variable_with_modes(var, nir, nir_var_mem_ubo |
                                             nir_var_mem_ssbo) {
      kgp_err("shader declares descriptor-set buffer \"%s\"; KoreGP "
              "shaders reach memory through the root pointer",
              var->name ? var->name : "?");
      return false;
   }

   nir_lower_compute_system_values_options csv_options = {
      .lower_local_invocation_index = true,
   };
   NIR_PASS(_, nir, nir_lower_compute_system_values, &csv_options);

   /* Lower push constants before the root lowering */
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const,
            nir_address_format_32bit_offset);

   /* Large constant support assumes cbufs */
   /* Needs to run before load_const_to_scalar */
   NIR_PASS(_, nir, nir_opt_large_constants, NULL, 32);

   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);

   nir_opt_access_options opt_access_options = {
      .is_vulkan = true,
   };
   NIR_PASS(_, nir, nir_opt_access, &opt_access_options);

   const bool compute = nir->info.stage == MESA_SHADER_COMPUTE;

   /* The root lowering checks where in the push-constant block a load
    * lands; fold the offset arithmetic so that it can.
    */
   NIR_PASS(_, nir, nir_opt_constant_folding);

   bool bad_push = false;
   NIR_PASS(_, nir, kgp_nir_lower_root, &bad_push);
   if (bad_push) {
      kgp_err("the push-constant block must be exactly one "
              "PhysicalStorageBuffer pointer (the root pointer)%s",
              nir->info.stage == MESA_SHADER_FRAGMENT ? ", at offset 8 in a "
              "pixel shader" : "");
      return false;
   }

   bool bad_heap = false;
   NIR_PASS(_, nir, kgp_nir_lower_heap, dev->debug & KGP_DEBUG_VALIDATE,
            &bad_heap);
   if (bad_heap) {
      kgp_err("textures and images must be in the texture heap (set 0, "
              "binding 0) and samplers in the sampler heap (set 0, binding 1)");
      return false;
   }

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global,
            nir_address_format_64bit_global);

   if (compute) {
      NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared,
               shared_var_info);
      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared,
               nir_address_format_32bit_offset);
   }

   nak_lower_nir_before_linking(nir, dev->nak);
   nir->info.io_lowered = true;

   /* Ancestor: nvk_lower_nir_late, compute branch */
   if (compute &&
       nir->info.zero_initialize_shared_memory && nir->info.shared_size > 0) {
      /* QMD::SHARED_MEMORY_SIZE requires an alignment of 256B so it's
       * safe to align everything up to 16B so we can write whole vec4s.
       */
      nir->info.shared_size = align(nir->info.shared_size, 16);
      NIR_PASS(_, nir, nir_zero_initialize_shared_memory,
               nir->info.shared_size, 16);

      /* nir_zero_initialize_shared_memory generates load_invocation_id
       * which has to be lowered to load_invocation_index.
       */
      NIR_PASS(_, nir, nir_lower_compute_system_values, &csv_options);
   }

   return true;
}

/* Ancestor: nvk_device.c:nvk_slm_area_ensure. Old areas are kept until
 * the device is destroyed: a command buffer that programmed one may still
 * be in flight, and they only ever grow, so there are few.
 */
bool
kgp_device_ensure_slm(struct KgpDevice_T *dev,
                      uint32_t slm_bytes_per_lane,
                      uint32_t crs_bytes_per_warp)
{
   const struct nv_device_info *info = kgp_dev_info(dev);

   assert(slm_bytes_per_lane < (1 << 24));
   assert(crs_bytes_per_warp <= (1 << 20));
   uint64_t bytes_per_warp = slm_bytes_per_lane * 32 + crs_bytes_per_warp;

   /* The hardware seems to require this alignment for
    * NV9097_SET_SHADER_LOCAL_MEMORY_E_DEFAULT_SIZE_PER_WARP
    */
   bytes_per_warp = align64(bytes_per_warp, 0x200);

   uint64_t bytes_per_mp = bytes_per_warp * info->max_warps_per_mp;
   uint64_t bytes_per_tpc = bytes_per_mp * info->mp_per_tpc;

   /* The hardware seems to require this alignment for
    * NVA0C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A_SIZE_LOWER.
    */
   bytes_per_tpc = align64(bytes_per_tpc, 0x8000);

   bool ok = true;
   simple_mtx_lock(&dev->mutex);
   if (bytes_per_tpc > dev->slm_bytes_per_tpc) {
      /* The hardware seems to require this alignment for
       * NV9097_SET_SHADER_LOCAL_MEMORY_D_SIZE_LOWER.
       */
      const uint64_t size = align64(bytes_per_tpc * info->tpc_count, 0x20000);

      struct nvkmd_mem *mem;
      if (nvkmd_dev_alloc_mem(dev->nvkmd, KGP_LOG_OBJ, size, 0,
                              NVKMD_MEM_LOCAL, &mem) == VK_SUCCESS) {
         if (dev->slm_mem != NULL)
            util_dynarray_append(&dev->old_slm_mems, dev->slm_mem);
         dev->slm_mem = mem;
         dev->slm_bytes_per_tpc = bytes_per_tpc;
         dev->slm_bytes_per_warp = bytes_per_warp;
      } else {
         ok = false;
      }
   }
   simple_mtx_unlock(&dev->mutex);

   return ok;
}

/* The QMD the way NVK builds it on every dispatch (nvk_cmd_upload_qmd):
 * NAK packs each field through its generic bit-field writer. Authoritative,
 * and the larger part of what recording a dispatch costs in NVK.
 */
static void
kgp_pipeline_fill_qmd_generic(const struct nv_device_info *info,
                              const struct KgpPipeline_T *pipeline,
                              uint32_t *qmd, uint64_t root_record_addr,
                              uint32_t x, uint32_t y, uint32_t z)
{
   struct nak_qmd_info qmd_info = {
      .addr = pipeline->code_addr,
      .smem_size = pipeline->info.cs.smem_size,
      .global_size = { x, y, z },
      .num_cbufs = 1,
      .cbufs = {
         {
            .index = KGP_CBUF_ROOT,
            .addr = root_record_addr,
            .size = sizeof(struct kgp_root_record),
         },
      },
   };
   if (pipeline->data_size_B > 0) {
      qmd_info.cbufs[qmd_info.num_cbufs++] = (struct nak_qmd_cbuf) {
         .index = KGP_CBUF_SHADER_DATA,
         .addr = pipeline->data_addr,
         .size = pipeline->data_size_B,
      };
   }

   nak_fill_qmd(info, &pipeline->info, &qmd_info, qmd, pipeline->qmd_size_B);
}

/* Returns false if the address field cannot be patched as whole dwords. */
static bool
kgp_qmd_addr_patch_init(const struct nv_device_info *info, uint8_t cbuf_idx,
                        struct kgp_qmd_addr_patch *patch)
{
   const struct nak_qmd_cbuf_desc_layout cbuf =
      nak_get_qmd_cbuf_desc_layout(info, cbuf_idx);

   *patch = (struct kgp_qmd_addr_patch) {
      .lo_dw = cbuf.addr_lo_start / 32,
      .hi_dw = cbuf.addr_hi_start / 32,
      .hi_shift = cbuf.addr_hi_start % 32,
      .addr_shift = cbuf.addr_shift,
   };
   return cbuf.addr_lo_start % 32 == 0 &&
          cbuf.addr_lo_end == cbuf.addr_lo_start + 32;
}

static inline void
kgp_qmd_addr_patch_apply(const struct kgp_qmd_addr_patch *patch,
                         uint32_t *qmd, uint64_t addr)
{
   const uint64_t addr_shifted = addr >> patch->addr_shift;
   qmd[patch->lo_dw] = (uint32_t)addr_shifted;
   qmd[patch->hi_dw] |= (uint32_t)(addr_shifted >> 32) << patch->hi_shift;
}

static void
kgp_pipeline_init_qmd(const struct nv_device_info *info,
                      struct KgpPipeline_T *pipeline)
{
   pipeline->qmd_size_B = nak_qmd_size_B(info);
   assert(pipeline->qmd_size_B <= sizeof(pipeline->qmd));
   kgp_pipeline_fill_qmd_generic(info, pipeline, pipeline->qmd, 0, 0, 0, 0);

   const struct nak_qmd_dispatch_size_layout size =
      nak_get_qmd_dispatch_size_layout(info);

   /* The group counts must be three consecutive dwords (the condition
    * under which NVK patches them with one copy for indirect dispatch,
    * nvk_CmdDispatchIndirect2KHR) and each low address field a whole dword.
    * True for every QMD version from Volta on; otherwise fall back.
    */
   pipeline->qmd_patchable =
      size.x_start % 32 == 0 && size.x_end == size.x_start + 32 &&
      size.y_start == size.x_start + 32 &&
      size.z_start == size.y_start + 32;
   pipeline->qmd_patchable &=
      kgp_qmd_addr_patch_init(info, KGP_CBUF_ROOT, &pipeline->qmd_root_record);
   pipeline->qmd_size_dw = size.x_start / 32;
   pipeline->max_groups_y = BITFIELD64_MASK(size.y_end - size.y_start);
   pipeline->max_groups_z = BITFIELD64_MASK(size.z_end - size.z_start);
}

/* Writes qmd_size_B bytes of QMD for one dispatch of this pipeline. */
void
kgp_pipeline_fill_qmd(const struct nv_device_info *info,
                      const struct KgpPipeline_T *pipeline,
                      uint32_t *qmd, uint64_t root_record_addr,
                      uint32_t x, uint32_t y, uint32_t z)
{
   if (unlikely(!pipeline->qmd_patchable)) {
      kgp_pipeline_fill_qmd_generic(info, pipeline, qmd, root_record_addr,
                                    x, y, z);
      return;
   }

   memcpy(qmd, pipeline->qmd, pipeline->qmd_size_B);

   qmd[pipeline->qmd_size_dw + 0] = x;
   qmd[pipeline->qmd_size_dw + 1] = y;
   qmd[pipeline->qmd_size_dw + 2] = z;

   kgp_qmd_addr_patch_apply(&pipeline->qmd_root_record, qmd, root_record_addr);

#ifndef NDEBUG
   /* Assert-enabled builds check every patched QMD against NAK's own. */
   uint32_t check[NAK_MAX_QMD_DWORDS];
   kgp_pipeline_fill_qmd_generic(info, pipeline, check, root_record_addr,
                                 x, y, z);
   assert(memcmp(qmd, check, pipeline->qmd_size_B) == 0);
#endif
}

KGP_API KgpPipeline
kgpCreateComputePipeline(KgpDevice dev, KgpByteSpan spirv,
                         const KgpSpecialization *constants)
{
   struct nir_spirv_specialization *spec;
   if (!kgp_spec_to_nir(constants, &spec)) {
      kgp_err("kgpCreateComputePipeline: bad constants");
      return NULL;
   }
   nir_shader *nir = kgp_spirv_to_nir(dev, spirv, MESA_SHADER_COMPUTE, spec);
   const bool spec_ok = kgp_spec_all_defined(spec);
   vtn_free_specialization(spec);
   if (nir == NULL)
      return NULL;
   if (!spec_ok) {
      ralloc_free(nir);
      return NULL;
   }

   struct KgpPipeline_T *pipeline = NULL;
   struct nak_shader_bin *bin = NULL;

   nak_preprocess_nir(nir, dev->nak);
   if (!kgp_lower_nir(dev, nir))
      goto fail;

   if (dev->debug & KGP_DEBUG_NIR)
      nir_print_shader(nir, stderr);

   const bool dump_asm = dev->debug & KGP_DEBUG_ASM;
   bin = nak_compile_shader(nir, dump_asm, dev->nak, 0 /* robust2_modes */,
                            NULL /* fs_key */, false /* has_task_shader */);
   if (bin == NULL) {
      kgp_err("internal compiler error in NAK");
      goto fail;
   }
   if (dump_asm && bin->asm_str != NULL) {
      fputs(bin->asm_str, stderr);
      fprintf(stderr, "koregp: %u instructions, %u GPRs, %u bytes of local "
              "memory per lane\n",
              bin->info.num_instrs, bin->info.num_gprs, bin->info.slm_size);
   }

   pipeline = calloc(1, sizeof(*pipeline));
   if (pipeline == NULL)
      goto fail;
   pipeline->info = bin->info;

   /* Layout: code | pad to cbuf alignment | constant data | overalloc.
    * Compute shaders have no header, and the allocation is page-aligned,
    * which covers the 0x80 instruction alignment NVK asks of its heap.
    */
   const uint32_t data_offset = align(bin->code_size, KGP_CBUF_ALIGN_B);
   const uint32_t data_size = align(nir->constant_data_size, KGP_CBUF_ALIGN_B);
   if (data_size > KGP_MAX_CBUF_SIZE_B) {
      kgp_err("shader constant data (%u bytes) exceeds one constant buffer",
              nir->constant_data_size);
      goto fail;
   }
   const uint32_t total_size = data_offset + data_size + KGP_SHADER_OVERALLOC_B;

   /* ponytail: one page-granular allocation per pipeline. Add a shader heap
    * (ancestor: nvk_heap) when pipeline counts make the rounding matter.
    */
   pipeline->alloc = kgp_alloc_create_mapped(dev, total_size, NVKMD_MEM_LOCAL,
                                             NVKMD_MEM_MAP_WR);
   if (pipeline->alloc == NULL)
      goto fail;
   struct nvkmd_mem *mem = pipeline->alloc->mem;

   /* One pass over the write-combined map, front to back. */
   char *map = mem->map;
   memcpy(map, bin->code, bin->code_size);
   memset(map + bin->code_size, 0, data_offset - bin->code_size);
   if (nir->constant_data_size > 0)
      memcpy(map + data_offset, nir->constant_data, nir->constant_data_size);
   memset(map + data_offset + nir->constant_data_size, 0,
          total_size - data_offset - nir->constant_data_size);
   nvkmd_mem_unmap(mem, 0);

   pipeline->code_addr = mem->va->addr;
   pipeline->data_addr = mem->va->addr + data_offset;
   pipeline->data_size_B = data_size;
   kgp_pipeline_init_qmd(kgp_dev_info(dev), pipeline);

   if (!kgp_device_ensure_slm(dev, pipeline->info.slm_size,
                              pipeline->info.crs_size))
      goto fail;

   nak_shader_bin_destroy(bin);
   ralloc_free(nir);
   return pipeline;

fail:
   if (pipeline != NULL && pipeline->alloc != NULL)
      kgp_alloc_free_deferred(dev, pipeline->alloc);
   free(pipeline);
   if (bin != NULL)
      nak_shader_bin_destroy(bin);
   ralloc_free(nir);
   return NULL;
}

/* Graphics pipelines ------------------------------------------------------- */

/* Ancestor: nvk_shader.c:nir_opts, the cleanup nir_opt_varyings_bulk runs
 * between its steps.
 */
static void
kgp_nir_opts(nir_shader *nir, void *data)
{
   bool progress;

   do {
      progress = false;

      NIR_PASS(progress, nir, nir_opt_loop);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);

      NIR_PASS(progress, nir, nir_opt_if, 0);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_cse);

      NIR_PASS(progress, nir, nir_opt_peephole_select,
               &(nir_opt_peephole_select_options){
                  .limit = 0,
                  .discard_ok = true,
               });

      NIR_PASS(progress, nir, nir_opt_phi_precision);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);

      NIR_PASS(progress, nir, nir_opt_undef);
      NIR_PASS(progress, nir, nir_opt_loop_unroll);
   } while (progress);
}

/* SPIR-V to lowered NIR for one raster stage. Ancestors: nvk_preprocess_nir
 * and nvk_lower_nir.
 */
static nir_shader *
kgp_gfx_stage_nir(struct KgpDevice_T *dev, KgpByteSpan spirv,
                  mesa_shader_stage stage,
                  struct nir_spirv_specialization *spec)
{
   nir_shader *nir = kgp_spirv_to_nir(dev, spirv, stage, spec);
   if (nir == NULL)
      return NULL;

   /* Lower primitive id to varyings to prepare for nir_opt_varyings_bulk */
   struct nir_lower_sysvals_to_varyings_options sysvals_opts = {
      .primitive_id = stage == MESA_SHADER_FRAGMENT,
   };
   nir_lower_sysvals_to_varyings(nir, &sysvals_opts);

   nak_preprocess_nir(nir, dev->nak);

   if (!kgp_lower_nir(dev, nir)) {
      ralloc_free(nir);
      return NULL;
   }
   return nir;
}

/* Bytes one shader takes in the pipeline's memory: the program header, the
 * code, and its constant data on a constant-buffer boundary. Ancestor:
 * nvk_shader_upload on Turing (header at the start, no offset games).
 */
static uint32_t
kgp_gfx_shader_layout(const struct nak_shader_bin *bin, const nir_shader *nir,
                      uint32_t *data_offset_out, uint32_t *data_size_out)
{
   const uint32_t code_end = sizeof(bin->info.hdr) + bin->code_size;
   *data_offset_out = align(code_end, KGP_CBUF_ALIGN_B);
   *data_size_out = align(nir->constant_data_size, KGP_CBUF_ALIGN_B);
   /* The next shader's first instruction must be 0x80-aligned; its header
    * is 0x80 long, so aligning the header does it.
    */
   return align(*data_offset_out + *data_size_out, 0x80);
}

KGP_API KgpPipeline
kgpCreateGraphicsPipeline(KgpDevice dev, KgpByteSpan vs, KgpByteSpan ps,
                          const KgpRasterDesc *raster,
                          const KgpSpecialization *constants)
{
   if (raster == NULL || raster->colorTargetCount == 0 ||
       raster->colorTargetCount > KGP_MAX_COLOR_TARGETS ||
       (raster->depthFormat != KGP_FORMAT_NONE &&
        raster->depthFormat != KGP_FORMAT_D32_FLOAT &&
        raster->depthFormat != KGP_FORMAT_D24_UNORM_S8_UINT) ||
       raster->topology > KGP_TOPOLOGY_TRIANGLE_STRIP ||
       raster->cull > KGP_CULL_CCW) {
      kgp_err("kgpCreateGraphicsPipeline: the raster desc needs 1 to %u "
              "colour targets and a known depth format, topology and cull "
              "mode", KGP_MAX_COLOR_TARGETS);
      return NULL;
   }
   for (uint32_t i = 0; i < raster->colorTargetCount; i++) {
      if (raster->colorWrite[i] > KGP_COLOR_WRITE_NONE) {
         kgp_err("kgpCreateGraphicsPipeline: colorWrite[%u] is not a "
                 "KGP_COLOR_WRITE value", i);
         return NULL;
      }
   }
   struct nir_spirv_specialization *spec;
   if (!kgp_spec_to_nir(constants, &spec)) {
      kgp_err("kgpCreateGraphicsPipeline: bad constants");
      return NULL;
   }

   /* Index = mesa stage order, as NVK hands them to the linker. */
   nir_shader *nirs[2] = { NULL, NULL };
   struct nak_shader_bin *bins[2] = { NULL, NULL };
   struct KgpPipeline_T *pipeline = NULL;

   nirs[0] = kgp_gfx_stage_nir(dev, vs, MESA_SHADER_VERTEX, spec);
   nirs[1] = kgp_gfx_stage_nir(dev, ps, MESA_SHADER_FRAGMENT, spec);
   /* An id may belong to either stage: checked once both are parsed. */
   const bool spec_ok = nirs[0] == NULL || nirs[1] == NULL ||
                        kgp_spec_all_defined(spec);
   vtn_free_specialization(spec);
   if (nirs[0] == NULL || nirs[1] == NULL || !spec_ok)
      goto fail;

   /* The only cross-stage linking there is (ancestor: nvk_compile_shaders).
    * What survives it decides the interface the two program headers
    * describe to the hardware.
    */
   nir_opt_varyings_bulk(nirs, 2, true, UINT32_MAX, UINT32_MAX,
                         kgp_nir_opts, NULL);

   /* Ancestor: nvk_populate_fs_key with pipeline state present: nothing set. */
   const struct nak_fs_key fs_key = { 0 };
   const bool dump_asm = dev->debug & KGP_DEBUG_ASM;
   for (unsigned i = 0; i < 2; i++) {
      if (dev->debug & KGP_DEBUG_NIR)
         nir_print_shader(nirs[i], stderr);

      bins[i] = nak_compile_shader(nirs[i], dump_asm, dev->nak,
                                   0 /* robust2_modes */,
                                   i == 1 ? &fs_key : NULL,
                                   false /* has_task_shader */);
      if (bins[i] == NULL) {
         kgp_err("internal compiler error in NAK");
         goto fail;
      }
      if (dump_asm && bins[i]->asm_str != NULL) {
         fputs(bins[i]->asm_str, stderr);
         fprintf(stderr, "koregp: %s shader: %u instructions, %u GPRs\n",
                 i == 0 ? "vertex" : "pixel", bins[i]->info.num_instrs,
                 bins[i]->info.num_gprs);
      }
   }

   pipeline = calloc(1, sizeof(*pipeline));
   if (pipeline == NULL)
      goto fail;
   pipeline->graphics = true;
   pipeline->info = bins[0]->info;
   pipeline->ps_info = bins[1]->info;
   pipeline->raster = *raster;

   /* Layout: [header | code | constant data] twice, then the overalloc. */
   uint32_t offset[2], data_offset[2], data_size[2], total_size = 0;
   for (unsigned i = 0; i < 2; i++) {
      offset[i] = total_size;
      total_size += kgp_gfx_shader_layout(bins[i], nirs[i],
                                          &data_offset[i], &data_size[i]);
      if (data_size[i] > KGP_MAX_CBUF_SIZE_B) {
         kgp_err("shader constant data (%u bytes) exceeds one constant buffer",
                 nirs[i]->constant_data_size);
         goto fail;
      }
   }
   total_size += KGP_SHADER_OVERALLOC_B;

   pipeline->alloc = kgp_alloc_create_mapped(dev, total_size, NVKMD_MEM_LOCAL,
                                             NVKMD_MEM_MAP_WR);
   if (pipeline->alloc == NULL)
      goto fail;
   struct nvkmd_mem *mem = pipeline->alloc->mem;

   /* Built in ordinary memory and copied once: the map is write-combined. */
   char *image = calloc(1, total_size);
   if (image == NULL)
      goto fail;
   struct kgp_gfx_shader shaders[2];
   for (unsigned i = 0; i < 2; i++) {
      const struct nak_shader_bin *bin = bins[i];
      char *base = image + offset[i];

      memcpy(base, bin->info.hdr, sizeof(bin->info.hdr));
      memcpy(base + sizeof(bin->info.hdr), bin->code, bin->code_size);
      if (nirs[i]->constant_data_size > 0) {
         memcpy(base + data_offset[i], nirs[i]->constant_data,
                nirs[i]->constant_data_size);
      }

      shaders[i] = (struct kgp_gfx_shader) {
         .info = i == 0 ? &pipeline->info : &pipeline->ps_info,
         .hdr_addr = mem->va->addr + offset[i],
         .size_B = sizeof(bin->info.hdr) + bin->code_size,
         .data_addr = mem->va->addr + offset[i] + data_offset[i],
         .data_size_B = data_size[i],
      };
   }
   memcpy(mem->map, image, total_size);
   free(image);
   nvkmd_mem_unmap(mem, 0);

   if (!kgp_gfx_pipeline_build_push(kgp_dev_info(dev), pipeline,
                                    &shaders[0], &shaders[1]))
      goto fail;

   if (!kgp_device_ensure_slm(dev,
                              MAX2(bins[0]->info.slm_size,
                                   bins[1]->info.slm_size),
                              MAX2(bins[0]->info.crs_size,
                                   bins[1]->info.crs_size)))
      goto fail;

   for (unsigned i = 0; i < 2; i++) {
      nak_shader_bin_destroy(bins[i]);
      ralloc_free(nirs[i]);
   }
   return pipeline;

fail:
   if (pipeline != NULL) {
      if (pipeline->alloc != NULL)
         kgp_alloc_free_deferred(dev, pipeline->alloc);
      free(pipeline->gfx_push);
      free(pipeline);
   }
   for (unsigned i = 0; i < 2; i++) {
      if (bins[i] != NULL)
         nak_shader_bin_destroy(bins[i]);
      ralloc_free(nirs[i]);
   }
   return NULL;
}

KGP_API void
kgpDestroyPipeline(KgpDevice dev, KgpPipeline pipeline)
{
   if (pipeline == NULL)
      return;

   /* Dispatches already submitted may still be running this code. */
   kgp_alloc_free_deferred(dev, pipeline->alloc);
   free(pipeline->gfx_push);
   free(pipeline);
}
