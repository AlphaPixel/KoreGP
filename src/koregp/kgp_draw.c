/*
 * Raster (plan N3). So far: the 3D engine's channel state, and the method
 * block that binds a graphics pipeline's shaders.
 *
 * Ancestors: src/nouveau/vulkan/nvk_cmd_draw.c and nvk_shader.c
 *   nvk_push_draw_state_init           kgp_push_draw_state_init
 *   nvk_shader_fill_push               kgp_shader_fill_push, _vtgm
 *   nvk_cmd_flush_gfx_shaders, _cbufs  kgp_gfx_pipeline_build_push
 * Cut: every branch for classes older than Turing (KoreGP refuses those
 * devices), SET_PROGRAM_REGION, the VAB area. Kept in NVK's order and with
 * NVK's values, so that the init push can be compared with NVK's
 * (tests/init3d_matches_nvk.sh).
 *
 * The macros (MME) are not extracted: nvk_build_mme() is called for all of
 * them, in NVK's slots. A draw, a clear and an index-buffer bind are macro
 * calls on this hardware, and a macro's slot number and the scratch
 * registers it reads are an ABI between it and this file.
 *
 * SPDX-License-Identifier: MIT
 */
#include "kgp_private.h"
#include "kgp_mme.h"

#include "mme_builder.h"

#include "nv_push_cl9097.h"
#include "nv_push_cla097.h"
#include "nv_push_clb197.h"
#include "nv_push_clc397.h"
#include "nv_push_clc597.h"
#include "nv_push_clc797.h"
#include "nv_push_clcb97.h"
#include "clc997.h"
#include "clcd97.h"
#include "util/format/u_format.h"
#include "util/u_math.h"

#include <stdlib.h>

/* The root descriptor table NVK keeps in constant buffer 0. On Turing its
 * contents live in the hardware's root tables instead and nothing reads cb0,
 * but NVK still sets it up, so this does too.
 *
 * NVK's size is sizeof(struct nvk_root_descriptor_table), whose header
 * cannot be included from here (it drags in NVK's generated headers). The
 * struct fills the eight root tables exactly; init3d_matches_nvk fails if
 * this ever stops being its size.
 */
#define KGP_DRAW_CB0_SIZE (NVK_HW_ROOT_TABLE_SIZE * NVK_HW_ROOT_TABLE_COUNT)
static_assert(KGP_DRAW_CB0_SIZE <= KGP_DRAW_CB0_MEM_SIZE_B,
              "cb0 fits its allocation");

static uint32_t
get_sm_disp_ctrl_reg(const struct nv_device_info *devinfo)
{
   if (devinfo->cls_eng3d >= HOPPER_A)
      return 0x4243a4;
   return 0x419ba4;
}

static uint32_t
get_sms_hww_warp_esp_report_mask_reg(const struct nv_device_info *devinfo)
{
   if (devinfo->cls_eng3d >= HOPPER_A)
      return 0x4246a8;
   return 0x419ea8;
}

/* Needs KGP_INIT_PUSH_DW of pushbuf. Returns false if a macro could not be
 * built (out of memory).
 */
/* KoreGP's own macros, into the slots and the instruction memory after
 * NVK's. Pushed after the whole of NVK's init rather than with NVK's macros,
 * so that everything before it stays NVK's stream word for word
 * (init3d_matches_nvk).
 */
void
kgp_push_own_macros(const struct nv_device_info *info, struct nv_push *p,
                    uint32_t mme_pos)
{
   struct mme_builder b;
   mme_builder_init(&b, info);
   kgp_mme_draw_indexed_multi(&b);
   size_t size;
   uint32_t *dw = mme_builder_finish(&b, &size);
   const uint32_t num_dw = size / sizeof(uint32_t);

   P_MTHD(p, NV9097, LOAD_MME_START_ADDRESS_RAM_POINTER);
   P_NV9097_LOAD_MME_START_ADDRESS_RAM_POINTER(p, KGP_MME_DRAW_INDEXED_MULTI);
   P_NV9097_LOAD_MME_START_ADDRESS_RAM(p, mme_pos);

   P_1INC(p, NV9097, LOAD_MME_INSTRUCTION_RAM_POINTER);
   P_NV9097_LOAD_MME_INSTRUCTION_RAM_POINTER(p, mme_pos);
   P_INLINE_ARRAY(p, dw, num_dw);

   free(dw);
}

bool
kgp_push_draw_state_init(const struct nv_device_info *info, struct nv_push *p,
                         uint64_t cb0_addr, uint64_t zero_addr,
                         uint32_t *mme_end_out)
{
   /* 3D state */
   P_MTHD(p, NV9097, SET_OBJECT);
   P_NV9097_SET_OBJECT(p, {
      .class_id = info->cls_eng3d,
      .engine_id = 0,
   });

   uint32_t mme_pos = 0;
   for (uint32_t mme = 0; mme < NVK_MME_COUNT; mme++) {
      size_t size;
      uint32_t *dw = nvk_build_mme(info, mme, &size);
      if (dw == NULL)
         return false;

      assert(size % sizeof(uint32_t) == 0);
      const uint32_t num_dw = size / sizeof(uint32_t);

      P_MTHD(p, NV9097, LOAD_MME_START_ADDRESS_RAM_POINTER);
      P_NV9097_LOAD_MME_START_ADDRESS_RAM_POINTER(p, mme);
      P_NV9097_LOAD_MME_START_ADDRESS_RAM(p, mme_pos);

      P_1INC(p, NV9097, LOAD_MME_INSTRUCTION_RAM_POINTER);
      P_NV9097_LOAD_MME_INSTRUCTION_RAM_POINTER(p, mme_pos);
      P_INLINE_ARRAY(p, dw, num_dw);

      mme_pos += num_dw;

      free(dw);
   }
   *mme_end_out = mme_pos;

   if (info->cls_eng3d < BLACKWELL_A)
      P_IMMD(p, NVC597, SET_MME_DATA_FIFO_CONFIG, FIFO_SIZE_SIZE_4KB);

   /* Enable FP helper invocation memory loads: clears bit 3 of
    * gr_gpcs_tpcs_sm_disp_ctrl. See the ancestor for why.
    */
   P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_SET_PRIV_REG));
   P_INLINE_DATA(p, 0);
   P_INLINE_DATA(p, BITFIELD_BIT(3));
   P_INLINE_DATA(p, get_sm_disp_ctrl_reg(info));

   /* Disable Out Of Range Address exceptions: clears bit 14 of
    * gr_gpcs_tpcs_sms_hww_warp_esr_report_mask. See the ancestor for why.
    */
   P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_SET_PRIV_REG));
   P_INLINE_DATA(p, 0);
   P_INLINE_DATA(p, BITFIELD_BIT(14));
   P_INLINE_DATA(p, get_sms_hww_warp_esp_report_mask_reg(info));

   /* Set CONSERVATIVE_RASTER_STATE to an invalid value, to ensure the
    * hardware reg is always set the first time conservative rasterization
    * is enabled */
   P_IMMD(p, NV9097, SET_MME_SHADOW_SCRATCH(NVK_MME_SCRATCH_CONSERVATIVE_RASTER_STATE),
                     ~0);

   /* Initialize tessellation parameters */
   P_IMMD(p, NV9097, SET_MME_SHADOW_SCRATCH(NVK_MME_SCRATCH_TESS_PARAMS), 0);
   P_IMMD(p, NV9097, SET_TESSELLATION_PARAMETERS, {
      .domain_type = DOMAIN_TYPE_ISOLINE,
      .spacing = SPACING_INTEGER,
      .output_primitives = OUTPUT_PRIMITIVES_LINES,
   });

   P_IMMD(p, NV9097, SET_RENDER_ENABLE_C, MODE_TRUE);

   P_IMMD(p, NV9097, SET_Z_COMPRESSION, ENABLE_TRUE);
   P_IMMD(p, NVB197, SET_STENCIL_COMPRESSION, ENABLE_TRUE);

   P_MTHD(p, NV9097, SET_COLOR_COMPRESSION(0));
   for (unsigned i = 0; i < 8; i++)
      P_NV9097_SET_COLOR_COMPRESSION(p, i, ENABLE_TRUE);

   P_IMMD(p, NV9097, SET_CT_SELECT, { .target_count = 1 });

   P_IMMD(p, NV9097, SET_ALIASED_LINE_WIDTH_ENABLE, V_TRUE);

   P_IMMD(p, NV9097, SET_DA_PRIMITIVE_RESTART_VERTEX_ARRAY, ENABLE_FALSE);

   P_IMMD(p, NV9097, SET_BLEND_SEPARATE_FOR_ALPHA, ENABLE_TRUE);
   P_IMMD(p, NV9097, SET_SINGLE_CT_WRITE_CONTROL, ENABLE_TRUE);
   P_IMMD(p, NV9097, SET_SINGLE_ROP_CONTROL, ENABLE_FALSE);
   P_IMMD(p, NV9097, SET_TWO_SIDED_STENCIL_TEST, ENABLE_TRUE);

   P_IMMD(p, NV9097, SET_ALPHA_TO_COVERAGE_OVERRIDE, {
      .qualify_by_anti_alias_enable = true,
      .qualify_by_ps_sample_mask_output = false,
   });

   P_IMMD(p, NV9097, SET_SHADE_MODE, V_OGL_SMOOTH);

   P_IMMD(p, NV9097, SET_API_VISIBLE_CALL_LIMIT, V__128);

   P_IMMD(p, NV9097, SET_ZCULL_STATS, ENABLE_TRUE);

   P_IMMD(p, NV9097, SET_REDUCE_COLOR_THRESHOLDS_ENABLE, V_FALSE);
   P_IMMD(p, NV9097, SET_REDUCE_COLOR_THRESHOLDS_UNORM8, {
      .all_covered_all_hit_once = 0x4,
   });
   P_MTHD(p, NV9097, SET_REDUCE_COLOR_THRESHOLDS_UNORM10);
   P_NV9097_SET_REDUCE_COLOR_THRESHOLDS_UNORM10(p, {
      .all_covered_all_hit_once = 0xff,
   });
   P_NV9097_SET_REDUCE_COLOR_THRESHOLDS_UNORM16(p, {
      .all_covered_all_hit_once = 0xff,
   });
   P_NV9097_SET_REDUCE_COLOR_THRESHOLDS_FP11(p, {
      .all_covered_all_hit_once = 0x3f,
   });
   P_NV9097_SET_REDUCE_COLOR_THRESHOLDS_FP16(p, {
      .all_covered_all_hit_once = 0xff,
   });
   P_NV9097_SET_REDUCE_COLOR_THRESHOLDS_SRGB8(p, {
      .all_covered_all_hit_once = 0x4,
   });
   P_IMMD(p, NV9097, SET_SHADER_CACHE_CONTROL, info->cls_eng3d >= ADA_A);

   P_IMMD(p, NV9097, CHECK_SPH_VERSION, {
      .current = 3,
      .oldest_supported = 3,
   });
   P_IMMD(p, NV9097, CHECK_AAM_VERSION, {
      .current = 2,
      .oldest_supported = 2,
   });

   P_IMMD(p, NV9097, SET_L2_CACHE_CONTROL_FOR_VAF_REQUESTS, {
      .system_memory_volatile = false,
      .policy                 = POLICY_EVICT_NORMAL,
   });
   P_IMMD(p, NV9097, SET_L2_CACHE_CONTROL_FOR_ROP_PREFETCH_READ_REQUESTS,
                     POLICY_EVICT_NORMAL);
   P_IMMD(p, NV9097, SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_READ_REQUESTS,
                     POLICY_EVICT_NORMAL);
   P_IMMD(p, NV9097, SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_READ_REQUESTS,
                     POLICY_EVICT_NORMAL);
   P_IMMD(p, NV9097, SET_L2_CACHE_CONTROL_FOR_ROP_NONINTERLOCKED_WRITE_REQUESTS,
                     POLICY_EVICT_NORMAL);
   P_IMMD(p, NV9097, SET_L2_CACHE_CONTROL_FOR_ROP_INTERLOCKED_WRITE_REQUESTS,
                     POLICY_EVICT_NORMAL);

   P_IMMD(p, NV9097, SET_BLEND_PER_FORMAT_ENABLE, SNORM8_UNORM16_SNORM16_TRUE);

   P_IMMD(p, NV9097, SET_ATTRIBUTE_DEFAULT, {
      .color_front_diffuse    = COLOR_FRONT_DIFFUSE_VECTOR_0001,
      .color_front_specular   = COLOR_FRONT_SPECULAR_VECTOR_0001,
      .generic_vector         = GENERIC_VECTOR_VECTOR_0001,
      .fixed_fnc_texture      = FIXED_FNC_TEXTURE_VECTOR_0001,
      .dx9_color0             = DX9_COLOR0_VECTOR_0001,
      .dx9_color1_to_color15  = DX9_COLOR1_TO_COLOR15_VECTOR_0000,
   });

   P_IMMD(p, NV9097, SET_DA_OUTPUT, VERTEX_ID_USES_ARRAY_START_TRUE);

   P_IMMD(p, NV9097, SET_RENDER_ENABLE_CONTROL,
                     CONDITIONAL_LOAD_CONSTANT_BUFFER_FALSE);

   P_IMMD(p, NV9097, SET_PS_OUTPUT_SAMPLE_MASK_USAGE, {
      .enable                       = ENABLE_TRUE,
      .qualify_by_anti_alias_enable = QUALIFY_BY_ANTI_ALIAS_ENABLE_ENABLE,
   });

   P_IMMD(p, NV9097, SET_BLEND_OPT_CONTROL, ALLOW_FLOAT_PIXEL_KILLS_TRUE);
   P_IMMD(p, NV9097, SET_BLEND_FLOAT_OPTION, ZERO_TIMES_ANYTHING_IS_ZERO_TRUE);
   P_IMMD(p, NV9097, SET_BLEND_STATE_PER_TARGET, ENABLE_TRUE);

   P_IMMD(p, NV9097, SET_ALPHA_TEST, ENABLE_FALSE);
   P_IMMD(p, NV9097, SET_TWO_SIDED_LIGHT, ENABLE_FALSE);
   P_IMMD(p, NV9097, SET_COLOR_CLAMP, ENABLE_TRUE);
   P_IMMD(p, NV9097, SET_PS_SATURATE, {
      .output0 = OUTPUT0_FALSE,
      .output1 = OUTPUT1_FALSE,
      .output2 = OUTPUT2_FALSE,
      .output3 = OUTPUT3_FALSE,
      .output4 = OUTPUT4_FALSE,
      .output5 = OUTPUT5_FALSE,
      .output6 = OUTPUT6_FALSE,
      .output7 = OUTPUT7_FALSE,
   });

   P_IMMD(p, NV9097, SET_ZPASS_PIXEL_COUNT, ENABLE_FALSE);
   P_IMMD(p, NV9097, SET_STATISTICS_COUNTER, {
      .da_vertices_generated_enable = false,
      .da_primitives_generated_enable = false,
      .vs_invocations_enable = false,
      .gs_invocations_enable = false,
      .gs_primitives_generated_enable = false,
      .streaming_primitives_succeeded_enable = false,
      .streaming_primitives_needed_enable = false,
      .clipper_invocations_enable = false,
      .clipper_primitives_generated_enable = false,
      .ps_invocations_enable = false,
      .ti_invocations_enable = false,
      .ts_invocations_enable = false,
      .ts_primitives_generated_enable = false,
      .total_streaming_primitives_needed_succeeded_enable = false,
      .vtg_primitives_out_enable = false,
   });

   P_IMMD(p, NV9097, SET_POINT_SIZE, fui(1.0));
   P_IMMD(p, NV9097, SET_ATTRIBUTE_POINT_SIZE, { .enable = ENABLE_TRUE });

   P_IMMD(p, NV9097, SET_POINT_SPRITE, ENABLE_TRUE);
   P_IMMD(p, NV9097, SET_POINT_SPRITE_SELECT, {
      .rmode      = RMODE_ZERO,
      .origin     = ORIGIN_TOP,
      .texture0   = TEXTURE0_PASSTHROUGH,
      .texture1   = TEXTURE1_PASSTHROUGH,
      .texture2   = TEXTURE2_PASSTHROUGH,
      .texture3   = TEXTURE3_PASSTHROUGH,
      .texture4   = TEXTURE4_PASSTHROUGH,
      .texture5   = TEXTURE5_PASSTHROUGH,
      .texture6   = TEXTURE6_PASSTHROUGH,
      .texture7   = TEXTURE7_PASSTHROUGH,
      .texture8   = TEXTURE8_PASSTHROUGH,
      .texture9   = TEXTURE9_PASSTHROUGH,
   });

   /* OpenGL's GL_POINT_SMOOTH */
   P_IMMD(p, NV9097, SET_ANTI_ALIASED_POINT, ENABLE_FALSE);

   P_IMMD(p, NVB197, SET_FILL_VIA_TRIANGLE, MODE_DISABLED);

   P_IMMD(p, NV9097, SET_POLY_SMOOTH, ENABLE_FALSE);

   P_IMMD(p, NV9097, SET_VIEWPORT_PIXEL, CENTER_AT_HALF_INTEGERS);

   P_IMMD(p, NV9097, SET_MME_SHADOW_SCRATCH(NVK_MME_SCRATCH_SHADING_RATE_CONTROL), 0);
   /* Ancestor: nvk_mme_anti_alias_init(). A valid value that is never set,
    * so the macro runs the first time anything is.
    */
   P_IMMD(p, NV9097, SET_MME_SHADOW_SCRATCH(NVK_MME_SCRATCH_ANTI_ALIAS), 0xf);

   /* Enable multisample rasterization even for one sample rasterization,
    * this way we get strict lines and rectangular line support.
    */
   P_IMMD(p, NV9097, SET_ANTI_ALIAS_ENABLE, V_TRUE);

   P_IMMD(p, NVB197, SET_POST_PS_INITIAL_COVERAGE, true);
   P_IMMD(p, NVB197, SET_OFFSET_RENDER_TARGET_INDEX,
                     BY_VIEWPORT_INDEX_FALSE);

   P_IMMD(p, NV9097, SET_WINDOW_ORIGIN, {
      .mode    = MODE_UPPER_LEFT,
      .flip_y  = FLIP_Y_FALSE,
   });

   P_MTHD(p, NV9097, SET_WINDOW_OFFSET_X);
   P_NV9097_SET_WINDOW_OFFSET_X(p, 0);
   P_NV9097_SET_WINDOW_OFFSET_Y(p, 0);

   P_IMMD(p, NV9097, SET_ZCULL_BOUNDS, {
      .z_min_unbounded_enable = false,
      .z_max_unbounded_enable = false,
   });
   P_IMMD(p, NV9097, SET_ZCULL, {
      .z_enable = true,
      .stencil_enable = false,
   });

   P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_UPDATE_WINDOW_CLIP));
   P_INLINE_DATA(p, 1);
   P_IMMD(p, NV9097, SET_CLIP_ID_TEST, ENABLE_FALSE);

   P_IMMD(p, NV9097, SET_VIEWPORT_SCALE_OFFSET, ENABLE_TRUE);

   P_IMMD(p, NV9097, SET_VIEWPORT_CLIP_CONTROL, {
      .min_z_zero_max_z_one      = MIN_Z_ZERO_MAX_Z_ONE_FALSE,
      .pixel_min_z               = PIXEL_MIN_Z_CLAMP,
      .pixel_max_z               = PIXEL_MAX_Z_CLAMP,
      .geometry_guardband        = GEOMETRY_GUARDBAND_SCALE_256,
      .line_point_cull_guardband = LINE_POINT_CULL_GUARDBAND_SCALE_256,
      .geometry_clip             = GEOMETRY_CLIP_WZERO_CLIP,
      .geometry_guardband_z      = GEOMETRY_GUARDBAND_Z_SAME_AS_XY_GUARDBAND,
   });

   for (unsigned i = 0; i < 16; i++)
      P_IMMD(p, NV9097, SET_SCISSOR_ENABLE(i), V_FALSE);

   P_IMMD(p, NV9097, SET_CT_MRT_ENABLE, V_TRUE);

   /* The eight hardware root tables: visible to every stage, zeroed.
    * Shaders read them as constant buffers 24 to 31.
    */
   P_MTHD(p, NVC597, SET_ROOT_TABLE_VISIBILITY(0));
   for (int i = 0; i < 8; i++) {
      P_NVC597_SET_ROOT_TABLE_VISIBILITY(p, i, {
         .binding_group0_enable = 0x3,
         .binding_group1_enable = 0x3,
         .binding_group2_enable = 0x3,
         .binding_group3_enable = 0x3,
         .binding_group4_enable = 0x3,
      });
   }

   for (int i = 0; i < 8; i++) {
      P_1INC(p, NVC597, SET_ROOT_TABLE_SELECTOR);
      P_NVC597_SET_ROOT_TABLE_SELECTOR(p, {
         .root_table = i,
         .offset = 0,
      });
      for (uint32_t dw = 0; dw < 64; dw++)
         P_INLINE_DATA(p, 0);
   }

   if (info->cls_eng3d >= AMPERE_B)
      P_IMMD(p, NVC797, SET_ROOT_TABLE_PREFETCH, 0x3f);

   /* I don't know what these values actually mean.  I just copied them
    * from the way the blob sets up the hardware. (The ancestor's words.)
    */
   P_MTHD(p, NVC597, SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(0));
   P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 0, 0xa23eb139);
   P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 1, 0xfb72ea61);
   P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 2, 0xd950c843);
   P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 3, 0x88fac4e5);
   P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 4, 0x1ab3e1b6);
   P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 5, 0xa98fedc2);
   P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 6, 0x2107654b);
   P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 7, 0xe0539773);
   P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 8, 0x698badcf);
   P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 9, 0x71032547);
   P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 10, 0xdef05397);
   P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 11, 0x56789abc);
   P_NVC597_SET_VARIABLE_PIXEL_RATE_SAMPLE_ORDER(p, 12, 0x1234);

   /* A 32-bit window at the top of the low 4 GiB, not reserved in the VA
    * allocator: NVK's own TODO, inherited.
    */
   P_IMMD(p, NV9097, SET_SHADER_LOCAL_MEMORY_WINDOW, 0xffu << 24);

   for (uint32_t group = 0; group < 5; group++) {
      for (uint32_t slot = 0; slot < 16; slot++) {
         P_IMMD(p, NV9097, BIND_GROUP_CONSTANT_BUFFER(group), {
            .valid = VALID_FALSE,
            .shader_slot = slot,
         });
      }
   }

   P_IMMD(p, NV9097, SET_RT_LAYER, {
      .v = 0,
      .control = CONTROL_V_SELECTS_LAYER,
   });

   P_IMMD(p, NV9097, SET_POINT_CENTER_MODE, V_OGL);
   P_IMMD(p, NV9097, SET_EDGE_FLAG, V_TRUE);
   P_IMMD(p, NV9097, SET_SAMPLER_BINDING, V_INDEPENDENTLY);
   P_IMMD(p, NV9097, SET_PRIMITIVE_TOPOLOGY_CONTROL, OVERRIDE_USE_SEPARATE_TOPOLOGY_STATE);

   P_MTHD(p, NV9097, SET_VERTEX_STREAM_SUBSTITUTE_A);
   P_NV9097_SET_VERTEX_STREAM_SUBSTITUTE_A(p, zero_addr >> 32);
   P_NV9097_SET_VERTEX_STREAM_SUBSTITUTE_B(p, zero_addr);

   /* These WATERMARK settings are based on what the blob sets. */
   P_IMMD(p, NV9097, SET_PS_WARP_WATERMARKS, {
      .low = 0x8,
      .high = info->max_warps_per_mp * info->mp_per_tpc,
   });
   P_IMMD(p, NV9097, SET_PS_REGISTER_WATERMARKS, {
      .low = 0x80,
      .high = 0x1000,
   });

   P_MTHD(p, NV9097, SET_MME_SHADOW_SCRATCH(NVK_MME_SCRATCH_VB_ENABLES));
   P_NV9097_SET_MME_SHADOW_SCRATCH(p, NVK_MME_SCRATCH_VB_ENABLES, 0);
   for (uint32_t b = 0; b < 32; b++) {
      P_IMMD(p, NV9097, SET_VERTEX_STREAM_A_FORMAT(b), {
         .enable = false,
      });
   }

   if (info->cls_eng3d >= HOPPER_A)
       P_IMMD(p, NVCB97, SET_TEXTURE_HEADER_VERSION, 1);

   /* Store the address to CB0 in a pair of state registers */
   P_MTHD(p, NV9097, SET_MME_SHADOW_SCRATCH(NVK_MME_SCRATCH_CB0_ADDR_HI));
   P_NV9097_SET_MME_SHADOW_SCRATCH(p, NVK_MME_SCRATCH_CB0_ADDR_HI, cb0_addr >> 32);
   P_NV9097_SET_MME_SHADOW_SCRATCH(p, NVK_MME_SCRATCH_CB0_ADDR_LO, cb0_addr);

   /* Store the address to the zero page in a pair of state registers */
   P_MTHD(p, NV9097, SET_MME_SHADOW_SCRATCH(NVK_MME_SCRATCH_ZERO_ADDR_HI));
   P_NV9097_SET_MME_SHADOW_SCRATCH(p, NVK_MME_SCRATCH_ZERO_ADDR_HI, zero_addr >> 32);
   P_NV9097_SET_MME_SHADOW_SCRATCH(p, NVK_MME_SCRATCH_ZERO_ADDR_LO, zero_addr);

   /* We leave CB0 selected by default */
   P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_SELECT_CB0));
   P_INLINE_DATA(p, 0);

   /* Bind CB0 to all shader groups */
   for (uint32_t group = 0; group < 5; group++) {
      P_IMMD(p, NV9097, BIND_GROUP_CONSTANT_BUFFER(group), {
         .valid = VALID_TRUE,
         .shader_slot = 0,
      });
   }

   /* Zero out CB0 */
   P_1INC(p, NV9097, LOAD_CONSTANT_BUFFER_OFFSET);
   P_NV9097_LOAD_CONSTANT_BUFFER_OFFSET(p, 0);
   for (uint32_t dw = 0; dw < KGP_DRAW_CB0_SIZE / 4; dw++)
      P_INLINE_DATA(p, 0);

   /* These are shadowed in cb0 so they need to be zeroed as well for
    * consistency.
    */
   P_IMMD(p, NV9097, SET_GLOBAL_BASE_INSTANCE_INDEX, 0);
   P_MTHD(p, NV9097, SET_MME_SHADOW_SCRATCH(NVK_MME_SCRATCH_CB0_FIRST_VERTEX));
   P_NV9097_SET_MME_SHADOW_SCRATCH(p, NVK_MME_SCRATCH_CB0_FIRST_VERTEX, 0);
   P_NV9097_SET_MME_SHADOW_SCRATCH(p, NVK_MME_SCRATCH_CB0_MESH_GROUP_COUNT_X, 0);
   P_NV9097_SET_MME_SHADOW_SCRATCH(p, NVK_MME_SCRATCH_CB0_MESH_GROUP_COUNT_Y, 0);
   P_NV9097_SET_MME_SHADOW_SCRATCH(p, NVK_MME_SCRATCH_CB0_MESH_GROUP_COUNT_Z, 0);
   P_NV9097_SET_MME_SHADOW_SCRATCH(p, NVK_MME_SCRATCH_CB0_DRAW_INDEX, 0);
   P_NV9097_SET_MME_SHADOW_SCRATCH(p, NVK_MME_SCRATCH_CB0_VIEW_INDEX, 0);

   return true;
}

/* Graphics pipeline binding ------------------------------------------------ */

/* Ancestor: mesa_to_nv9097_shader_type. */
static uint32_t
kgp_shader_type(const struct nak_shader_info *sh)
{
   /* A task shader, or a mesh shader without one, takes the vertex
    * shader's place; a mesh shader behind a task shader takes the
    * tessellation evaluation shader's.
    */
   switch (sh->stage) {
   case MESA_SHADER_FRAGMENT:
      return NV9097_SET_PIPELINE_SHADER_TYPE_PIXEL;
   case MESA_SHADER_MESH:
      return sh->mesh.has_task_shader
             ? NV9097_SET_PIPELINE_SHADER_TYPE_TESSELLATION
             : NV9097_SET_PIPELINE_SHADER_TYPE_VERTEX;
   default:
      assert(sh->stage == MESA_SHADER_VERTEX || sh->stage == MESA_SHADER_TASK);
      return NV9097_SET_PIPELINE_SHADER_TYPE_VERTEX;
   }
}

/* Ancestors: nvk_pipeline_bind_group, nvk_cbuf_binding_for_stage. */
static uint32_t
kgp_bind_group(const struct nak_shader_info *sh)
{
   if (sh->stage == MESA_SHADER_MESH)
      return sh->mesh.has_task_shader ? MESA_SHADER_TESS_EVAL
                                      : MESA_SHADER_VERTEX;
   return sh->stage == MESA_SHADER_TASK ? MESA_SHADER_VERTEX : sh->stage;
}

/* Ancestor: nvk_shader_fill_push up to "Stash this", vertex and pixel
 * stages. No sample-shading API state: only what the shader itself asks for.
 */
static void
kgp_shader_fill_push(const struct nv_device_info *info, struct nv_push *p,
                     const struct kgp_gfx_shader *shader)
{
   const struct nak_shader_info *sh = shader->info;
   const uint32_t type = kgp_shader_type(sh);

   /* We always map index == type */
   const uint32_t idx = type;

   P_IMMD(p, NV9097, SET_PIPELINE_SHADER(idx), {
      .enable  = ENABLE_TRUE,
      .type    = type,
   });

   const uint64_t addr = shader->hdr_addr;
   P_MTHD(p, NVC397, SET_PIPELINE_PROGRAM_ADDRESS_A(idx));
   P_NVC397_SET_PIPELINE_PROGRAM_ADDRESS_A(p, idx, addr >> 32);
   P_NVC397_SET_PIPELINE_PROGRAM_ADDRESS_B(p, idx, addr);

   /* On Ampere B and later, we can be prefetched for up to 127 blocks of a
    * shader. */
   if (info->cls_eng3d >= AMPERE_B) {
      P_NVC797_SET_PIPELINE_PROGRAM_PREFETCH(
         p, idx, MIN2(DIV_ROUND_UP(shader->size_B, 256), 127));
   }

   /* The binding group is the stage, as for constant buffers. */
   P_MTHD(p, NVC397, SET_PIPELINE_REGISTER_COUNT(idx));
   P_NVC397_SET_PIPELINE_REGISTER_COUNT(p, idx, sh->num_gprs);
   P_NVC397_SET_PIPELINE_BINDING(p, idx, kgp_bind_group(sh));

   /* Set by the first stage: on when that is not a vertex shader. */
   if (type == NV9097_SET_PIPELINE_SHADER_TYPE_VERTEX) {
      P_IMMD(p, NVC597, SET_MESH_CONTROL, sh->stage != MESA_SHADER_VERTEX);
   }

   if (sh->stage == MESA_SHADER_TASK) {
      const uint16_t smem_lines = DIV_ROUND_UP(sh->task.smem_size, 128);
      const uint16_t task_smem_lines =
         DIV_ROUND_UP(sh->task.payload_smem_size, 128);

      /* Task payload should be part of shared memory */
      assert(task_smem_lines <= smem_lines);

      P_IMMD(p, NVC597, SET_MESH_INIT_SHADER, {
         .thread_count = sh->task.local_size,
         .local_buffer_lines = smem_lines,
         .output_to_m_s_lines = task_smem_lines,
      });
   } else if (sh->stage == MESA_SHADER_MESH) {
      assert(sh->mesh.max_vertices != 0 && sh->mesh.max_primitives != 0);

      /* The shadow-RAM mode around the two methods is a workaround NVK
       * takes from the proprietary driver; see the ancestor.
       */
      P_IMMD(p, NV9097, SET_MME_SHADOW_RAM_CONTROL, MODE_METHOD_TRACK);
      P_MTHD(p, NVC597, SET_MESH_SHADER_A);
      P_NVC597_SET_MESH_SHADER_A(p, {
         .output_topology = sh->mesh.topology,
         .max_vertex = sh->mesh.max_vertices,
         .max_primitive = sh->mesh.max_primitives,
      });
      P_NVC597_SET_MESH_SHADER_B(p, {
         .shared_mem_lines = DIV_ROUND_UP(sh->mesh.smem_size, 128),
         .thread_count = sh->mesh.local_size,
      });
      P_IMMD(p, NV9097, SET_MME_SHADOW_RAM_CONTROL,
             MODE_METHOD_TRACK_WITH_FILTER);

      if (sh->mesh.has_gs_sph) {
         const uint32_t gs = NV9097_SET_PIPELINE_SHADER_TYPE_GEOMETRY;
         P_IMMD(p, NV9097, SET_PIPELINE_SHADER(gs), {
            .enable  = true,
            .type    = TYPE_GEOMETRY,
         });
         P_MTHD(p, NVC397, SET_PIPELINE_PROGRAM_ADDRESS_A(gs));
         P_NVC397_SET_PIPELINE_PROGRAM_ADDRESS_A(p, gs,
                                                 shader->gs_hdr_addr >> 32);
         P_NVC397_SET_PIPELINE_PROGRAM_ADDRESS_B(p, gs, shader->gs_hdr_addr);
         P_IMMD(p, NVC397, SET_GS_MODE, TYPE_ANY);
      }
   } else if (sh->stage == MESA_SHADER_FRAGMENT) {
      P_MTHD(p, NVC397, SET_SUBTILING_PERF_KNOB_A);
      P_NV9097_SET_SUBTILING_PERF_KNOB_A(p, {
         .fraction_of_spm_register_file_per_subtile         = 0x10,
         .fraction_of_spm_pixel_output_buffer_per_subtile   = 0x40,
         .fraction_of_spm_triangle_ram_per_subtile          = 0x16,
         .fraction_of_max_quads_per_subtile                 = 0x20,
      });
      P_NV9097_SET_SUBTILING_PERF_KNOB_B(p, 0x20);

      P_IMMD(p, NV9097, SET_API_MANDATED_EARLY_Z,
             sh->fs.early_fragment_tests);

      P_IMMD(p, NVB197, SET_POST_Z_PS_IMASK, sh->fs.post_depth_coverage);

      P_IMMD(p, NV9097, SET_ZCULL_BOUNDS, {
         .z_min_unbounded_enable = sh->fs.writes_depth,
         .z_max_unbounded_enable = sh->fs.writes_depth,
      });

      /* Sample shading disables shading-rate control and runs once per
       * sample; see the ancestor for the spec text.
       */
      P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_SET_SHADING_RATE_CONTROL));
      P_INLINE_DATA(p, nvk_mme_shading_rate_control_sample_shading(
         sh->fs.uses_sample_shading));

      P_1INC(p, NVB197, CALL_MME_MACRO(NVK_MME_SET_ANTI_ALIAS));
      P_INLINE_DATA(p, nvk_mme_anti_alias_min_sample_shading(
         sh->fs.uses_sample_shading ? 1 : 0));
   }
}

/* Ancestor: the tail of nvk_shader_fill_push, for the last geometry stage
 * (here always the vertex shader): layer, clip/cull planes, transform
 * feedback off.
 */
static void
kgp_shader_fill_push_vtgm(const struct nv_device_info *info,
                          struct nv_push *p, const struct nak_shader_info *sh)
{
   P_IMMD(p, NV9097, SET_RT_LAYER, {
      .v       = 0,
      .control = sh->vtg.writes_layer ?
                 CONTROL_GEOMETRY_SHADER_SELECTS_LAYER :
                 CONTROL_V_SELECTS_LAYER,
   });

   if (info->cls_eng3d >= AMPERE_B) {
      P_IMMD(p, NVC797, SET_VARIABLE_PIXEL_RATE_SHADING_TABLE_SELECT, {
         .source = sh->vtg.writes_vprs_table_index ?
                   SOURCE_FROM_VPRS_TABLE_INDEX :
                   SOURCE_FROM_CONSTANT,
         .source_constant_value = 0,
      });
   }

   const uint8_t clip_enable = sh->vtg.clip_enable;
   const uint8_t cull_enable = sh->vtg.cull_enable;
   P_IMMD(p, NV9097, SET_USER_CLIP_ENABLE, {
      .plane0 = ((clip_enable | cull_enable) >> 0) & 1,
      .plane1 = ((clip_enable | cull_enable) >> 1) & 1,
      .plane2 = ((clip_enable | cull_enable) >> 2) & 1,
      .plane3 = ((clip_enable | cull_enable) >> 3) & 1,
      .plane4 = ((clip_enable | cull_enable) >> 4) & 1,
      .plane5 = ((clip_enable | cull_enable) >> 5) & 1,
      .plane6 = ((clip_enable | cull_enable) >> 6) & 1,
      .plane7 = ((clip_enable | cull_enable) >> 7) & 1,
   });
   P_IMMD(p, NV9097, SET_USER_CLIP_OP, {
      .plane0 = (cull_enable >> 0) & 1,
      .plane1 = (cull_enable >> 1) & 1,
      .plane2 = (cull_enable >> 2) & 1,
      .plane3 = (cull_enable >> 3) & 1,
      .plane4 = (cull_enable >> 4) & 1,
      .plane5 = (cull_enable >> 5) & 1,
      .plane6 = (cull_enable >> 6) & 1,
      .plane7 = (cull_enable >> 7) & 1,
   });

   const struct nak_xfb_info *xfb = &sh->vtg.xfb;
   for (uint8_t b = 0; b < ARRAY_SIZE(xfb->attr_count); b++) {
      const uint8_t attr_count = xfb->attr_count[b];

      P_MTHD(p, NV9097, SET_STREAM_OUT_CONTROL_STREAM(b));
      P_NV9097_SET_STREAM_OUT_CONTROL_STREAM(p, b, xfb->stream[b]);
      P_NV9097_SET_STREAM_OUT_CONTROL_COMPONENT_COUNT(p, b, attr_count);
      P_NV9097_SET_STREAM_OUT_CONTROL_STRIDE(p, b, xfb->stride[b]);

      if (attr_count > 0) {
         /* upload packed varying indices in multiples of 4 bytes */
         const uint32_t n = DIV_ROUND_UP(attr_count, 4);
         P_MTHD(p, NV9097, SET_STREAM_OUT_LAYOUT_SELECT(b, 0));
         P_INLINE_ARRAY(p, (const uint32_t*)xfb->attr_index[b], n);
      }
   }
}

/* Builds pipeline->gfx_push: what NVK's first draw after binding these two
 * shaders emits for them, in its order. Shader types ascend; the three
 * stages KoreGP does not have are disabled explicitly (the hardware keeps
 * whatever was last bound); then the last geometry stage's tail; then each
 * stage's constant data as its constant buffer 1, leaving constant buffer 0
 * selected as the macros expect.
 * Ancestors: nvk_cmd_flush_gfx_shaders, nvk_cmd_flush_gfx_cbufs.
 */
bool
kgp_gfx_pipeline_build_push(const struct nv_device_info *info,
                            struct KgpPipeline_T *pipeline,
                            const struct kgp_gfx_shader *shaders,
                            unsigned count)
{
   /* The last stage before the rasterizer: vertex or mesh. */
   const struct kgp_gfx_shader *vs = &shaders[count - 2];
   const struct kgp_gfx_shader *by_type[6] = { NULL };
   for (unsigned i = 0; i < count; i++)
      by_type[kgp_shader_type(shaders[i].info)] = &shaders[i];

   uint32_t push_dw[512];
   struct nv_push push, *p = &push;
   nv_push_init(&push, push_dw, ARRAY_SIZE(push_dw),
                BITFIELD_BIT(SUBC_NV9097));

   static const uint32_t types[] = {
      NV9097_SET_PIPELINE_SHADER_TYPE_VERTEX,
      NV9097_SET_PIPELINE_SHADER_TYPE_TESSELLATION_INIT,
      NV9097_SET_PIPELINE_SHADER_TYPE_TESSELLATION,
      NV9097_SET_PIPELINE_SHADER_TYPE_GEOMETRY,
      NV9097_SET_PIPELINE_SHADER_TYPE_PIXEL,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(types); i++) {
      assert(i == 0 || types[i] > types[i - 1]);
      const uint32_t type = types[i];
      if (by_type[type] != NULL) {
         kgp_shader_fill_push(info, p, by_type[type]);
      } else if (type == NV9097_SET_PIPELINE_SHADER_TYPE_GEOMETRY &&
                 vs->info->stage == MESA_SHADER_MESH &&
                 vs->info->mesh.has_gs_sph) {
         /* The mesh shader's block has bound its geometry header there. */
      } else {
         P_IMMD(p, NV9097, SET_PIPELINE_SHADER(type), {
            .enable  = ENABLE_FALSE,
            .type    = type,
         });
      }
   }

   kgp_shader_fill_push_vtgm(info, p, vs->info);

   bool bound_any_cbuf = false;
   for (unsigned i = 0; i < count; i++) {
      const struct kgp_gfx_shader *shader = &shaders[i];
      if (shader->data_size_B == 0)
         continue;

      assert(shader->data_addr % KGP_CBUF_ALIGN_B == 0);
      P_MTHD(p, NV9097, SET_CONSTANT_BUFFER_SELECTOR_A);
      P_NV9097_SET_CONSTANT_BUFFER_SELECTOR_A(p, shader->data_size_B);
      P_NV9097_SET_CONSTANT_BUFFER_SELECTOR_B(p, shader->data_addr >> 32);
      P_NV9097_SET_CONSTANT_BUFFER_SELECTOR_C(p, shader->data_addr);

      P_IMMD(p, NV9097,
             BIND_GROUP_CONSTANT_BUFFER(kgp_bind_group(shader->info)), {
         .valid = VALID_TRUE,
         .shader_slot = KGP_CBUF_SHADER_DATA,
      });
      bound_any_cbuf = true;
   }
   if (bound_any_cbuf) {
      P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_SELECT_CB0));
      P_INLINE_DATA(p, 0);
   }

   pipeline->gfx_push_dw = nv_push_dw_count(&push);
   pipeline->gfx_push = malloc(pipeline->gfx_push_dw * sizeof(uint32_t));
   if (pipeline->gfx_push == NULL)
      return false;
   memcpy(pipeline->gfx_push, push_dw,
          pipeline->gfx_push_dw * sizeof(uint32_t));
   return true;
}

/* Render passes ------------------------------------------------------------ */

/* Ancestor: nvk_cmd_buffer_begin_graphics. Chunks and heaps are recycled
 * between command buffers, so the 3D engine's caches of them are dropped at
 * the top of every command buffer that draws (a compute-only one skips it).
 */
static void
kgp_cb_begin_graphics(struct KgpCommandBuffer_T *cb)
{
   if (cb->gfx_begun)
      return;
   cb->gfx_begun = true;

   struct nv_push *p = kgp_cb_push(cb, 5);
   P_MTHD(p, NV9097, INVALIDATE_SAMPLER_CACHE_NO_WFI);
   P_NV9097_INVALIDATE_SAMPLER_CACHE_NO_WFI(p, {
      .lines = LINES_ALL
   });
   P_NV9097_INVALIDATE_TEXTURE_HEADER_CACHE_NO_WFI(p, {
      .lines = LINES_ALL
   });

   P_IMMD(p, NVA097, INVALIDATE_SHADER_CACHES_NO_WFI, {
      .constant = CONSTANT_TRUE
   });
}

/* Ancestor: nvk_cmd_set_sample_layout, Maxwell B and later. The scratch
 * values are which samples each pass of per-sample shading covers; NVK's
 * macros read them. Needs 14 words.
 */
static void
kgp_push_sample_layout(struct nv_push *p, enum nil_sample_layout sample_layout)
{
   static const struct {
      uint32_t mode, pass2[4], pass4[4];
   } layouts[] = {
#define MODE(S) .mode = NV9097_SET_ANTI_ALIAS_SAMPLES_MODE_##S
      [NIL_SAMPLE_LAYOUT_1X1] = { MODE(1X1) },
      [NIL_SAMPLE_LAYOUT_2X1] = { MODE(2X1) },
      [NIL_SAMPLE_LAYOUT_2X1_D3D] = { MODE(2X1_D3D) },
      [NIL_SAMPLE_LAYOUT_2X2] = {
         MODE(2X2), { 0x000a0005, 0x000a0005, 0, 0 },
      },
      [NIL_SAMPLE_LAYOUT_4X2] = {
         MODE(4X2),
         { 0x000f000f, 0x000f000f, 0x00f000f0, 0x00f000f0 },
         { 0x00030003, 0x000c000c, 0x00300030, 0x00c000c0 },
      },
      [NIL_SAMPLE_LAYOUT_4X2_D3D] = {
         MODE(4X2_D3D),
         { 0x003a00c5, 0x003a00c5, 0x003a003a, 0x00c500c5 },
         { 0x00120081, 0x00280044, 0x00280012, 0x00810044 },
      },
#undef MODE
   };
   assert(sample_layout < ARRAY_SIZE(layouts));

   P_IMMD(p, NV9097, SET_ANTI_ALIAS, layouts[sample_layout].mode);
   P_MTHD(p, NV9097, SET_MME_SHADOW_SCRATCH(NVK_MME_SCRATCH_SAMPLE_MASKS_2PASS_0));
   P_INLINE_ARRAY(p, layouts[sample_layout].pass2, 4);
   P_MTHD(p, NV9097, SET_MME_SHADOW_SCRATCH(NVK_MME_SCRATCH_SAMPLE_MASKS_4PASS_0));
   P_INLINE_ARRAY(p, layouts[sample_layout].pass4, 4);

   /* Ancestor: nvk_mme_anti_alias_samples: log2 of the sample count in
    * bits 7:4.
    */
   const uint32_t samples = nil_sample_layout_samples(sample_layout);
   P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_SET_ANTI_ALIAS));
   P_INLINE_DATA(p, nvk_mme_val_mask(util_logbase2(samples) << 4, 0x00f0));
}

/* Vulkan's standard sample locations, in sixteenths of a pixel, x in the
 * low nibble (struct nak_sample_location), by log2 of the sample count.
 * Ancestor: vk_standard_sample_locations_state.
 */
static const uint8_t kgp_sample_locations[4][8] = {
   { 0x88 },
   { 0xcc, 0x44 },
   { 0x26, 0x6e, 0xa2, 0xea },
   { 0x59, 0xb7, 0x9d, 0x35, 0xd3, 0x71, 0xfb, 0x1f },
};

/* Ancestors: nvk_CmdBeginRendering (no stencil, no stencil, no ZCULL, no
 * multiview, no shading-rate attachment), nvk_cmd_set_sample_layout, and
 * for the clears
 * nvk_CmdClearAttachments + emit_clear_rects.
 */
KGP_API void
kgpBeginRenderPass(KgpCommandBuffer cb, const KgpRenderPassDesc *desc)
{
   const struct nv_device_info *info = kgp_dev_info(cb->queue->dev);

   if (unlikely(cb->rp_active || desc == NULL ||
                desc->colorTargetCount == 0 ||
                desc->colorTargetCount > KGP_MAX_COLOR_TARGETS)) {
      kgp_err("kgpBeginRenderPass: needs 1 to %u colour targets and no "
              "render pass already open", KGP_MAX_COLOR_TARGETS);
      cb->failed = true;
      return;
   }

   uint32_t width = 0, height = 0;
   for (uint32_t i = 0; i < desc->colorTargetCount; i++) {
      const KgpColorTarget *ct = &desc->colorTargets[i];
      const struct nil_image *image = ct->texture ? &ct->texture->image : NULL;

      if (unlikely(image == NULL || ct->mip >= image->num_levels ||
                   ct->layer >= image->extent_px.array_len ||
                   image->dim != NIL_IMAGE_DIM_2D ||
                   image->levels[ct->mip].tiling.gob_type ==
                      NIL_GOB_TYPE_LINEAR ||
                   !nil_format_supports_color_targets(
                      info, image->format.p_format))) {
         kgp_err("kgpBeginRenderPass: colour target %u must be an existing "
                 "mip level and layer of a tiled 2D texture in a renderable "
                 "format", i);
         cb->failed = true;
         return;
      }

      const struct nil_Extent4D_Pixels extent =
         nil_image_level_extent_px(image, ct->mip);
      if (i == 0) {
         width = extent.width;
         height = extent.height;
      } else if (unlikely(extent.width != width || extent.height != height)) {
         kgp_err("kgpBeginRenderPass: colour targets differ in extent");
         cb->failed = true;
         return;
      }
   }

   /* One sample count for the whole pass. */
   const enum nil_sample_layout sample_layout =
      desc->colorTargets[0].texture->image.sample_layout;
   bool same_samples =
      desc->depthTarget.texture == NULL ||
      desc->depthTarget.texture->image.sample_layout == sample_layout;
   for (uint32_t i = 1; i < desc->colorTargetCount; i++) {
      same_samples &= desc->colorTargets[i].texture->image.sample_layout ==
                      sample_layout;
   }
   if (unlikely(!same_samples)) {
      kgp_err("kgpBeginRenderPass: the targets differ in sample count");
      cb->failed = true;
      return;
   }

   const KgpDepthTarget *dt = &desc->depthTarget;
   if (dt->texture != NULL) {
      const struct nil_image *image = &dt->texture->image;
      if (unlikely(dt->mip >= image->num_levels ||
                   dt->layer >= image->extent_px.array_len ||
                   !util_format_is_depth_or_stencil(image->format.p_format) ||
                   nil_image_level_extent_px(image, dt->mip).width != width ||
                   nil_image_level_extent_px(image, dt->mip).height != height)) {
         kgp_err("kgpBeginRenderPass: the depth target must be an existing "
                 "mip level and layer of a depth texture, of the colour "
                 "targets' extent");
         cb->failed = true;
         return;
      }
   }

   for (uint32_t i = 0; i < desc->colorTargetCount; i++) {
      if (!kgp_check_texture(cb, "kgpBeginRenderPass: a colour target",
                             desc->colorTargets[i].texture))
         return;
   }
   if (!kgp_check_texture(cb, "kgpBeginRenderPass: the depth target",
                          dt->texture))
      return;

   kgp_cb_begin_graphics(cb);

   struct nv_push *p = kgp_cb_push(cb, KGP_MAX_COLOR_TARGETS * 12 + 46 + 20);

   P_IMMD(p, NV9097, SET_MME_SHADOW_SCRATCH(NVK_MME_SCRATCH_VIEW_MASK), 0);

   P_MTHD(p, NV9097, SET_SURFACE_CLIP_HORIZONTAL);
   P_NV9097_SET_SURFACE_CLIP_HORIZONTAL(p, {
      .x       = 0,
      .width   = width,
   });
   P_NV9097_SET_SURFACE_CLIP_VERTICAL(p, {
      .y       = 0,
      .height  = height,
   });

   /* Every colour target is programmed, used or not, so that nothing is
    * left pointing at a previous render pass's memory.
    */
   for (uint32_t i = 0; i < KGP_MAX_COLOR_TARGETS; i++) {
      if (i < desc->colorTargetCount) {
         const KgpColorTarget *ct = &desc->colorTargets[i];
         const struct nil_image *image = &ct->texture->image;
         const struct nil_image_level *level = &image->levels[ct->mip];
         const struct nil_Extent4D_Samples level_extent_sa =
            nil_image_level_extent_sa(image, ct->mip);
         const enum pipe_format p_format = image->format.p_format;

         const uint64_t addr = ct->texture->addr + level->offset_B +
                               ct->layer * (uint64_t)image->array_stride_B;

         P_MTHD(p, NV9097, SET_COLOR_TARGET_A(i));
         P_NV9097_SET_COLOR_TARGET_A(p, i, addr >> 32);
         P_NV9097_SET_COLOR_TARGET_B(p, i, addr);

         /* The width is the stride in elements, as for copies. */
         const uint32_t row_stride_el =
            level->row_stride_B / util_format_get_blocksize(p_format);
         P_NV9097_SET_COLOR_TARGET_WIDTH(p, i, row_stride_el);
         P_NV9097_SET_COLOR_TARGET_HEIGHT(p, i, level_extent_sa.height);
         P_NV9097_SET_COLOR_TARGET_FORMAT(p, i,
                                          nil_format_to_color_target(p_format));

         P_NV9097_SET_COLOR_TARGET_MEMORY(p, i, {
            .block_width   = BLOCK_WIDTH_ONE_GOB,
            .block_height  = level->tiling.y_log2,
            .block_depth   = level->tiling.z_log2,
            .layout        = LAYOUT_BLOCKLINEAR,
            .third_dimension_control =
               THIRD_DIMENSION_CONTROL_THIRD_DIMENSION_DEFINES_ARRAY_SIZE,
         });

         P_NV9097_SET_COLOR_TARGET_THIRD_DIMENSION(p, i, 1);
         P_NV9097_SET_COLOR_TARGET_ARRAY_PITCH(p, i,
            image->array_stride_B >> 2);
         P_NV9097_SET_COLOR_TARGET_LAYER(p, i, 0);

         P_IMMD(p, NV9097, SET_COLOR_COMPRESSION(i), ENABLE_FALSE);
      } else {
         P_MTHD(p, NV9097, SET_COLOR_TARGET_A(i));
         P_NV9097_SET_COLOR_TARGET_A(p, i, 0);
         P_NV9097_SET_COLOR_TARGET_B(p, i, 0);
         P_NV9097_SET_COLOR_TARGET_WIDTH(p, i, 64);
         P_NV9097_SET_COLOR_TARGET_HEIGHT(p, i, 0);
         P_NV9097_SET_COLOR_TARGET_FORMAT(p, i, V_DISABLED);
         P_NV9097_SET_COLOR_TARGET_MEMORY(p, i, {
            .layout        = LAYOUT_BLOCKLINEAR,
         });
         P_NV9097_SET_COLOR_TARGET_THIRD_DIMENSION(p, i, 1);
         P_NV9097_SET_COLOR_TARGET_ARRAY_PITCH(p, i, 0);
         P_NV9097_SET_COLOR_TARGET_LAYER(p, i, 0);

         P_IMMD(p, NV9097, SET_COLOR_COMPRESSION(i), ENABLE_TRUE);
      }
   }

   /* The depth target. Ancestor: the depth half of nvk_CmdBeginRendering
    * (Turing: combined Z/S formats only, never compressed).
    */
   if (dt->texture != NULL) {
      const struct nil_image *image = &dt->texture->image;
      const struct nil_image_level *level = &image->levels[dt->mip];
      const struct nil_Extent4D_Samples level_extent_sa =
         nil_image_level_extent_sa(image, dt->mip);
      const enum pipe_format p_format = image->format.p_format;
      const uint64_t addr = dt->texture->addr + level->offset_B;

      P_MTHD(p, NV9097, SET_ZT_A);
      P_NV9097_SET_ZT_A(p, addr >> 32);
      P_NV9097_SET_ZT_B(p, addr);
      P_NV9097_SET_ZT_FORMAT(p, nil_format_to_depth_stencil(p_format));
      P_NV9097_SET_ZT_BLOCK_SIZE(p, {
         .width = WIDTH_ONE_GOB,
         .height = level->tiling.y_log2,
         .depth = DEPTH_ONE_GOB,
      });
      P_NV9097_SET_ZT_ARRAY_PITCH(p, image->array_stride_B >> 2);

      P_IMMD(p, NV9097, SET_ZT_SELECT, 1 /* target_count */);

      /* The Z hardware has no tile width: the width is the row stride. */
      const uint32_t row_stride_el =
         level->row_stride_B / util_format_get_blocksize(p_format);
      P_MTHD(p, NV9097, SET_ZT_SIZE_A);
      P_NV9097_SET_ZT_SIZE_A(p, row_stride_el);
      P_NV9097_SET_ZT_SIZE_B(p, level_extent_sa.height);
      P_NV9097_SET_ZT_SIZE_C(p, {
         .third_dimension  = dt->layer + 1,
         .control          = CONTROL_THIRD_DIMENSION_DEFINES_ARRAY_SIZE,
      });

      P_IMMD(p, NV9097, SET_ZT_LAYER, dt->layer);

      P_IMMD(p, NV9097, SET_Z_COMPRESSION, false);
      P_IMMD(p, NVB197, SET_STENCIL_COMPRESSION, false);
      P_IMMD(p, NVC597, SET_ZT_SPARSE, {
         .enable = ENABLE_FALSE,
      });
   } else {
      P_IMMD(p, NV9097, SET_ZT_SELECT, 0 /* target_count */);
   }

   /* ponytail: no ZCULL. NVK sets up a region here when the kernel reports
    * the ZCULL geometry and the depth target is cleared; it is the coarse
    * depth test, an optimization. Without it the hardware tests every
    * sample, which is what NVK itself does on older kernels. Add it (about
    * fifty words of setup from nil_zcull_new) when a depth-heavy scene shows
    * up in a profile.
    */
   P_IMMD(p, NV9097, SET_ACTIVE_ZCULL_REGION, 0x3f);

   P_MTHD(p, NVC597, SET_SHADING_RATE_INDEX_SURFACE_ADDRESS_A(0));
   P_NVC597_SET_SHADING_RATE_INDEX_SURFACE_ADDRESS_A(p, 0, 0);
   P_NVC597_SET_SHADING_RATE_INDEX_SURFACE_ADDRESS_B(p, 0, 0);
   P_NVC597_SET_SHADING_RATE_INDEX_SURFACE_SIZE_A(p, 0, { });
   P_NVC597_SET_SHADING_RATE_INDEX_SURFACE_SIZE_B(p, 0, 0);
   P_NVC597_SET_SHADING_RATE_INDEX_SURFACE_LAYER(p, 0, 0);
   P_NVC597_SET_SHADING_RATE_INDEX_SURFACE_ARRAY_PITCH(p, 0, 0);
   P_NVC597_SET_SHADING_RATE_INDEX_SURFACE_BLOCK_SIZE(p, 0, { });
   P_NVC597_SET_SHADING_RATE_INDEX_SURFACE_ALLOCATED_SIZE(p, 0, 0);

   p = kgp_cb_push(cb, 14);
   kgp_push_sample_layout(p, sample_layout);

   /* Depth is cleared along with the first colour target that is, or on
    * its own after them: the order of nvk_CmdClearAttachments.
    */
   bool clear_depth = dt->texture != NULL && dt->load == KGP_LOAD_CLEAR;
   bool clear_stencil = clear_depth &&
      util_format_has_stencil(
         util_format_description(dt->texture->image.format.p_format));
   bool any_clear = clear_depth;
   for (uint32_t i = 0; i < desc->colorTargetCount; i++)
      any_clear |= desc->colorTargets[i].load == KGP_LOAD_CLEAR;

   if (any_clear) {
      p = kgp_cb_push(cb, 6 + 4 + (desc->colorTargetCount + 1) * 11);
      P_IMMD(p, NV9097, SET_RENDER_ENABLE_OVERRIDE, MODE_ALWAYS_RENDER);

      P_IMMD(p, NV9097, SET_CLEAR_SURFACE_CONTROL, {
         .respect_stencil_mask   = RESPECT_STENCIL_MASK_FALSE,
         .use_clear_rect         = USE_CLEAR_RECT_TRUE,
         .use_scissor0           = USE_SCISSOR0_FALSE,
         .use_viewport_clip0     = USE_VIEWPORT_CLIP0_FALSE,
      });

      if (clear_depth)
         P_IMMD(p, NV9097, SET_Z_CLEAR_VALUE, fui(dt->clearDepth));
      if (clear_stencil)
         P_IMMD(p, NV9097, SET_STENCIL_CLEAR_VALUE, dt->clearStencil);

      for (uint32_t i = 0; i <= desc->colorTargetCount; i++) {
         /* One extra round for a depth clear no colour clear has taken. */
         if (i == desc->colorTargetCount) {
            if (!clear_depth)
               break;
            P_MTHD(p, NV9097, SET_CLEAR_RECT_HORIZONTAL);
            P_NV9097_SET_CLEAR_RECT_HORIZONTAL(p, {
               .xmin = 0,
               .xmax = width,
            });
            P_NV9097_SET_CLEAR_RECT_VERTICAL(p, {
               .ymin = 0,
               .ymax = height,
            });
            uint32_t z_payload;
            V_NV9097_CLEAR_SURFACE(z_payload, {
               .z_enable       = true,
               .stencil_enable = clear_stencil,
            });
            P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_CLEAR));
            P_INLINE_DATA(p, z_payload);
            P_INLINE_DATA(p, 1 /* layer count */);
            break;
         }

         const KgpColorTarget *ct = &desc->colorTargets[i];
         if (ct->load != KGP_LOAD_CLEAR)
            continue;

         /* The hardware takes the clear value in the target's own number
          * type: float bits, or integers for integer formats.
          */
         const bool integer = util_format_is_pure_integer(
            ct->texture->image.format.p_format);
         P_MTHD(p, NV9097, SET_COLOR_CLEAR_VALUE(0));
         for (unsigned c = 0; c < 4; c++) {
            P_NV9097_SET_COLOR_CLEAR_VALUE(p, c,
               integer ? (uint32_t)ct->clearColor[c] : fui(ct->clearColor[c]));
         }

         P_MTHD(p, NV9097, SET_CLEAR_RECT_HORIZONTAL);
         P_NV9097_SET_CLEAR_RECT_HORIZONTAL(p, {
            .xmin = 0,
            .xmax = width,
         });
         P_NV9097_SET_CLEAR_RECT_VERTICAL(p, {
            .ymin = 0,
            .ymax = height,
         });

         uint32_t payload;
         V_NV9097_CLEAR_SURFACE(payload, {
            .z_enable       = clear_depth,
            .stencil_enable = clear_stencil,
            .r_enable       = true,
            .g_enable       = true,
            .b_enable       = true,
            .a_enable       = true,
            .mrt_select     = i,
            .rt_array_index = 0,
         });

         P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_CLEAR));
         P_INLINE_DATA(p, payload);
         P_INLINE_DATA(p, 1 /* layer count */);
         clear_depth = false;
         clear_stencil = false;
      }

      P_IMMD(p, NV9097, SET_RENDER_ENABLE_OVERRIDE, MODE_USE_RENDER_ENABLE);
   }

   cb->rp_active = true;
   cb->rp_width = width;
   cb->rp_height = height;
   cb->rp_color_count = desc->colorTargetCount;
   cb->rp_samples = nil_sample_layout_samples(sample_layout);
   for (uint32_t i = 0; i < desc->colorTargetCount; i++)
      cb->rp_formats[i] = desc->colorTargets[i].texture->format;
   cb->rp_depth_format = dt->texture ? dt->texture->format : KGP_FORMAT_NONE;
   cb->gfx_dirty = true;
}

/* Ancestor: nvk_CmdEndRendering2KHR, which for this case emits nothing. */
KGP_API void
kgpEndRenderPass(KgpCommandBuffer cb)
{
   if (unlikely(!cb->rp_active)) {
      kgp_err("kgpEndRenderPass: no render pass is open");
      cb->failed = true;
      return;
   }
   cb->rp_active = false;
}

/* Depth-stencil and blend state --------------------------------------------- */

/* Ancestor: nvk_flush_ds_state with every group dirty. Depth bounds are
 * always off. The hardware's compare values are OpenGL's, 0x200 + the
 * Vulkan compare op, and its stencil ops Direct3D's, 1 + the Vulkan
 * stencil op; KGP_COMPARE and KGP_STENCIL_OP have Vulkan's order (nvk:
 * vk_to_nv9097_compare_op, vk_to_nv9097_stencil_op).
 */
static void
kgp_emit_ds_state(struct nv_push *p, const KgpDepthStencilDesc *desc)
{
   P_IMMD(p, NV9097, SET_DEPTH_TEST, desc->depthTest);
   P_IMMD(p, NV9097, SET_DEPTH_WRITE, desc->depthWrite);
   P_IMMD(p, NV9097, SET_DEPTH_FUNC, 0x200 | desc->depthCompare);
   P_IMMD(p, NV9097, SET_DEPTH_BOUNDS_TEST, false);

   P_MTHD(p, NV9097, SET_DEPTH_BOUNDS_MIN);
   P_NV9097_SET_DEPTH_BOUNDS_MIN(p, fui(0.0f));
   P_NV9097_SET_DEPTH_BOUNDS_MAX(p, fui(0.0f));

   P_IMMD(p, NV9097, SET_STENCIL_TEST, desc->stencilTest);

   const KgpStencilFace *front = &desc->stencilFront;
   const KgpStencilFace *back = &desc->stencilBack;
   P_MTHD(p, NV9097, SET_STENCIL_OP_FAIL);
   P_NV9097_SET_STENCIL_OP_FAIL(p, front->fail + 1);
   P_NV9097_SET_STENCIL_OP_ZFAIL(p, front->depthFail + 1);
   P_NV9097_SET_STENCIL_OP_ZPASS(p, front->pass + 1);
   P_NV9097_SET_STENCIL_FUNC(p, 0x200 | front->compare);

   P_MTHD(p, NV9097, SET_BACK_STENCIL_OP_FAIL);
   P_NV9097_SET_BACK_STENCIL_OP_FAIL(p, back->fail + 1);
   P_NV9097_SET_BACK_STENCIL_OP_ZFAIL(p, back->depthFail + 1);
   P_NV9097_SET_BACK_STENCIL_OP_ZPASS(p, back->pass + 1);
   P_NV9097_SET_BACK_STENCIL_FUNC(p, 0x200 | back->compare);

   P_IMMD(p, NV9097, SET_STENCIL_FUNC_MASK, desc->stencilReadMask);
   P_IMMD(p, NV9097, SET_BACK_STENCIL_FUNC_MASK, desc->stencilReadMask);
   P_IMMD(p, NV9097, SET_STENCIL_MASK, desc->stencilWriteMask);
   P_IMMD(p, NV9097, SET_BACK_STENCIL_MASK, desc->stencilWriteMask);
   P_IMMD(p, NV9097, SET_STENCIL_FUNC_REF, desc->stencilReference);
   P_IMMD(p, NV9097, SET_BACK_STENCIL_FUNC_REF, desc->stencilReference);
}

static bool
kgp_stencil_face_valid(const KgpStencilFace *face)
{
   return (unsigned)face->fail <= KGP_STENCIL_DECR_WRAP &&
          (unsigned)face->depthFail <= KGP_STENCIL_DECR_WRAP &&
          (unsigned)face->pass <= KGP_STENCIL_DECR_WRAP &&
          (unsigned)face->compare <= KGP_COMPARE_ALWAYS;
}

static void
kgp_push_ds_state(struct nv_push *p, const struct KgpDepthStencilState_T *state)
{
   if (state != NULL) {
      nv_push_raw(p, state->push, state->push_dw);
   } else {
      static const KgpDepthStencilDesc off = { 0 };
      kgp_emit_ds_state(p, &off);
   }
}

/* Ancestor: the two depth-bias groups of nvk_flush_rs_state. The unit of
 * the constant is the depth format's resolution, Vulkan's default
 * representation.
 */
static void
kgp_push_depth_bias(struct nv_push *p,
                    const struct KgpDepthStencilState_T *state)
{
   const bool enable = state != NULL && state->bias_enable;

   P_MTHD(p, NV9097, SET_POLY_OFFSET_POINT);
   P_NV9097_SET_POLY_OFFSET_POINT(p, enable);
   P_NV9097_SET_POLY_OFFSET_LINE(p, enable);
   P_NV9097_SET_POLY_OFFSET_FILL(p, enable);

   P_IMMD(p, NV9097, SET_DEPTH_BIAS_CONTROL, DEPTH_FORMAT_DEPENDENT_TRUE);
   P_IMMD(p, NV9097, SET_DEPTH_BIAS, fui(enable ? state->bias : 0.0f));
   P_IMMD(p, NV9097, SET_SLOPE_SCALE_DEPTH_BIAS,
          fui(enable ? state->bias_slope : 0.0f));
   P_IMMD(p, NV9097, SET_DEPTH_BIAS_CLAMP,
          fui(enable ? state->bias_clamp : 0.0f));
}
#define KGP_DEPTH_BIAS_PUSH_DW 12

/* The proposal's state snippet: the methods are built here, once, and a
 * draw copies them.
 */
KGP_API KgpDepthStencilState
kgpCreateDepthStencilState(KgpDevice dev, const KgpDepthStencilDesc *desc)
{
   if (desc == NULL || (unsigned)desc->depthCompare > KGP_COMPARE_ALWAYS ||
       !kgp_stencil_face_valid(&desc->stencilFront) ||
       !kgp_stencil_face_valid(&desc->stencilBack) ||
       !isfinite(desc->depthBias) || !isfinite(desc->depthBiasSlope) ||
       !isfinite(desc->depthBiasClamp)) {
      kgp_err("kgpCreateDepthStencilState: bad desc");
      return NULL;
   }

   struct KgpDepthStencilState_T *state = calloc(1, sizeof(*state));
   if (state == NULL)
      return NULL;

   struct nv_push push;
   nv_push_init(&push, state->push, ARRAY_SIZE(state->push),
                BITFIELD_BIT(SUBC_NV9097));
   kgp_emit_ds_state(&push, desc);
   state->push_dw = nv_push_dw_count(&push);
   state->needs_depth_target = desc->depthTest || desc->depthWrite;
   state->needs_stencil = desc->stencilTest;
   state->bias_enable = desc->depthBias != 0.0f || desc->depthBiasSlope != 0.0f;
   state->bias = desc->depthBias;
   state->bias_slope = desc->depthBiasSlope;
   state->bias_clamp = desc->depthBiasClamp;
   return state;
}

KGP_API void
kgpDestroyDepthStencilState(KgpDevice dev, KgpDepthStencilState state)
{
   free(state);
}

static uint32_t
kgp_blend_factor(KGP_BLEND_FACTOR factor)
{
#define FACTOR(kgp, nv) [KGP_BLEND_##kgp] = \
   NV9097_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF_V_OGL_##nv
   static const uint16_t factors[] = {
      FACTOR(ZERO,          ZERO),
      FACTOR(ONE,           ONE),
      FACTOR(SRC_COLOR,     SRC_COLOR),
      FACTOR(INV_SRC_COLOR, ONE_MINUS_SRC_COLOR),
      FACTOR(DST_COLOR,     DST_COLOR),
      FACTOR(INV_DST_COLOR, ONE_MINUS_DST_COLOR),
      FACTOR(SRC_ALPHA,     SRC_ALPHA),
      FACTOR(INV_SRC_ALPHA, ONE_MINUS_SRC_ALPHA),
      FACTOR(DST_ALPHA,     DST_ALPHA),
      FACTOR(INV_DST_ALPHA, ONE_MINUS_DST_ALPHA),
      FACTOR(CONSTANT_COLOR,     CONSTANT_COLOR),
      FACTOR(INV_CONSTANT_COLOR, ONE_MINUS_CONSTANT_COLOR),
      FACTOR(CONSTANT_ALPHA,     CONSTANT_ALPHA),
      FACTOR(INV_CONSTANT_ALPHA, ONE_MINUS_CONSTANT_ALPHA),
   };
#undef FACTOR
   return factors[factor];
}

static uint32_t
kgp_blend_op(KGP_BLEND_OP op)
{
#define OP(kgp, nv) [KGP_BLEND_OP_##kgp] = \
   NV9097_SET_BLEND_PER_TARGET_COLOR_OP_V_OGL_##nv
   static const uint16_t ops[] = {
      OP(ADD,          FUNC_ADD),
      OP(SUBTRACT,     FUNC_SUBTRACT),
      OP(REV_SUBTRACT, FUNC_REVERSE_SUBTRACT),
      OP(MIN,          MIN),
      OP(MAX,          MAX),
   };
#undef OP
   return ops[op];
}

/* Ancestor: the blend-enable and blend-equation groups of
 * nvk_flush_cb_state. A disabled target still gets an equation, the one a
 * zeroed Vulkan attachment state gives (add, zero, zero).
 *
 * ponytail: emitted method by method, because the number of targets is the
 * render pass's. Prebuild one snippet per target count if blend state
 * changes show up in a profile.
 */
static void
kgp_push_blend_state(struct nv_push *p, const struct KgpBlendState_T *state,
                     uint32_t color_count)
{
   static const KgpBlendDesc off = { 0 };
   const KgpBlendDesc *desc = state ? &state->desc : &off;

   for (uint8_t a = 0; a < color_count; a++)
      P_IMMD(p, NV9097, SET_BLEND(a), state != NULL && state->enable);

   for (uint8_t a = 0; a < color_count; a++) {
      P_MTHD(p, NV9097, SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA(a));
      P_NV9097_SET_BLEND_PER_TARGET_SEPARATE_FOR_ALPHA(p, a, ENABLE_TRUE);
      P_NV9097_SET_BLEND_PER_TARGET_COLOR_OP(p, a,
         kgp_blend_op(desc->colorOp));
      P_NV9097_SET_BLEND_PER_TARGET_COLOR_SOURCE_COEFF(p, a,
         kgp_blend_factor(desc->srcColor));
      P_NV9097_SET_BLEND_PER_TARGET_COLOR_DEST_COEFF(p, a,
         kgp_blend_factor(desc->dstColor));
      P_NV9097_SET_BLEND_PER_TARGET_ALPHA_OP(p, a,
         kgp_blend_op(desc->alphaOp));
      P_NV9097_SET_BLEND_PER_TARGET_ALPHA_SOURCE_COEFF(p, a,
         kgp_blend_factor(desc->srcAlpha));
      P_NV9097_SET_BLEND_PER_TARGET_ALPHA_DEST_COEFF(p, a,
         kgp_blend_factor(desc->dstAlpha));
   }
}

/* Ancestor: the write-mask group of nvk_flush_cb_state. The macro takes
 * four bits per target, in Vulkan's order, which is KGP_COLOR_WRITE's. A
 * target's mask is the blend state's, narrowed by the pipeline's for it.
 */
static void
kgp_push_write_mask(struct nv_push *p, const struct KgpBlendState_T *state,
                    const KgpRasterDesc *raster, uint32_t color_count)
{
   const uint32_t all = state ? state->write_mask : 0xf;
   uint32_t write_mask = 0;
   for (uint8_t a = 0; a < color_count; a++) {
      write_mask |= (all & kgp_color_write_bits(raster->colorWrite[a]))
                    << (4 * a);
   }
   P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_SET_WRITE_MASK));
   P_INLINE_DATA(p, color_count);
   P_INLINE_DATA(p, write_mask);
}

static void
kgp_push_blend_constant(struct nv_push *p,
                        const struct KgpBlendState_T *state)
{
   static const float zero[4] = { 0 };
   const float *c = state ? state->desc.constant : zero;

   P_MTHD(p, NV9097, SET_BLEND_CONST_RED);
   P_NV9097_SET_BLEND_CONST_RED(p,     fui(c[0]));
   P_NV9097_SET_BLEND_CONST_GREEN(p,   fui(c[1]));
   P_NV9097_SET_BLEND_CONST_BLUE(p,    fui(c[2]));
   P_NV9097_SET_BLEND_CONST_ALPHA(p,   fui(c[3]));
}
#define KGP_BLEND_TAIL_PUSH_DW 8    /* the write mask and the constant */

KGP_API KgpBlendState
kgpCreateBlendState(KgpDevice dev, const KgpBlendDesc *desc)
{
   if (desc == NULL ||
       (unsigned)desc->srcColor > KGP_BLEND_INV_CONSTANT_ALPHA ||
       (unsigned)desc->dstColor > KGP_BLEND_INV_CONSTANT_ALPHA ||
       (unsigned)desc->srcAlpha > KGP_BLEND_INV_CONSTANT_ALPHA ||
       (unsigned)desc->dstAlpha > KGP_BLEND_INV_CONSTANT_ALPHA ||
       (unsigned)desc->colorOp > KGP_BLEND_OP_MAX ||
       (unsigned)desc->alphaOp > KGP_BLEND_OP_MAX ||
       desc->colorWrite > KGP_COLOR_WRITE_NONE ||
       !isfinite(desc->constant[0]) || !isfinite(desc->constant[1]) ||
       !isfinite(desc->constant[2]) || !isfinite(desc->constant[3])) {
      kgp_err("kgpCreateBlendState: bad desc");
      return NULL;
   }

   struct KgpBlendState_T *state = calloc(1, sizeof(*state));
   if (state == NULL)
      return NULL;

   state->desc = *desc;
   state->enable = !(desc->srcColor == KGP_BLEND_ONE &&
                     desc->dstColor == KGP_BLEND_ZERO &&
                     desc->colorOp == KGP_BLEND_OP_ADD &&
                     desc->srcAlpha == KGP_BLEND_ONE &&
                     desc->dstAlpha == KGP_BLEND_ZERO &&
                     desc->alphaOp == KGP_BLEND_OP_ADD);
   state->write_mask = kgp_color_write_bits(desc->colorWrite);
   return state;
}

KGP_API void
kgpDestroyBlendState(KgpDevice dev, KgpBlendState state)
{
   free(state);
}

/* Draws ------------------------------------------------------------------- */

/* Everything fixed-function a draw depends on, then the pipeline's shader
 * block: what NVK emits at the first draw of a command buffer, group by
 * group in its order, with KoreGP's much smaller state behind it (the
 * pipeline's raster desc, the render pass's extent and the two state
 * objects; no vertex input).
 * Ancestors: nvk_flush_{vi,ia,vp,dr,rs,fsr,ms,ds,cb}_state.
 *
 * ponytail: emitted method by method on each pipeline or render pass
 * change. Everything but the viewport is known at pipeline creation; build
 * it into the pipeline's block (the proposal's "state snippets") when
 * pipeline switches show up in a profile.
 */
static void
kgp_cb_flush_gfx_state(struct KgpCommandBuffer_T *cb)
{
   const struct KgpPipeline_T *pipeline = cb->pipeline;
   const KgpRasterDesc *raster = &pipeline->raster;
   const uint32_t color_count = cb->rp_color_count;

   struct nv_push *p = kgp_cb_push(cb, 320 + 10 * color_count);

   /* vi: no vertex buffers, ever. */
   P_1INC(p, NVB197, CALL_MME_MACRO(NVK_MME_SET_VB_ENABLES));
   P_INLINE_DATA(p, 0);

   P_MTHD(p, NV9097, SET_VERTEX_ATTRIBUTE_A(0));
   for (uint32_t a = 0; a < 32; a++) {
      P_NV9097_SET_VERTEX_ATTRIBUTE_A(p, a, {
         .source                 = SOURCE_INACTIVE,
         .component_bit_widths   = COMPONENT_BIT_WIDTHS_R32_G32_B32_A32,
         .numerical_type         = NUMERICAL_TYPE_NUM_FLOAT,
      });
   }

   /* ia */
   P_MTHD(p, NV9097, SET_PRIMITIVE_TOPOLOGY);
   /* A mesh shader declares its own, but this state takes precedence:
    * point list, like NVK and the proprietary driver (nvk_flush_ia_state).
    */
   P_INLINE_DATA(p, pipeline->mesh ?
                    NV9097_SET_PRIMITIVE_TOPOLOGY_V_POINTLIST :
                    raster->topology == KGP_TOPOLOGY_TRIANGLE_STRIP ?
                    NV9097_SET_PRIMITIVE_TOPOLOGY_V_TRIANGLESTRIP :
                    NV9097_SET_PRIMITIVE_TOPOLOGY_V_TRIANGLELIST);
   P_IMMD(p, NV9097, SET_DA_PRIMITIVE_RESTART, ENABLE_FALSE);
   /* Input-assembly state a mesh pipeline does not have, so NVK leaves it. */
   if (!pipeline->mesh)
      P_IMMD(p, NV9097, SET_DA_PRIMITIVE_RESTART_INDEX, 0xffffffff);

   /* vp: one viewport and one scissor, the whole render pass, depth 0..1.
    * Ancestors: nvk_emit_viewport, nvk_emit_scissor.
    */
   const float w = cb->rp_width, h = cb->rp_height;
   P_MTHD(p, NV9097, SET_VIEWPORT_SCALE_X(0));
   P_NV9097_SET_VIEWPORT_SCALE_X(p, 0, fui(0.5f * w));
   P_NV9097_SET_VIEWPORT_SCALE_Y(p, 0, fui(0.5f * h));
   P_NV9097_SET_VIEWPORT_SCALE_Z(p, 0, fui(1.0f));
   P_NV9097_SET_VIEWPORT_OFFSET_X(p, 0, fui(0.5f * w));
   P_NV9097_SET_VIEWPORT_OFFSET_Y(p, 0, fui(0.5f * h));
   P_NV9097_SET_VIEWPORT_OFFSET_Z(p, 0, fui(0.0f));

   P_MTHD(p, NV9097, SET_VIEWPORT_CLIP_HORIZONTAL(0));
   P_NV9097_SET_VIEWPORT_CLIP_HORIZONTAL(p, 0, {
      .x0      = 0,
      .width   = cb->rp_width,
   });
   P_NV9097_SET_VIEWPORT_CLIP_VERTICAL(p, 0, {
      .y0      = 0,
      .height  = cb->rp_height,
   });
   P_NV9097_SET_VIEWPORT_CLIP_MIN_Z(p, 0, fui(0.0f));
   P_NV9097_SET_VIEWPORT_CLIP_MAX_Z(p, 0, fui(1.0f));

   P_IMMD(p, NVB197, SET_VIEWPORT_COORDINATE_SWIZZLE(0), {
      .x = X_POS_X,
      .y = Y_POS_Y,
      .z = Z_POS_Z,
      .w = W_POS_W,
   });

   P_IMMD(p, NV9097, SET_VIEWPORT_Z_CLIP, RANGE_ZERO_TO_POSITIVE_W);

   for (unsigned i = 1; i < 16; i++)
      P_IMMD(p, NV9097, SET_SCISSOR_ENABLE(i), V_FALSE);

   P_MTHD(p, NV9097, SET_SCISSOR_ENABLE(0));
   P_NV9097_SET_SCISSOR_ENABLE(p, 0, V_TRUE);
   P_NV9097_SET_SCISSOR_HORIZONTAL(p, 0, {
      .xmin = 0,
      .xmax = cb->rp_width,
   });
   P_NV9097_SET_SCISSOR_VERTICAL(p, 0, {
      .ymin = 0,
      .ymax = cb->rp_height,
   });

   /* dr: no discard rectangles. */
   P_IMMD(p, NV9097, SET_MME_SHADOW_SCRATCH(NVK_MME_SCRATCH_WINDOW_CLIP_ENABLED),
          0);
   P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_UPDATE_WINDOW_CLIP));
   P_INLINE_DATA(p, 1);

   P_IMMD(p, NV9097, SET_WINDOW_CLIP_TYPE,
          NV9097_SET_WINDOW_CLIP_TYPE_V_INCLUSIVE);

   P_MTHD(p, NV9097, SET_WINDOW_CLIP_HORIZONTAL(0));
   for (unsigned i = 0; i < 8; i++) {
      P_NV9097_SET_WINDOW_CLIP_HORIZONTAL(p, i, {
         .xmin = 0,
         .xmax = 0,
      });
      P_NV9097_SET_WINDOW_CLIP_VERTICAL(p, i, {
         .ymin = 0,
         .ymax = 0,
      });
   }

   /* rs: filled triangles, depth clipped to 0..1, the depth-stencil
    * state's bias, default lines.
    */
   P_IMMD(p, NV9097, SET_RASTER_ENABLE, true);

   /* As nvk_flush_rs_state without depth clamp: the extra clamp of depth
    * to 0..1 is on unless the depth target is floating point.
    */
   const bool z_clamp_zero_one = cb->rp_depth_format != KGP_FORMAT_D32_FLOAT;
   P_IMMD(p, NVC397, SET_VIEWPORT_CLIP_CONTROL, {
      .min_z_zero_max_z_one = z_clamp_zero_one ? MIN_Z_ZERO_MAX_Z_ONE_TRUE
                                               : MIN_Z_ZERO_MAX_Z_ONE_FALSE,
      .z_clip_range = z_clamp_zero_one
                      ? Z_CLIP_RANGE_USE_FIELD_MIN_Z_ZERO_MAX_Z_ONE
                      : Z_CLIP_RANGE_MINUS_INF_PLUS_INF,
      .pixel_min_z = PIXEL_MIN_Z_CLAMP,
      .pixel_max_z = PIXEL_MAX_Z_CLAMP,
      .geometry_guardband = GEOMETRY_GUARDBAND_SCALE_256,
      .line_point_cull_guardband = LINE_POINT_CULL_GUARDBAND_SCALE_256,
      .geometry_clip = GEOMETRY_CLIP_FRUSTUM_XYZ_CLIP,
      .geometry_guardband_z = GEOMETRY_GUARDBAND_Z_SCALE_1,
   });

   P_MTHD(p, NV9097, SET_FRONT_POLYGON_MODE);
   P_NV9097_SET_FRONT_POLYGON_MODE(p, NV9097_SET_FRONT_POLYGON_MODE_V_FILL);
   P_NV9097_SET_BACK_POLYGON_MODE(p, NV9097_SET_FRONT_POLYGON_MODE_V_FILL);

   /* Counter-clockwise is the front face (kgp.h); the framebuffer's y
    * points down, as in NVK, which is why the hardware enum reads as it
    * does (see vk_to_nv9097_front_face).
    */
   P_IMMD(p, NV9097, OGL_SET_CULL, raster->cull != KGP_CULL_NONE);
   if (raster->cull != KGP_CULL_NONE) {
      P_IMMD(p, NV9097, OGL_SET_CULL_FACE,
             raster->cull == KGP_CULL_CCW ? NV9097_OGL_SET_CULL_FACE_V_FRONT
                                          : NV9097_OGL_SET_CULL_FACE_V_BACK);
   }
   P_IMMD(p, NV9097, OGL_SET_FRONT_FACE, NV9097_OGL_SET_FRONT_FACE_V_CCW);

   P_IMMD(p, NV9097, SET_PROVOKING_VERTEX,
          NV9097_SET_PROVOKING_VERTEX_V_FIRST);

   kgp_push_depth_bias(p, cb->ds_state);

   P_MTHD(p, NV9097, SET_LINE_WIDTH_FLOAT);
   P_NV9097_SET_LINE_WIDTH_FLOAT(p, fui(1.0f));
   P_NV9097_SET_ALIASED_LINE_WIDTH_FLOAT(p, fui(1.0f));

   P_IMMD(p, NV9097, SET_LINE_MULTISAMPLE_OVERRIDE, ENABLE_FALSE);
   P_IMMD(p, NV9097, SET_ANTI_ALIASED_LINE, ENABLE_FALSE);

   P_IMMD(p, NV9097, SET_LINE_STIPPLE, false);
   P_IMMD(p, NV9097, SET_LINE_STIPPLE_PARAMETERS, {
      .factor  = 0,
      .pattern = 0,
   });

   P_IMMD(p, NV9097, SET_RASTER_INPUT, 0);
   P_IMMD(p, NVB197, SET_CONSERVATIVE_RASTER, ENABLE_FALSE);

   /* fsr: shading-rate control off. Ancestor:
    * nvk_mme_shading_rate_control_enable(false).
    */
   P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_SET_SHADING_RATE_CONTROL));
   P_INLINE_DATA(p, nvk_mme_val_mask(0, 1 << 0));

   /* ms (ancestor: nvk_flush_ms_state): the standard locations of the
    * pass's sample count. They also go where NAK's pixel shaders look for
    * them (nak_const_offsets_turing_graphics): eight bytes at offset 28
    * of root table 0.
    */
   const uint8_t *locs = kgp_sample_locations[util_logbase2(cb->rp_samples)];
   uint32_t loc_words[2];
   memcpy(loc_words, locs, sizeof(loc_words));
   P_IMMD(p, NV9097, SET_ANTI_ALIAS_ALPHA_CONTROL, {
      .alpha_to_coverage = raster->alphaToCoverage,
      .alpha_to_one      = false,
   });

   P_1INC(p, NVC597, SET_ROOT_TABLE_SELECTOR);
   P_NVC597_SET_ROOT_TABLE_SELECTOR(p, {
      .root_table = 0,
      .offset = 28,
   });
   P_INLINE_ARRAY(p, loc_words, 2);

   P_MTHD(p, NVB197, SET_ANTI_ALIAS_SAMPLE_POSITIONS(0));
   for (uint32_t i = 0; i < 4; i++) {
      /* Sixteen slots: the samples of four pixels, or of one repeated. */
      const uint8_t *l[4];
      for (uint32_t j = 0; j < 4; j++)
         l[j] = &locs[(i * 4 + j) % cb->rp_samples];
      P_NVB197_SET_ANTI_ALIAS_SAMPLE_POSITIONS(p, i, {
         .x0 = *l[0] & 0xf, .y0 = *l[0] >> 4,
         .x1 = *l[1] & 0xf, .y1 = *l[1] >> 4,
         .x2 = *l[2] & 0xf, .y2 = *l[2] >> 4,
         .x3 = *l[3] & 0xf, .y3 = *l[3] >> 4,
      });
   }

   P_MTHD(p, NV9097, SET_SAMPLE_MASK_X0_Y0);
   P_NV9097_SET_SAMPLE_MASK_X0_Y0(p, 0xffff);
   P_NV9097_SET_SAMPLE_MASK_X1_Y0(p, 0xffff);
   P_NV9097_SET_SAMPLE_MASK_X0_Y1(p, 0xffff);
   P_NV9097_SET_SAMPLE_MASK_X1_Y1(p, 0xffff);

   /* ds: the state object's words, or all off. */
   kgp_push_ds_state(p, cb->ds_state);

   /* cb: no logic op, and the blend state's equation, write mask and
    * constant.
    */
   P_IMMD(p, NV9097, SET_LOGIC_OP, false);
   P_IMMD(p, NV9097, SET_LOGIC_OP_FUNC, NV9097_SET_LOGIC_OP_FUNC_V_CLEAR);

   kgp_push_blend_state(p, cb->blend_state, color_count);

   kgp_push_write_mask(p, cb->blend_state, raster, color_count);

   P_IMMD(p, NV9097, SET_CT_SELECT, {
      .target_count = color_count,
      .target0 = 0,
      .target1 = 1,
      .target2 = 2,
      .target3 = 3,
      .target4 = 4,
      .target5 = 5,
      .target6 = 6,
      .target7 = 7,
   });

   kgp_push_blend_constant(p, cb->blend_state);

   /* The shaders, and their constant data. */
   p = kgp_cb_push(cb, pipeline->gfx_push_dw);
   nv_push_raw(p, pipeline->gfx_push, pipeline->gfx_push_dw);

   cb->gfx_dirty = false;
   cb->ds_dirty = false;
   cb->blend_dirty = false;
}

/* Everything of a draw but the draw itself: checks, the two roots, the
 * index buffer, and the pipeline's state if it is due. `index_bytes` is how
 * much of the index buffer the hardware may read.
 * Ancestors: nvk_cmd_flush_gfx_root_desc for the roots (delta D10) and
 * nvk_CmdBindIndexBuffer3KHR.
 */
static bool
kgp_cb_draw_setup(struct KgpCommandBuffer_T *cb, void *vsData, void *psData,
                  void *indicesGpu, uint32_t index_bytes, bool mesh)
{
   const struct KgpPipeline_T *pipeline = cb->pipeline;

   if (unlikely(!cb->rp_active || pipeline == NULL || !pipeline->graphics ||
                pipeline->mesh != mesh ||
                pipeline->raster.colorTargetCount != cb->rp_color_count)) {
      kgp_err("draw: needs an open render pass and a %s pipeline with "
              "its number of colour targets", mesh ? "mesh" : "graphics");
      cb->failed = true;
      return false;
   }
   if (cb->gfx_dirty) {
      for (uint32_t i = 0; i < cb->rp_color_count; i++) {
         if (unlikely(pipeline->raster.colorFormats[i] != cb->rp_formats[i])) {
            kgp_err("draw: colour target %u is not in the format the "
                    "pipeline was created for", i);
            cb->failed = true;
            return false;
         }
      }
      if (unlikely(pipeline->raster.depthFormat != cb->rp_depth_format)) {
         kgp_err("draw: the render pass's depth target is not what the "
                 "pipeline was created for (format, or none)");
         cb->failed = true;
         return false;
      }
   }
   if (unlikely(cb->gfx_dirty &&
                MAX2(pipeline->raster.sampleCount, 1) != cb->rp_samples)) {
      kgp_err("draw: the render pass's targets have %u samples and the "
              "pipeline was created for %u", cb->rp_samples,
              MAX2(pipeline->raster.sampleCount, 1));
      cb->failed = true;
      return false;
   }
   if (unlikely((cb->gfx_dirty || cb->ds_dirty) && cb->ds_state != NULL &&
                ((cb->ds_state->needs_depth_target &&
                  cb->rp_depth_format == KGP_FORMAT_NONE) ||
                 (cb->ds_state->needs_stencil &&
                  cb->rp_depth_format != KGP_FORMAT_D24_UNORM_S8_UINT)))) {
      kgp_err("draw: the depth-stencil state uses depth or stencil and the "
              "render pass's depth target does not have it");
      cb->failed = true;
      return false;
   }
   const uint64_t index_addr = (uintptr_t)indicesGpu;
   if (unlikely(!mesh && (index_addr == 0 || index_addr % 4 != 0))) {
      kgp_err("draw: indices must be a 4-byte aligned GPU pointer to uint32");
      cb->failed = true;
      return false;
   }
   /* An indirect draw's index count is the GPU's to know. */
   const bool indices_known = index_bytes != (UINT32_MAX & ~3u);
   if (!kgp_check_ptr(cb, "draw: the vertex shader's root pointer", vsData,
                      1, KGP_ROOT_ALIGN_B) ||
       !kgp_check_ptr(cb, "draw: the pixel shader's root pointer", psData,
                      1, KGP_ROOT_ALIGN_B) ||
       (!mesh && !kgp_check_ptr(cb, "draw: indices", indicesGpu,
                                indices_known ? index_bytes : 4, 4)))
      return false;

   /* The two root pointers, inline, into the hardware root table that the
    * shaders read as c[31][0] and c[31][8]. Under validation the heap
    * counts go behind them, for the shaders' bounds checks.
    */
   const uint64_t roots[2] = { (uintptr_t)vsData, (uintptr_t)psData };
   static_assert(KGP_GFX_ROOT_HEAP_COUNTS_OFFSET_B == sizeof(roots),
                 "the heap counts follow the root pointers");

   struct nv_push *p = kgp_cb_push(cb, 13);

   P_1INC(p, NVC597, SET_ROOT_TABLE_SELECTOR);
   P_NVC597_SET_ROOT_TABLE_SELECTOR(p, {
      .root_table = KGP_GFX_ROOT_TABLE,
      .offset = 0,
   });
   P_INLINE_ARRAY(p, (const uint32_t *)roots, 4);
   if (unlikely(cb->validate))
      P_INLINE_ARRAY(p, cb->heap_counts, 2);

   /* The index buffer, unless this command buffer's last bind covers it:
    * draws from one buffer bind it once, as vkCmdBindIndexBuffer would.
    * The size is only the hardware's read limit, so a larger one from an
    * earlier draw stands.
    */
   if (!mesh && (cb->ib_addr != index_addr || index_bytes > cb->ib_bytes)) {
      P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_BIND_IB));
      P_INLINE_DATA(p, index_addr >> 32);
      P_INLINE_DATA(p, index_addr);
      P_INLINE_DATA(p, index_bytes);
      P_INLINE_DATA(p, 1 /* VK_INDEX_TYPE_UINT32, which the macro decodes */);
      cb->ib_addr = index_addr;
      cb->ib_bytes = index_bytes;
   }

   if (cb->gfx_dirty) {
      kgp_cb_flush_gfx_state(cb);
   } else if (cb->ds_dirty || cb->blend_dirty) {
      /* A state object changed between draws of one pipeline: only its
       * methods, as NVK sends only the dynamic state that changed.
       */
      p = kgp_cb_push(cb, KGP_DEPTH_BIAS_PUSH_DW + KGP_DS_PUSH_DW +
                          9 * cb->rp_color_count + KGP_BLEND_TAIL_PUSH_DW);
      if (cb->ds_dirty) {
         kgp_push_depth_bias(p, cb->ds_state);
         kgp_push_ds_state(p, cb->ds_state);
      }
      if (cb->blend_dirty) {
         kgp_push_blend_state(p, cb->blend_state, cb->rp_color_count);
         kgp_push_write_mask(p, cb->blend_state, &pipeline->raster,
                             cb->rp_color_count);
         kgp_push_blend_constant(p, cb->blend_state);
      }
      cb->ds_dirty = false;
      cb->blend_dirty = false;
   }
   return true;
}

/* Ancestor: nvk_CmdDrawIndexed. The draw is NVK's macro, and
 * hardware-instanced.
 */
KGP_API void
kgpDrawIndexedInstanced(KgpCommandBuffer cb, void *vsData, void *psData,
                        void *indicesGpu, uint32_t indexCount,
                        uint32_t instanceCount)
{
   if (unlikely(indexCount > UINT32_MAX / 4)) {
      kgp_err("kgpDrawIndexedInstanced: at most 2^30 indices");
      cb->failed = true;
      return;
   }
   if (unlikely(indexCount == 0 || instanceCount == 0))
      return;
   if (!kgp_cb_draw_setup(cb, vsData, psData, indicesGpu, indexCount * 4,
                          false))
      return;

   struct nv_push *p = kgp_cb_push(cb, 7);
   P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_DRAW_INDEXED));
   P_INLINE_DATA(p, 0 /* draw_index */);
   P_INLINE_DATA(p, indexCount);
   P_INLINE_DATA(p, instanceCount);
   P_INLINE_DATA(p, 0 /* firstIndex */);
   P_INLINE_DATA(p, 0 /* vertexOffset */);
   P_INLINE_DATA(p, 0 /* firstInstance */);
}

/* Ancestor: nvk_CmdDrawIndexedIndirect2KHR with one draw. NVK's macro reads
 * the five argument words on the GPU's macro processor, so arguments an
 * earlier dispatch or draw wrote need KGP_HAZARD_DRAW_ARGUMENTS first.
 * The index buffer's extent is not known here: the hardware's limit is set
 * as high as it goes and the caller answers for the indices.
 */
KGP_API void
kgpDrawIndexedInstancedIndirect(KgpCommandBuffer cb, void *vsData,
                                void *psData, void *indicesGpu, void *argsGpu)
{
   const uint64_t args_addr = (uintptr_t)argsGpu;
   if (unlikely(args_addr == 0 || args_addr % 4 != 0)) {
      kgp_err("kgpDrawIndexedInstancedIndirect: argsGpu must be a 4-byte "
              "aligned GPU pointer");
      cb->failed = true;
      return;
   }
   if (!kgp_check_ptr(cb, "kgpDrawIndexedInstancedIndirect: argsGpu", argsGpu,
                      sizeof(KgpDrawIndexedArgs), 4))
      return;
   if (!kgp_cb_draw_setup(cb, vsData, psData, indicesGpu, UINT32_MAX & ~3u,
                          false))
      return;

   const uint64_t stride = sizeof(KgpDrawIndexedArgs);
   struct nv_push *p = kgp_cb_push(cb, 6);
   P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_DRAW_INDEXED_INDIRECT));
   P_INLINE_DATA(p, args_addr >> 32);
   P_INLINE_DATA(p, args_addr);
   P_INLINE_DATA(p, 1 /* drawCount */);
   P_INLINE_DATA(p, stride >> 32);
   P_INLINE_DATA(p, stride);
}

/* No ancestor in NVK: Vulkan has no per-draw root data. The whole draw is
 * one call of KoreGP's own macro (kgp_mme.c), which reads the count and the
 * arguments on the GPU and steps the two root pointers itself. The roots
 * kgp_cb_draw_setup writes are draw 0's; the macro writes them again.
 */
KGP_API void
kgpDrawIndexedInstancedIndirectMulti(KgpCommandBuffer cb,
                                     void *vsData, uint32_t vsStride,
                                     void *psData, uint32_t psStride,
                                     void *indicesGpu, void *argsGpu,
                                     void *countGpu, uint32_t maxDrawCount)
{
   const uint64_t args_addr = (uintptr_t)argsGpu;
   const uint64_t count_addr = (uintptr_t)countGpu;
   const uint64_t vs_addr = (uintptr_t)vsData;
   const uint64_t ps_addr = (uintptr_t)psData;
   if (unlikely(args_addr == 0 || args_addr % 4 != 0 ||
                count_addr == 0 || count_addr % 4 != 0)) {
      kgp_err("kgpDrawIndexedInstancedIndirectMulti: argsGpu and countGpu "
              "must be 4-byte aligned GPU pointers");
      cb->failed = true;
      return;
   }
   /* How many draws there are, the GPU decides: the first one's arguments
    * (and, in kgp_cb_draw_setup, roots) are all that can be checked here.
    */
   if (!kgp_check_ptr(cb, "kgpDrawIndexedInstancedIndirectMulti: argsGpu",
                      argsGpu, sizeof(KgpDrawIndexedArgs), 4) ||
       !kgp_check_ptr(cb, "kgpDrawIndexedInstancedIndirectMulti: countGpu",
                      countGpu, sizeof(uint32_t), 4))
      return;
   if (maxDrawCount == 0)
      return;
   if (!kgp_cb_draw_setup(cb, vsData, psData, indicesGpu, UINT32_MAX & ~3u,
                          false))
      return;

   struct nv_push *p = kgp_cb_push(cb, 12);
   P_1INC(p, NV9097, CALL_MME_MACRO(KGP_MME_DRAW_INDEXED_MULTI));
   P_INLINE_DATA(p, args_addr >> 32);
   P_INLINE_DATA(p, args_addr);
   P_INLINE_DATA(p, count_addr >> 32);
   P_INLINE_DATA(p, count_addr);
   P_INLINE_DATA(p, maxDrawCount);
   P_INLINE_DATA(p, vs_addr >> 32);
   P_INLINE_DATA(p, vs_addr);
   P_INLINE_DATA(p, vsStride);
   P_INLINE_DATA(p, ps_addr >> 32);
   P_INLINE_DATA(p, ps_addr);
   P_INLINE_DATA(p, psStride);
}

/* Mesh draws ---------------------------------------------------------------- */

/* Ancestor: nvk_CmdDrawMeshTasksEXT. NVK's macro puts the group counts
 * where the shader reads gl_NumWorkGroups and draws x * y * z points, one
 * workgroup each.
 */
KGP_API void
kgpDrawMeshlets(KgpCommandBuffer cb, void *msData, void *psData,
                KgpUvec3 groups)
{
   /* NVK's maxMeshWorkGroupCount and maxMeshWorkGroupTotalCount */
   if (unlikely(groups.y > UINT16_MAX || groups.z > UINT16_MAX ||
                (uint64_t)groups.x * groups.y * groups.z > (1u << 22))) {
      kgp_err("kgpDrawMeshlets: at most 2^22 workgroups, 65535 in Y and Z");
      cb->failed = true;
      return;
   }
   if (unlikely(groups.x == 0 || groups.y == 0 || groups.z == 0))
      return;
   if (!kgp_cb_draw_setup(cb, msData, psData, NULL, 0, true))
      return;

   struct nv_push *p = kgp_cb_push(cb, 4);
   P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_DRAW_MESH));
   P_INLINE_DATA(p, groups.x);
   P_INLINE_DATA(p, groups.y);
   P_INLINE_DATA(p, groups.z);
}

/* Ancestor: nvk_CmdDrawMeshTasksIndirect2EXT with one draw. */
KGP_API void
kgpDrawMeshletsIndirect(KgpCommandBuffer cb, void *msData, void *psData,
                        void *groupsGpu)
{
   const uint64_t addr = (uintptr_t)groupsGpu;
   if (unlikely(addr == 0 || addr % 4 != 0)) {
      kgp_err("kgpDrawMeshletsIndirect: groupsGpu must be a 4-byte aligned "
              "GPU pointer");
      cb->failed = true;
      return;
   }
   if (!kgp_check_ptr(cb, "kgpDrawMeshletsIndirect: the group counts",
                      groupsGpu, sizeof(KgpUvec3), 4) ||
       !kgp_cb_draw_setup(cb, msData, psData, NULL, 0, true))
      return;

   struct nv_push *p = kgp_cb_push(cb, 6);
   P_1INC(p, NV9097, CALL_MME_MACRO(NVK_MME_DRAW_MESH_INDIRECT));
   P_INLINE_DATA(p, addr >> 32);
   P_INLINE_DATA(p, addr);
   P_INLINE_DATA(p, 1 /* draw count */);
   P_INLINE_DATA(p, 0);
   P_INLINE_DATA(p, sizeof(KgpUvec3) /* stride */);
}
