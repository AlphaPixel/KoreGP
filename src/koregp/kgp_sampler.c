/*
 * Sampler descriptors: the 32-byte sampler entry (TSC) the hardware reads
 * from the sampler heap, written into caller memory.
 *
 * Ancestor: nvk_sampler.c. The packing helpers and nvk_sampler_get_header
 * are copied (the original is static) and kept in NVK's terms, a
 * vk_sampler_state in and eight words out, so the body stays diffable;
 * kgpSamplerDescriptor only translates the KoreGP desc into that state.
 * Cut: VkSampler objects, the sampler table, YCbCr, the border-colour
 * swizzle and custom-border-colour formats, cube-map and compare state
 * (KoreGP exposes none of them yet).
 *
 * SPDX-License-Identifier: MIT
 */
#include "kgp_private.h"

#include "util/bitpack_helpers.h"
#include "util/format_srgb.h"
#include "vk_sampler.h"

#include "cl9097tex.h"
#include "cla097.h"
#include "cla097tex.h"
#include "clb197.h"
#include "clb197tex.h"
#include "drf.h"

ALWAYS_INLINE static void
__set_u32(uint32_t *o, uint32_t v, unsigned lo, unsigned hi)
{
   assert(lo <= hi && hi < 32);
   *o |= util_bitpack_uint(v, lo % 32, hi % 32);
}

#define FIXED_FRAC_BITS 8

ALWAYS_INLINE static void
__set_ufixed(uint32_t *o, float v, unsigned lo, unsigned hi)
{
   assert(lo <= hi && hi < 32);
   *o |= util_bitpack_ufixed_clamp(v, lo % 32, hi % 32, FIXED_FRAC_BITS);
}

ALWAYS_INLINE static void
__set_sfixed(uint32_t *o, float v, unsigned lo, unsigned hi)
{
   assert(lo <= hi && hi < 32);
   *o |= util_bitpack_sfixed_clamp(v, lo % 32, hi % 32, FIXED_FRAC_BITS);
}

ALWAYS_INLINE static void
__set_bool(uint32_t *o, bool b, unsigned lo, unsigned hi)
{
   assert(lo == hi && hi < 32);
   *o |= util_bitpack_uint(b, lo % 32, hi % 32);
}

#define MW(x) x

#define SAMP_SET_U(o, NV, i, FIELD, val) \
   __set_u32(&o.bits[i], (val), DRF_LO(NV##_TEXSAMP##i##_##FIELD),\
                             DRF_HI(NV##_TEXSAMP##i##_##FIELD))

#define SAMP_SET_UF(o, NV, i, FIELD, val) \
   __set_ufixed(&o.bits[i], (val), DRF_LO(NV##_TEXSAMP##i##_##FIELD),\
                                DRF_HI(NV##_TEXSAMP##i##_##FIELD))

#define SAMP_SET_SF(o, NV, i, FIELD, val) \
   __set_sfixed(&o.bits[i], (val), DRF_LO(NV##_TEXSAMP##i##_##FIELD),\
                                DRF_HI(NV##_TEXSAMP##i##_##FIELD))

#define SAMP_SET_B(o, NV, i, FIELD, b) \
   __set_bool(&o.bits[i], (b), DRF_LO(NV##_TEXSAMP##i##_##FIELD),\
                            DRF_HI(NV##_TEXSAMP##i##_##FIELD))

#define SAMP_SET_E(o, NV, i, FIELD, E) \
   SAMP_SET_U(o, NV, i, FIELD, NV##_TEXSAMP##i##_##FIELD##_##E)

static inline uint32_t
vk_to_9097_address_mode(VkSamplerAddressMode addr_mode)
{
#define MODE(VK, NV) \
   [VK_SAMPLER_ADDRESS_MODE_##VK] = NV9097_TEXSAMP0_ADDRESS_U_##NV
   static const uint8_t vk_to_9097[] = {
      MODE(REPEAT,               WRAP),
      MODE(MIRRORED_REPEAT,      MIRROR),
      MODE(CLAMP_TO_EDGE,        CLAMP_TO_EDGE),
      MODE(CLAMP_TO_BORDER,      BORDER),
      MODE(MIRROR_CLAMP_TO_EDGE, MIRROR_ONCE_CLAMP_TO_EDGE),
   };
#undef MODE

   assert(addr_mode < ARRAY_SIZE(vk_to_9097));
   return vk_to_9097[addr_mode];
}

static uint32_t
vk_to_9097_max_anisotropy(float max_anisotropy)
{
   if (max_anisotropy >= 16)
      return NV9097_TEXSAMP0_MAX_ANISOTROPY_ANISO_16_TO_1;

   if (max_anisotropy >= 12)
      return NV9097_TEXSAMP0_MAX_ANISOTROPY_ANISO_12_TO_1;

   uint32_t aniso_u32 = MAX2(0.0f, max_anisotropy);
   return aniso_u32 >> 1;
}

static uint32_t
vk_to_9097_trilin_opt(float max_anisotropy)
{
   /* No idea if we want this but matching nouveau */
   if (max_anisotropy >= 12)
      return 0;

   if (max_anisotropy >= 4)
      return 6;

   if (max_anisotropy >= 2)
      return 4;

   return 0;
}

/* Ancestor: nvk_sampler_get_header, Maxwell B and later. */
static KgpDescriptor256
kgp_sampler_get_header(const struct vk_sampler_state *state)
{
   KgpDescriptor256 samp = {};

   SAMP_SET_U(samp, NV9097, 0, ADDRESS_U,
              vk_to_9097_address_mode(state->address_mode_u));
   SAMP_SET_U(samp, NV9097, 0, ADDRESS_V,
              vk_to_9097_address_mode(state->address_mode_v));
   SAMP_SET_U(samp, NV9097, 0, ADDRESS_P,
              vk_to_9097_address_mode(state->address_mode_w));

   SAMP_SET_B(samp, NV9097, 0, S_R_G_B_CONVERSION, true);
   SAMP_SET_E(samp, NV9097, 0, FONT_FILTER_WIDTH, SIZE_2);
   SAMP_SET_E(samp, NV9097, 0, FONT_FILTER_HEIGHT, SIZE_2);

   if (state->anisotropy_enable) {
      SAMP_SET_U(samp, NV9097, 0, MAX_ANISOTROPY,
                 vk_to_9097_max_anisotropy(state->max_anisotropy));
   }

   switch (state->mag_filter) {
   case VK_FILTER_NEAREST:
      SAMP_SET_E(samp, NV9097, 1, MAG_FILTER, MAG_POINT);
      break;
   case VK_FILTER_LINEAR:
      SAMP_SET_E(samp, NV9097, 1, MAG_FILTER, MAG_LINEAR);
      break;
   default:
      UNREACHABLE("Invalid filter");
   }

   switch (state->min_filter) {
   case VK_FILTER_NEAREST:
      SAMP_SET_E(samp, NV9097, 1, MIN_FILTER, MIN_POINT);
      break;
   case VK_FILTER_LINEAR:
      if (state->anisotropy_enable)
         SAMP_SET_E(samp, NV9097, 1, MIN_FILTER, MIN_ANISO);
      else
         SAMP_SET_E(samp, NV9097, 1, MIN_FILTER, MIN_LINEAR);
      break;
   default:
      UNREACHABLE("Invalid filter");
   }

   switch (state->mipmap_mode) {
   case VK_SAMPLER_MIPMAP_MODE_NEAREST:
      SAMP_SET_E(samp, NV9097, 1, MIP_FILTER, MIP_POINT);
      break;
   case VK_SAMPLER_MIPMAP_MODE_LINEAR:
      SAMP_SET_E(samp, NV9097, 1, MIP_FILTER, MIP_LINEAR);
      break;
   default:
      UNREACHABLE("Invalid mipmap mode");
   }

   SAMP_SET_E(samp, NVA097, 1, CUBEMAP_INTERFACE_FILTERING, AUTO_SPAN_SEAM);
   SAMP_SET_E(samp, NVB197, 1, REDUCTION_FILTER, RED_NONE);

   SAMP_SET_SF(samp, NV9097, 1, MIP_LOD_BIAS, state->mip_lod_bias);

   SAMP_SET_E(samp, NVA097, 1, FLOAT_COORD_NORMALIZATION,
                               USE_HEADER_SETTING);
   SAMP_SET_U(samp, NV9097, 1, TRILIN_OPT,
              vk_to_9097_trilin_opt(state->max_anisotropy));

   SAMP_SET_UF(samp, NV9097, 2, MIN_LOD_CLAMP, state->min_lod);
   SAMP_SET_UF(samp, NV9097, 2, MAX_LOD_CLAMP, state->max_lod);

   const VkClearColorValue bc = state->border_color_value;

   /* No swizzle and a float border colour: there's nothing to do but
    * convert the border color to sRGB.
    */
   uint8_t bc_srgb[3];
   for (unsigned i = 0; i < 3; i++)
      bc_srgb[i] = util_format_linear_float_to_srgb_8unorm(bc.float32[i]);

   SAMP_SET_U(samp, NV9097, 2, S_R_G_B_BORDER_COLOR_R, bc_srgb[0]);
   SAMP_SET_U(samp, NV9097, 3, S_R_G_B_BORDER_COLOR_G, bc_srgb[1]);
   SAMP_SET_U(samp, NV9097, 3, S_R_G_B_BORDER_COLOR_B, bc_srgb[2]);

   SAMP_SET_U(samp, NV9097, 4, BORDER_COLOR_R, bc.uint32[0]);
   SAMP_SET_U(samp, NV9097, 5, BORDER_COLOR_G, bc.uint32[1]);
   SAMP_SET_U(samp, NV9097, 6, BORDER_COLOR_B, bc.uint32[2]);
   SAMP_SET_U(samp, NV9097, 7, BORDER_COLOR_A, bc.uint32[3]);

   return samp;
}

/* The KoreGP enums are numerically the Vulkan ones the ancestor switches
 * on, so the translation below is a range check and a cast.
 */
static_assert((int)KGP_FILTER_NEAREST == VK_FILTER_NEAREST &&
              (int)KGP_FILTER_LINEAR == VK_FILTER_LINEAR &&
              (int)KGP_FILTER_NEAREST == VK_SAMPLER_MIPMAP_MODE_NEAREST &&
              (int)KGP_FILTER_LINEAR == VK_SAMPLER_MIPMAP_MODE_LINEAR,
              "filter enums match");
static_assert(
   (int)KGP_ADDRESS_REPEAT == VK_SAMPLER_ADDRESS_MODE_REPEAT &&
   (int)KGP_ADDRESS_MIRRORED_REPEAT == VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT &&
   (int)KGP_ADDRESS_CLAMP_TO_EDGE == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE &&
   (int)KGP_ADDRESS_CLAMP_TO_BORDER == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
   "address enums match");

KGP_API bool
kgpSamplerDescriptor(KgpDevice dev, const KgpSamplerDesc *desc,
                     KgpDescriptor256 *out)
{
   if ((unsigned)desc->minFilter > KGP_FILTER_LINEAR ||
       (unsigned)desc->magFilter > KGP_FILTER_LINEAR ||
       (unsigned)desc->mipFilter > KGP_FILTER_LINEAR ||
       (unsigned)desc->addressU > KGP_ADDRESS_CLAMP_TO_BORDER ||
       (unsigned)desc->addressV > KGP_ADDRESS_CLAMP_TO_BORDER ||
       (unsigned)desc->addressW > KGP_ADDRESS_CLAMP_TO_BORDER ||
       !(desc->maxAnisotropy >= 0.0f && desc->maxAnisotropy <= 16.0f) ||
       !(desc->minLod >= 0.0f) || !(desc->maxLod >= desc->minLod)) {
      kgp_err("kgpSamplerDescriptor: invalid sampler description");
      return false;
   }

   const struct vk_sampler_state state = {
      .mag_filter = (VkFilter)desc->magFilter,
      .min_filter = (VkFilter)desc->minFilter,
      .mipmap_mode = (VkSamplerMipmapMode)desc->mipFilter,
      .address_mode_u = (VkSamplerAddressMode)desc->addressU,
      .address_mode_v = (VkSamplerAddressMode)desc->addressV,
      .address_mode_w = (VkSamplerAddressMode)desc->addressW,
      .mip_lod_bias = desc->mipLodBias,
      .anisotropy_enable = desc->maxAnisotropy > 1.0f,
      .max_anisotropy = desc->maxAnisotropy,
      .min_lod = desc->minLod,
      .max_lod = desc->maxLod,
      .border_color_value.float32 = {
         desc->borderColor[0], desc->borderColor[1],
         desc->borderColor[2], desc->borderColor[3],
      },
   };

   *out = kgp_sampler_get_header(&state);
   return true;
}
