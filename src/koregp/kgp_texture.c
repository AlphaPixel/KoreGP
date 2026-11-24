/*
 * Textures: a layout over memory the caller already owns, and the 256-bit
 * descriptor the hardware reads from the texture heap.
 *
 * NIL does the layout and packs the descriptor (the texture header, TIC);
 * it is called, not copied. Ancestors for how it is called:
 *   nvk_image.c       nvk_image_init                 (kgp_texture_layout)
 *   nvk_image_view.c  nvk_image_view_init_no_alloc   (kgpTextureViewDescriptor)
 *   nvk_host_copy.c   nvk_copy_memory_to_image       (kgpWriteTexture)
 * Cut: vk_image, planes, YCbCr, compression, sparse, modifiers, MSAA.
 *
 * Delta D4 (docs/plan.md N2): NVK maps tiled images through a VA range
 * tagged with a PTE kind; here the texture sits at the caller's address,
 * which is kind 0. The TU104 accepts that for colour (tests/kgp_heap.c on
 * hardware, 2026-11-21). KGP_TEXTURE_LINEAR is the layout NIL itself gives
 * kind 0.
 *
 * SPDX-License-Identifier: MIT
 */
#include "kgp_private.h"

#include "drm-uapi/drm_fourcc.h"
#include "util/format/u_format.h"

#include <stdlib.h>

static enum pipe_format
kgp_format_to_pipe(KGP_FORMAT format)
{
   switch (format) {
   case KGP_FORMAT_RGBA8_UNORM:  return PIPE_FORMAT_R8G8B8A8_UNORM;
   case KGP_FORMAT_BGRA8_UNORM:  return PIPE_FORMAT_B8G8R8A8_UNORM;
   case KGP_FORMAT_RGBA8_SRGB:   return PIPE_FORMAT_R8G8B8A8_SRGB;
   case KGP_FORMAT_R32_UINT:     return PIPE_FORMAT_R32_UINT;
   case KGP_FORMAT_R32_FLOAT:    return PIPE_FORMAT_R32_FLOAT;
   case KGP_FORMAT_RGBA16_FLOAT: return PIPE_FORMAT_R16G16B16A16_FLOAT;
   case KGP_FORMAT_RGBA32_FLOAT: return PIPE_FORMAT_R32G32B32A32_FLOAT;
   case KGP_FORMAT_D32_FLOAT:    return PIPE_FORMAT_Z32_FLOAT;
   case KGP_FORMAT_D24_UNORM_S8_UINT: return PIPE_FORMAT_Z24_UNORM_S8_UINT;
   default:                      return PIPE_FORMAT_NONE;
   }
}

/* NIL asserts on input it does not like; everything it would trip over is
 * checked here first so that a bad desc is an error, not an abort.
 */
static bool
kgp_texture_layout(struct KgpDevice_T *dev, const KgpTextureDesc *desc,
                   struct nil_image *image_out)
{
   const struct nv_device_info *info = kgp_dev_info(dev);
   const enum pipe_format p_format = kgp_format_to_pipe(desc->format);
   const uint32_t depth = MAX2(desc->depth, 1);
   const uint32_t layers = MAX2(desc->layerCount, 1);
   const uint32_t levels = MAX2(desc->mipCount, 1);
   const bool linear = desc->flags & KGP_TEXTURE_LINEAR;

   if (p_format == PIPE_FORMAT_NONE ||
       !nil_format_supports_texturing(info, p_format)) {
      kgp_err("texture: unsupported format %d", desc->format);
      return false;
   }
   if (desc->width == 0 || desc->height == 0 ||
       desc->width > 16384 || desc->height > 16384 || depth > 2048 ||
       layers > 2048 || (depth > 1 && layers > 1)) {
      kgp_err("texture: bad extent %ux%ux%u, %u layers",
              desc->width, desc->height, depth, layers);
      return false;
   }
   const uint32_t max_dim = MAX3(desc->width, desc->height, depth);
   if (levels > util_logbase2(max_dim) + 1) {
      kgp_err("texture: %u mip levels is more than the extent has", levels);
      return false;
   }
   if (util_format_is_depth_or_stencil(p_format) && (linear || depth > 1)) {
      kgp_err("texture: a depth format is tiled and not 3D");
      return false;
   }
   if (linear && (depth > 1 || layers > 1 || levels > 1)) {
      kgp_err("texture: KGP_TEXTURE_LINEAR is 2D, one level, no layers");
      return false;
   }
   if (desc->flags & ~KGP_TEXTURE_LINEAR) {
      kgp_err("texture: unknown flags 0x%x", desc->flags);
      return false;
   }

   const struct nil_image_init_info init = {
      .dim = depth > 1 ? NIL_IMAGE_DIM_3D : NIL_IMAGE_DIM_2D,
      .format = nil_format(p_format),
      .extent_px = {
         .width = desc->width,
         .height = desc->height,
         .depth = depth,
         .array_len = layers,
      },
      .levels = levels,
      .samples = 1,
      .usage = NIL_IMAGE_USAGE_UNCOMPRESSED_BIT |
               (linear ? NIL_IMAGE_USAGE_LINEAR_BIT : 0),
      /* Not 0: that is DRM_FORMAT_MOD_LINEAR and selects the linear layout */
      .modifier = DRM_FORMAT_MOD_INVALID,
   };
   if (!nil_image_init(info, image_out, &init)) {
      kgp_err("texture: NIL rejected the layout");
      return false;
   }
   return true;
}

KGP_API bool
kgpTextureSizeAlign(KgpDevice dev, const KgpTextureDesc *desc,
                    KgpSizeAlign *out)
{
   struct nil_image image;
   if (!kgp_texture_layout(dev, desc, &image))
      return false;

   out->size = image.size_B;
   out->align = image.align_B;
   return true;
}

/* Maps the texture's pages through a VA range tagged with the PTE kind NIL
 * chose for it and makes that the texture's address, which is what NVK
 * does for every tiled image. Ancestors: nvk_image_plane_alloc_va,
 * nvk_image_plane_bind.
 */
static bool
kgp_texture_map_kind(struct KgpDevice_T *dev, struct KgpTexture_T *tex,
                     struct nvkmd_mem *mem, uint64_t offset_B)
{
   const uint64_t range_B = align64(tex->image.size_B, 4096);
   if (mem == NULL || offset_B % 4096 != 0 ||
       offset_B + range_B > mem->size_B ||
       nvkmd_dev_alloc_va(dev->nvkmd, KGP_LOG_OBJ, 0, tex->image.pte_kind,
                          range_B, tex->image.align_B, 0,
                          &tex->alias_va) != VK_SUCCESS)
      return false;
   if (nvkmd_va_bind_mem(tex->alias_va, KGP_LOG_OBJ, 0, mem, offset_B,
                         range_B) != VK_SUCCESS) {
      nvkmd_va_free(tex->alias_va);
      tex->alias_va = NULL;
      return false;
   }
   tex->addr = tex->alias_va->addr;
   return true;
}

/* A texture over memory KoreGP did not allocate, with a layout it did not
 * compute: a swapchain image (kgp_present.c). Takes over the reference to
 * `mem`, also on failure.
 */
struct KgpTexture_T *
kgp_texture_over_mem(struct KgpDevice_T *dev, const struct nil_image *image,
                     KGP_FORMAT format, struct nvkmd_mem *mem,
                     uint64_t offset_B)
{
   struct KgpTexture_T *tex = calloc(1, sizeof(*tex));
   if (tex != NULL) {
      tex->dev = dev;
      tex->format = format;
      tex->image = *image;
      if (kgp_texture_map_kind(dev, tex, mem, offset_B)) {
         tex->owned_mem = mem;
         return tex;
      }
      kgp_err("could not map shared texture memory");
      free(tex);
   }
   nvkmd_mem_unref(mem);
   return NULL;
}

KGP_API KgpTexture
kgpCreateTexture(KgpDevice dev, const KgpTextureDesc *desc, void *gpuPtr)
{
   struct KgpTexture_T *tex = calloc(1, sizeof(*tex));
   if (tex == NULL)
      return NULL;

   tex->dev = dev;
   tex->addr = (uintptr_t)gpuPtr;
   tex->format = desc->format;
   if (!kgp_texture_layout(dev, desc, &tex->image)) {
      free(tex);
      return NULL;
   }

   /* The descriptor stores the address without its low 9 bits and with 48
    * bits in all (NIL asserts both); the layout's own alignment is larger.
    */
   if (tex->addr == 0 || tex->addr % tex->image.align_B != 0 ||
       tex->addr >> 48 != 0) {
      kgp_err("kgpCreateTexture: gpuPtr %p is not a GPU pointer aligned to "
              "the %u bytes kgpTextureSizeAlign asked for",
              gpuPtr, tex->image.align_B);
      free(tex);
      return NULL;
   }
   tex->mem_addr = tex->addr;
   if (unlikely(dev->debug & KGP_DEBUG_VALIDATE) &&
       !kgp_validate_range(dev, NULL, "kgpCreateTexture: gpuPtr", tex->addr,
                           tex->image.size_B, 1)) {
      free(tex);
      return NULL;
   }

   /* Delta D4, the one format measured to need its PTE kind: as a depth
    * target at an ordinary (kind 0) address, D24S8 raises "3D-Z KIND
    * Violation" and the channel dies; colour and D32_FLOAT work there
    * (docs/results-n1.md, fifth window). So it is mapped a second time,
    * under its kind.
    * ponytail: KGP_MEMORY_GPU only, and the texture must start on a page
    * of its allocation; mapped memory when someone needs to write stencil
    * from the CPU.
    */
   if (tex->format == KGP_FORMAT_D24_UNORM_S8_UINT) {
      uint64_t offset_B = 0;
      struct nvkmd_mem *mem = kgp_gpu_only_mem(dev, tex->addr, &offset_B);
      if (!kgp_texture_map_kind(dev, tex, mem, offset_B)) {
         kgp_err("kgpCreateTexture: a depth-stencil texture must lie in "
                 "KGP_MEMORY_GPU memory, page-aligned within its allocation "
                 "(%p)", gpuPtr);
         free(tex);
         return NULL;
      }
   }

   return tex;
}

KGP_API void
kgpDestroyTexture(KgpDevice dev, KgpTexture tex)
{
   if (tex != NULL && tex->alias_va != NULL)
      nvkmd_va_free(tex->alias_va);
   if (tex != NULL && tex->owned_mem != NULL)
      nvkmd_mem_unref(tex->owned_mem);
   free(tex);
}

KGP_API bool
kgpTextureViewDescriptor(KgpTexture tex, const KgpViewDesc *view_desc,
                         KgpDescriptor256 *out)
{
   const struct nil_image *image = &tex->image;
   static const KgpViewDesc whole = { 0 };
   const KgpViewDesc *v = view_desc ? view_desc : &whole;

   const uint32_t num_levels = v->mipCount ? v->mipCount
                                           : image->num_levels - v->baseMip;
   const uint32_t array_len = v->layerCount
      ? v->layerCount : image->extent_px.array_len - v->baseLayer;
   const enum pipe_format p_format =
      v->format ? kgp_format_to_pipe(v->format) : image->format.p_format;

   if (v->baseMip >= image->num_levels ||
       num_levels > image->num_levels - v->baseMip ||
       v->baseLayer >= image->extent_px.array_len ||
       array_len > image->extent_px.array_len - v->baseLayer) {
      kgp_err("kgpTextureViewDescriptor: view range is outside the texture");
      return false;
   }
   if (p_format == PIPE_FORMAT_NONE ||
       util_format_get_blocksize(p_format) !=
       util_format_get_blocksize(image->format.p_format)) {
      kgp_err("kgpTextureViewDescriptor: view format must have the texel "
              "size of the texture's format");
      return false;
   }

   const struct nil_view view = {
      .view_type = image->dim == NIL_IMAGE_DIM_3D ? NIL_VIEW_TYPE_3D :
                   image->extent_px.array_len > 1 ? NIL_VIEW_TYPE_2D_ARRAY :
                                                    NIL_VIEW_TYPE_2D,
      /* One descriptor serves sampling and storage for single-sampled
       * images; the access only matters for MSAA.
       */
      .access = NIL_VIEW_ACCESS_TEXTURE,
      .format = nil_format(p_format),
      .base_level = v->baseMip,
      .num_levels = num_levels,
      .base_array_layer = v->baseLayer,
      .array_len = array_len,
      .swizzle = {
         PIPE_SWIZZLE_X, PIPE_SWIZZLE_Y, PIPE_SWIZZLE_Z, PIPE_SWIZZLE_W,
      },
   };

   const struct nil_descriptor desc =
      nil_image_view_descriptor(kgp_dev_info(tex->dev), image, &view,
                                tex->addr);
   static_assert(sizeof(desc) == sizeof(*out), "the descriptor is 256 bits");
   memcpy(out, &desc, sizeof(*out));
   return true;
}

/* CPU upload of one mip level of one layer, for texture memory the caller
 * has mapped: textureCpu is the CPU pointer to the same memory whose GPU
 * pointer went to kgpCreateTexture. The layout on the GPU side is tiled, so
 * this is not a memcpy. Ancestor: nvk_copy_memory_to_image.
 */
KGP_API bool
kgpWriteTexture(KgpTexture tex, void *textureCpu, uint32_t mip,
                uint32_t layer, const void *pixels, size_t rowStride)
{
   const struct nil_image *image = &tex->image;

   if (mip >= image->num_levels || layer >= image->extent_px.array_len ||
       textureCpu == NULL || pixels == NULL) {
      kgp_err("kgpWriteTexture: no such mip level or layer");
      return false;
   }

   const struct nil_image_level *level = &image->levels[mip];
   const struct nil_Extent4D_Pixels extent_px =
      nil_image_level_extent_px(image, mip);
   struct nil_Extent4D_Bytes extent_B =
      nil_extent4d_px_to_B(extent_px, image->format, image->sample_layout);
   extent_B.array_len = 1;

   if (rowStride < extent_B.width) {
      kgp_err("kgpWriteTexture: rowStride %zu is less than a row (%u bytes)",
              rowStride, extent_B.width);
      return false;
   }

   char *dst = (char *)textureCpu + level->offset_B +
               layer * image->array_stride_B;

   if (level->tiling.gob_type == NIL_GOB_TYPE_LINEAR) {
      for (uint32_t y = 0; y < extent_B.height; y++) {
         memcpy(dst + (size_t)y * level->row_stride_B,
                (const char *)pixels + y * rowStride, extent_B.width);
      }
   } else {
      nil_copy_linear_to_tiled(dst, extent_B, pixels, rowStride,
                               rowStride * extent_B.height,
                               (struct nil_Offset4D_Bytes) { 0 },
                               extent_B, &level->tiling);
   }
   return true;
}
