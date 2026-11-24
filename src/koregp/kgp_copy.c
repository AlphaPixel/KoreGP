/*
 * GPU-side copies, on the copy engine: the "dedicated hardware path, used
 * directly" of the proposal's section 4.
 *
 * Ancestors, all in src/nouveau/vulkan/nvk_cmd_copy.c:
 *   nvk_cmd_copy_memory_ce             kgpMemCpy
 *   nouveau_copy_rect                  kgp_copy_rect
 *   nouveau_copy_rect_{buffer,image},
 *   nouveau_copy_remap_format,
 *   nvk_cmd_copy_memory_to_image_ce,
 *   nvk_cmd_copy_image_to_memory_ce    kgpCopyToTexture, kgpCopyFromTexture
 * Cut: regions and offsets (a copy here is one whole mip level of one
 * layer), depth/stencil aspect remaps, multisampling, pre-Pascal branches.
 *
 * NVK reaches these paths only on its transfer-only queue; on a queue that
 * has compute it copies with internal shaders instead. That is why the
 * Vulkan twin of tests/kgp_copy.c records on the transfer queue.
 *
 * SPDX-License-Identifier: MIT
 */
#include "kgp_private.h"

#include "clcab5.h"
#include "nv_push_cl90b5.h"
#include "nv_push_clc1b5.h"
#include "nv_push_clcab5.h"
#include "util/format/u_format.h"

/* Ancestor: nvk_cmd_copy_memory_ce. */
KGP_API void
kgpMemCpy(KgpCommandBuffer cb, void *dstGpu, void *srcGpu, uint64_t bytes)
{
   uint64_t dst_addr = (uintptr_t)dstGpu;
   uint64_t src_addr = (uintptr_t)srcGpu;

   if (unlikely(dst_addr == 0 || src_addr == 0)) {
      kgp_err("kgpMemCpy: NULL GPU pointer");
      cb->failed = true;
      return;
   }
   if (!kgp_check_ptr(cb, "kgpMemCpy: dstGpu", dstGpu, bytes, 1) ||
       !kgp_check_ptr(cb, "kgpMemCpy: srcGpu", srcGpu, bytes, 1))
      return;

   while (bytes) {
      struct nv_push *p = kgp_cb_push(cb, 10);

      P_MTHD(p, NV90B5, OFFSET_IN_UPPER);
      P_NV90B5_OFFSET_IN_UPPER(p, src_addr >> 32);
      P_NV90B5_OFFSET_IN_LOWER(p, src_addr & 0xffffffff);
      P_NV90B5_OFFSET_OUT_UPPER(p, dst_addr >> 32);
      P_NV90B5_OFFSET_OUT_LOWER(p, dst_addr & 0xffffffff);

      const unsigned chunk = MIN2(bytes, 1 << 17);

      P_MTHD(p, NV90B5, LINE_LENGTH_IN);
      P_NV90B5_LINE_LENGTH_IN(p, chunk);
      P_NV90B5_LINE_COUNT(p, 1);

      P_IMMD(p, NV90B5, LAUNCH_DMA, {
             .data_transfer_type = DATA_TRANSFER_TYPE_PIPELINED,
             .multi_line_enable = MULTI_LINE_ENABLE_TRUE,
             .flush_enable = FLUSH_ENABLE_TRUE,
             .src_memory_layout = SRC_MEMORY_LAYOUT_PITCH,
             .dst_memory_layout = DST_MEMORY_LAYOUT_PITCH,
      });

      src_addr += chunk;
      dst_addr += chunk;
      bytes -= chunk;
   }
}

/* One side of a rectangle copy. Ancestor: nouveau_copy_buffer, without the
 * offsets. A zeroed tiling is NIL_GOB_TYPE_LINEAR: rows `row_stride` apart.
 */
struct kgp_copy_side {
   uint64_t base_addr;
   bool is_3d;
   struct nil_Extent4D_Elements extent_el;   /* of the whole level; tiled only */
   uint32_t bpp;
   uint32_t row_stride;
   uint32_t array_stride;
   struct nil_tiling tiling;
};

/* Ancestor: nil_to_nvcab5_gob_type. */
static uint32_t
kgp_nvcab5_gob_type(enum nil_gob_type gob_type)
{
#define NVCAB5_GOB_KIND(NV) NVCAB5_SET_SRC_BLOCK_SIZE_KIND_BPP_BL_##NV
   switch (gob_type) {
   case NIL_GOB_TYPE_TURING_COLOR2_D:  return NVCAB5_GOB_KIND(32);
   case NIL_GOB_TYPE_BLACKWELL8_BIT:   return NVCAB5_GOB_KIND(8);
   case NIL_GOB_TYPE_BLACKWELL16_BIT:  return NVCAB5_GOB_KIND(16);
   case NIL_GOB_TYPE_BLACKWELL_Z24:    return NVCAB5_GOB_KIND(24);
   default:
      UNREACHABLE("Invalid GOB type on Blackwell+");
   }
#undef NVCAB5_GOB_KIND
}

/* Copies `extent_el` (width, height, and depth slices or one layer) from
 * the start of `src` to the start of `dst`. Ancestor: nouveau_copy_rect,
 * always with component remapping, so X dimensions are in elements.
 */
static void
kgp_copy_rect(struct KgpCommandBuffer_T *cb, const struct kgp_copy_side *src,
              const struct kgp_copy_side *dst,
              struct nil_Extent4D_Elements extent_el, uint8_t comp_size)
{
   const uint16_t cls_copy = kgp_dev_info(cb->queue->dev)->cls_copy;

   assert(src->bpp % comp_size == 0 && dst->bpp % comp_size == 0);

   struct nv_push *p = kgp_cb_push(cb, 2);
   P_IMMD(p, NV90B5, SET_REMAP_COMPONENTS, {
      .dst_x = 0,
      .dst_y = 1,
      .dst_z = 2,
      .dst_w = 3,
      .component_size = comp_size - 1,
      .num_src_components = src->bpp / comp_size - 1,
      .num_dst_components = dst->bpp / comp_size - 1,
   });

   for (unsigned z = 0; z < extent_el.depth; z++) {
      uint64_t src_addr = src->base_addr;
      uint64_t dst_addr = dst->base_addr;

      if (!src->is_3d)
         src_addr += (uint64_t)z * src->array_stride;
      if (!dst->is_3d)
         dst_addr += (uint64_t)z * dst->array_stride;

      p = kgp_cb_push(cb, 31);

      P_MTHD(p, NV90B5, OFFSET_IN_UPPER);
      P_NV90B5_OFFSET_IN_UPPER(p, src_addr >> 32);
      P_NV90B5_OFFSET_IN_LOWER(p, src_addr & 0xffffffff);
      P_NV90B5_OFFSET_OUT_UPPER(p, dst_addr >> 32);
      P_NV90B5_OFFSET_OUT_LOWER(p, dst_addr & 0xffffffff);
      P_NV90B5_PITCH_IN(p, src->row_stride);
      P_NV90B5_PITCH_OUT(p, dst->row_stride);
      P_NV90B5_LINE_LENGTH_IN(p, extent_el.width);
      P_NV90B5_LINE_COUNT(p, extent_el.height);

      uint32_t src_layout, dst_layout;
      if (src->tiling.gob_type != NIL_GOB_TYPE_LINEAR) {
         P_MTHD(p, NV90B5, SET_SRC_BLOCK_SIZE);
         assert(nil_gob_type_height(src->tiling.gob_type) == 8);

         if (cls_copy >= BLACKWELL_DMA_COPY_B) {
            P_NVCAB5_SET_SRC_BLOCK_SIZE(p, {
               .width = 0, /* Tiles are always 1 GOB wide */
               .height = src->tiling.y_log2,
               .depth = src->tiling.z_log2,
               .gob_height = GOB_HEIGHT_GOB_HEIGHT_FERMI_8,
               .kind_bpp = kgp_nvcab5_gob_type(src->tiling.gob_type),
            });
         } else {
            P_NV90B5_SET_SRC_BLOCK_SIZE(p, {
               .width = 0, /* Tiles are always 1 GOB wide */
               .height = src->tiling.y_log2,
               .depth = src->tiling.z_log2,
               .gob_height = GOB_HEIGHT_GOB_HEIGHT_FERMI_8,
            });
         }
         /* The copy hardware has no concept of a tile width: the width is
          * the stride in elements.
          */
         P_NV90B5_SET_SRC_WIDTH(p, src->row_stride / src->bpp);
         P_NV90B5_SET_SRC_HEIGHT(p, src->extent_el.height);
         P_NV90B5_SET_SRC_DEPTH(p, src->extent_el.depth);
         P_NV90B5_SET_SRC_LAYER(p, src->is_3d ? z : 0);

         P_MTHD(p, NVC1B5, SRC_ORIGIN_X);
         P_NVC1B5_SRC_ORIGIN_X(p, 0);
         P_NVC1B5_SRC_ORIGIN_Y(p, 0);

         src_layout = NV90B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_BLOCKLINEAR;
      } else {
         src_layout = NV90B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH;
      }

      if (dst->tiling.gob_type != NIL_GOB_TYPE_LINEAR) {
         P_MTHD(p, NV90B5, SET_DST_BLOCK_SIZE);
         assert(nil_gob_type_height(dst->tiling.gob_type) == 8);

         if (cls_copy >= BLACKWELL_DMA_COPY_B) {
            P_NVCAB5_SET_DST_BLOCK_SIZE(p, {
               .width = 0, /* Tiles are always 1 GOB wide */
               .height = dst->tiling.y_log2,
               .depth = dst->tiling.z_log2,
               .gob_height = GOB_HEIGHT_GOB_HEIGHT_FERMI_8,
               .kind_bpp = kgp_nvcab5_gob_type(dst->tiling.gob_type),
            });
         } else {
            P_NV90B5_SET_DST_BLOCK_SIZE(p, {
               .width = 0, /* Tiles are always 1 GOB wide */
               .height = dst->tiling.y_log2,
               .depth = dst->tiling.z_log2,
               .gob_height = GOB_HEIGHT_GOB_HEIGHT_FERMI_8,
            });
         }
         P_NV90B5_SET_DST_WIDTH(p, dst->row_stride / dst->bpp);
         P_NV90B5_SET_DST_HEIGHT(p, dst->extent_el.height);
         P_NV90B5_SET_DST_DEPTH(p, dst->extent_el.depth);
         P_NV90B5_SET_DST_LAYER(p, dst->is_3d ? z : 0);

         P_MTHD(p, NVC1B5, DST_ORIGIN_X);
         P_NVC1B5_DST_ORIGIN_X(p, 0);
         P_NVC1B5_DST_ORIGIN_Y(p, 0);

         dst_layout = NV90B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_BLOCKLINEAR;
      } else {
         dst_layout = NV90B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH;
      }

      P_IMMD(p, NV90B5, LAUNCH_DMA, {
         .data_transfer_type = DATA_TRANSFER_TYPE_PIPELINED,
         .multi_line_enable = MULTI_LINE_ENABLE_TRUE,
         .flush_enable = FLUSH_ENABLE_TRUE,
         .src_memory_layout = src_layout,
         .dst_memory_layout = dst_layout,
         .remap_enable = REMAP_ENABLE_TRUE,
      });
   }
}

/* Ancestor: nouveau_copy_remap_format. Any component size that divides the
 * texel and is at most 4 bytes will do for a plain copy.
 */
static uint8_t
kgp_copy_comp_size(uint32_t bpp)
{
   if (bpp % 3 == 0)
      return bpp / 3;
   return MIN2(bpp, 4);
}

/* One mip level of one layer (every depth slice of a 3D texture), to or
 * from row-major texels in caller memory.
 */
static void
kgp_copy_texture(KgpCommandBuffer cb, KgpTexture tex, uint32_t mip,
                 uint32_t layer, void *memGpu, size_t rowStride,
                 bool to_texture, const char *func)
{
   const struct nil_image *image = &tex->image;

   if (unlikely(mip >= image->num_levels ||
                layer >= image->extent_px.array_len || memGpu == NULL)) {
      kgp_err("%s: no such mip level or layer, or a NULL GPU pointer", func);
      cb->failed = true;
      return;
   }

   const struct nil_image_level *level = &image->levels[mip];
   struct nil_Extent4D_Elements extent_el =
      nil_extent4d_px_to_el(nil_image_level_extent_px(image, mip),
                            image->format, image->sample_layout);
   extent_el.array_len = 1;
   const uint32_t bpp = util_format_get_blocksize(image->format.p_format);

   if (unlikely(rowStride < (size_t)extent_el.width * bpp ||
                rowStride % bpp != 0 || rowStride > UINT32_MAX)) {
      kgp_err("%s: rowStride %zu must be a multiple of the texel size and "
              "at least a row (%u bytes)", func, rowStride,
              extent_el.width * bpp);
      cb->failed = true;
      return;
   }

   /* The last row is read or written only as far as the texels go. */
   const uint64_t rows = (uint64_t)extent_el.height * extent_el.depth;
   if (!kgp_check_texture(cb, func, tex) ||
       !kgp_check_ptr(cb, func, memGpu,
                      (rows - 1) * rowStride + (uint64_t)extent_el.width * bpp,
                      1))
      return;

   const struct kgp_copy_side mem_side = {
      .base_addr = (uintptr_t)memGpu,
      .bpp = bpp,
      .row_stride = rowStride,
      .array_stride = rowStride * extent_el.height,
   };
   const struct kgp_copy_side tex_side = {
      .base_addr = tex->addr + level->offset_B +
                   (uint64_t)layer * image->array_stride_B,
      .is_3d = image->dim == NIL_IMAGE_DIM_3D,
      .extent_el = extent_el,
      .bpp = bpp,
      .row_stride = level->row_stride_B,
      .array_stride = image->array_stride_B,
      .tiling = level->tiling,
   };

   if (to_texture)
      kgp_copy_rect(cb, &mem_side, &tex_side, extent_el,
                    kgp_copy_comp_size(bpp));
   else
      kgp_copy_rect(cb, &tex_side, &mem_side, extent_el,
                    kgp_copy_comp_size(bpp));
}

/* Ancestor: nvk_cmd_copy_memory_to_image_ce. */
KGP_API void
kgpCopyToTexture(KgpCommandBuffer cb, KgpTexture tex, uint32_t mip,
                 uint32_t layer, void *srcGpu, size_t rowStride)
{
   kgp_copy_texture(cb, tex, mip, layer, srcGpu, rowStride, true,
                    "kgpCopyToTexture");
}

/* Ancestor: nvk_cmd_copy_image_to_memory_ce. */
KGP_API void
kgpCopyFromTexture(KgpCommandBuffer cb, KgpTexture tex, uint32_t mip,
                   uint32_t layer, void *dstGpu, size_t rowStride)
{
   kgp_copy_texture(cb, tex, mip, layer, dstGpu, rowStride, false,
                    "kgpCopyFromTexture");
}
