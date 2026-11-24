/* The KoreGP tests again, through Vulkan on the same driver and with the
 * same SPIR-V: vk_baseline saxpy | prefix_sum | scratch | indirect |
 * copy [linear]. If these pass on hardware and the KoreGP ones do not, the
 * bug is ours.
 */
#include "vk_compute.h"

#include "copy_pattern.h"
#include "genargs_spv.h"
#include "prefix_sum_spv.h"
#include "saxpy_spv.h"
#include "scratch_oracle.h"
#include "scratch_spv.h"
#include "tri_ps_spv.h"
#include "tri_scene.h"
#include "tri_vs_spv.h"

#include <string.h>

struct saxpy_root {
   uint64_t x;
   uint64_t y;
   float a;
   uint32_t count;
};

static int
run_saxpy(struct vkc *c)
{
   enum { COUNT = 100000 };
   const float a = 2.5f;

   struct vkc_buffer xb = vkc_buffer_create(c, COUNT * sizeof(float));
   struct vkc_buffer yb = vkc_buffer_create(c, COUNT * sizeof(float));
   struct vkc_buffer rb = vkc_buffer_create(c, sizeof(struct saxpy_root));
   float *x = xb.map, *y = yb.map;

   for (uint32_t i = 0; i < COUNT; i++) {
      x[i] = (float)i;
      y[i] = 1.0f + (float)(i % 7);
   }
   *(struct saxpy_root *)rb.map = (struct saxpy_root) {
      .x = xb.addr, .y = yb.addr, .a = a, .count = COUNT,
   };

   VkPipeline pipeline = vkc_pipeline_create(c, saxpy_spv, sizeof(saxpy_spv));
   VkCommandBuffer cmd = vkc_cmd_alloc(c);
   vkc_cmd_begin(cmd);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   vkc_cmd_dispatch(c, cmd, rb.addr, (COUNT + 63) / 64, 1, 1);
   vkc_cmd_submit(c, cmd);
   VK_OK(vkQueueWaitIdle(c->queue));

   int ret = 0;
   if (kgp_test_offline()) {
      printf("vk saxpy: recorded and submitted (offline)\n");
   } else {
      for (uint32_t i = 0; i < COUNT && ret == 0; i++) {
         const float expect = a * (float)i + (1.0f + (float)(i % 7));
         if (y[i] != expect) {
            fprintf(stderr, "vk saxpy: y[%u] = %f, expected %f\n",
                    i, y[i], expect);
            ret = 1;
         }
      }
      if (ret == 0)
         printf("vk saxpy: %u elements correct\n", COUNT);
   }

   vkDestroyPipeline(c->dev, pipeline, NULL);
   vkc_buffer_destroy(c, &rb);
   vkc_buffer_destroy(c, &yb);
   vkc_buffer_destroy(c, &xb);
   return ret;
}

struct scan_root {
   uint64_t src;
   uint64_t dst;
   uint32_t offset;
   uint32_t count;
};

static uint32_t
input_value(uint32_t i)
{
   return (i * 2654435761u) >> 20;
}

static int
run_prefix_sum(struct vkc *c)
{
   enum { COUNT = 70000, MAX_PASSES = 32 };

   struct vkc_buffer bb = vkc_buffer_create(c, 2 * COUNT * sizeof(uint32_t));
   struct vkc_buffer rb = vkc_buffer_create(c, MAX_PASSES * sizeof(struct scan_root));
   uint32_t *bufs = bb.map;
   struct scan_root *roots = rb.map;
   const uint64_t buf_gpu[2] = { bb.addr, bb.addr + COUNT * sizeof(uint32_t) };

   for (uint32_t i = 0; i < COUNT; i++)
      bufs[i] = input_value(i);

   VkPipeline pipeline =
      vkc_pipeline_create(c, prefix_sum_spv, sizeof(prefix_sum_spv));
   VkCommandBuffer cmd = vkc_cmd_alloc(c);
   vkc_cmd_begin(cmd);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

   uint32_t passes = 0;
   for (uint32_t offset = 1; offset < COUNT; offset *= 2, passes++) {
      CHECK(passes < MAX_PASSES);
      roots[passes] = (struct scan_root) {
         .src = buf_gpu[passes & 1],
         .dst = buf_gpu[(passes + 1) & 1],
         .offset = offset,
         .count = COUNT,
      };
      if (passes > 0) {
         vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
            1, &(VkMemoryBarrier) {
               .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
               .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
               .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            }, 0, NULL, 0, NULL);
      }
      vkc_cmd_dispatch(c, cmd, rb.addr + passes * sizeof(struct scan_root),
                       (COUNT + 63) / 64, 1, 1);
   }
   vkc_cmd_submit(c, cmd);
   VK_OK(vkQueueWaitIdle(c->queue));

   int ret = 0;
   if (kgp_test_offline()) {
      printf("vk prefix_sum: %u passes recorded and submitted (offline)\n",
             passes);
   } else {
      const uint32_t *result = bufs + (passes & 1) * COUNT;
      uint32_t sum = 0;
      for (uint32_t i = 0; i < COUNT && ret == 0; i++) {
         sum += input_value(i);
         if (result[i] != sum) {
            fprintf(stderr, "vk prefix_sum: [%u] = %u, expected %u\n",
                    i, result[i], sum);
            ret = 1;
         }
      }
      if (ret == 0)
         printf("vk prefix_sum: %u elements correct after %u passes\n",
                COUNT, passes);
   }

   vkDestroyPipeline(c->dev, pipeline, NULL);
   vkc_buffer_destroy(c, &rb);
   vkc_buffer_destroy(c, &bb);
   return ret;
}

static int
run_scratch(struct vkc *c)
{
   struct vkc_buffer sb = vkc_buffer_create(c, SCRATCH_COUNT * sizeof(uint32_t));
   struct vkc_buffer db[2];
   struct vkc_buffer rb = vkc_buffer_create(c, 2 * sizeof(struct scratch_root));
   uint32_t *src = sb.map;
   struct scratch_root *roots = rb.map;

   for (uint32_t i = 0; i < SCRATCH_COUNT; i++)
      src[i] = scratch_input(i);

   VkPipeline pipeline =
      vkc_pipeline_create(c, scratch_spv, sizeof(scratch_spv));

   /* Two command buffers in one submit, like kgp_scratch. */
   VkCommandBuffer cmds[2];
   for (unsigned r = 0; r < 2; r++) {
      db[r] = vkc_buffer_create(c, SCRATCH_COUNT * sizeof(uint32_t));
      roots[r] = (struct scratch_root) {
         .src = sb.addr, .dst = db[r].addr, .count = SCRATCH_COUNT,
      };
      cmds[r] = vkc_cmd_alloc(c);
      vkc_cmd_begin(cmds[r]);
      vkCmdBindPipeline(cmds[r], VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
      vkc_cmd_dispatch(c, cmds[r], rb.addr + r * sizeof(struct scratch_root),
                       SCRATCH_GROUPS, 1, 1);
      VK_OK(vkEndCommandBuffer(cmds[r]));
   }
   VK_OK(vkQueueSubmit(c->queue, 1, &(VkSubmitInfo) {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 2,
      .pCommandBuffers = cmds,
   }, VK_NULL_HANDLE));
   VK_OK(vkQueueWaitIdle(c->queue));

   int ret = 0;
   if (kgp_test_offline()) {
      printf("vk scratch: recorded and submitted (offline)\n");
   } else {
      for (unsigned r = 0; r < 2 && ret == 0; r++) {
         const uint32_t *dst = db[r].map;
         for (uint32_t i = 0; i < SCRATCH_COUNT && ret == 0; i++) {
            const uint32_t expect = scratch_expected(i, SCRATCH_GROUPS);
            if (dst[i] != expect) {
               fprintf(stderr, "vk scratch: dst%u[%u] = %u, expected %u\n",
                       r, i, dst[i], expect);
               ret = 1;
            }
         }
      }
      if (ret == 0)
         printf("vk scratch: 2 x %u elements correct\n", SCRATCH_COUNT);
   }

   vkDestroyPipeline(c->dev, pipeline, NULL);
   vkc_buffer_destroy(c, &rb);
   vkc_buffer_destroy(c, &db[1]);
   vkc_buffer_destroy(c, &db[0]);
   vkc_buffer_destroy(c, &sb);
   return ret;
}

struct genargs_root {
   uint64_t args;
   uint32_t count;
   uint32_t pad;
};

static int
run_indirect(struct vkc *c)
{
   enum { COUNT = 100000 };
   const float a = 2.5f;

   struct vkc_buffer xb = vkc_buffer_create(c, COUNT * sizeof(float));
   struct vkc_buffer yb = vkc_buffer_create(c, COUNT * sizeof(float));
   struct vkc_buffer sb = vkc_buffer_create(c, sizeof(struct saxpy_root));
   struct vkc_buffer gb = vkc_buffer_create(c, sizeof(struct genargs_root));
   struct vkc_buffer ab = vkc_buffer_create(c, 3 * sizeof(uint32_t));
   float *x = xb.map, *y = yb.map;
   const uint32_t *args = ab.map;

   for (uint32_t i = 0; i < COUNT; i++) {
      x[i] = (float)i;
      y[i] = 1.0f + (float)(i % 7);
   }
   *(struct saxpy_root *)sb.map = (struct saxpy_root) {
      .x = xb.addr, .y = yb.addr, .a = a, .count = COUNT,
   };
   *(struct genargs_root *)gb.map = (struct genargs_root) {
      .args = ab.addr, .count = COUNT,
   };

   VkPipeline genargs =
      vkc_pipeline_create(c, genargs_spv, sizeof(genargs_spv));
   VkPipeline saxpy = vkc_pipeline_create(c, saxpy_spv, sizeof(saxpy_spv));

   VkCommandBuffer cmd = vkc_cmd_alloc(c);
   vkc_cmd_begin(cmd);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, genargs);
   vkc_cmd_dispatch(c, cmd, gb.addr, 1, 1, 1);
   /* What kgpBarrier(COMPUTE, COMPUTE, KGP_HAZARD_DRAW_ARGUMENTS) means. */
   vkCmdPipelineBarrier(cmd,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
      VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0,
      1, &(VkMemoryBarrier) {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                          VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
      }, 0, NULL, 0, NULL);
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, saxpy);
   vkCmdPushConstants(cmd, c->layout, VK_SHADER_STAGE_COMPUTE_BIT,
                      0, sizeof(sb.addr), &sb.addr);
   vkCmdDispatchIndirect(cmd, ab.buffer, 0);
   vkc_cmd_submit(c, cmd);
   VK_OK(vkQueueWaitIdle(c->queue));

   int ret = 0;
   if (kgp_test_offline()) {
      printf("vk indirect: recorded and submitted (offline)\n");
   } else {
      if (args[0] != (COUNT + 63) / 64 || args[1] != 1 || args[2] != 1) {
         fprintf(stderr, "vk indirect: args are (%u, %u, %u)\n",
                 args[0], args[1], args[2]);
         ret = 1;
      }
      for (uint32_t i = 0; i < COUNT && ret == 0; i++) {
         const float expect = a * (float)i + (1.0f + (float)(i % 7));
         if (y[i] != expect) {
            fprintf(stderr, "vk indirect: y[%u] = %f, expected %f\n",
                    i, y[i], expect);
            ret = 1;
         }
      }
      if (ret == 0)
         printf("vk indirect: %u groups from the GPU, %u elements correct\n",
                args[0], COUNT);
   }

   vkDestroyPipeline(c->dev, saxpy, NULL);
   vkDestroyPipeline(c->dev, genargs, NULL);
   vkc_buffer_destroy(c, &ab);
   vkc_buffer_destroy(c, &gb);
   vkc_buffer_destroy(c, &sb);
   vkc_buffer_destroy(c, &yb);
   vkc_buffer_destroy(c, &xb);
   return ret;
}

/* The twin of tests/kgp_copy.c, minus its shader read: the same three
 * copies with the same three waits between and after them, recorded on the
 * transfer queue so that NVK uses the copy engine.
 */
static int
run_copy(struct vkc *c, bool linear)
{
   struct vkc_buffer sb = vkc_buffer_create(c, COPY_BYTES);
   struct vkc_buffer mb = vkc_buffer_create(c, COPY_BYTES);
   struct vkc_buffer ob = vkc_buffer_create(c, COPY_OUT_BYTES);
   copy_fill(sb.map);
   memset(ob.map, 0, COPY_OUT_BYTES);

   VkImage image;
   VK_OK(vkCreateImage(c->dev, &(VkImageCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = { COPY_W, COPY_H, 1 },
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = linear ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
   }, NULL, &image));

   VkMemoryRequirements reqs;
   vkGetImageMemoryRequirements(c->dev, image, &reqs);
   CHECK(reqs.memoryTypeBits != 0);
   VkDeviceMemory image_mem;
   VK_OK(vkAllocateMemory(c->dev, &(VkMemoryAllocateInfo) {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = __builtin_ctz(reqs.memoryTypeBits),
   }, NULL, &image_mem));
   VK_OK(vkBindImageMemory(c->dev, image, image_mem, 0));

   const VkImageSubresourceLayers layers = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1,
   };
   const VkMemoryBarrier wrote = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
   };
   VkImageMemoryBarrier layout = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1, .layerCount = 1,
      },
   };

   VkCommandBuffer cmd = vkc_cmd_alloc(c);
   vkc_cmd_begin(cmd);

   /* kgpMemCpy */
   vkCmdCopyBuffer(cmd, sb.buffer, mb.buffer, 1,
                   &(VkBufferCopy) { .size = COPY_BYTES });
   /* kgpBarrier(TRANSFER, TRANSFER) */
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                        1, &wrote, 0, NULL, 1, &layout);
   /* kgpCopyToTexture */
   vkCmdCopyBufferToImage(cmd, mb.buffer, image,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                          &(VkBufferImageCopy) {
                             .imageSubresource = layers,
                             .imageExtent = { COPY_W, COPY_H, 1 },
                          });
   /* kgpBarrier(TRANSFER, TRANSFER) */
   layout.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
   layout.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
   layout.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   layout.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                        0, NULL, 0, NULL, 1, &layout);
   /* kgpCopyFromTexture */
   vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          ob.buffer, 1,
                          &(VkBufferImageCopy) {
                             .bufferRowLength = COPY_OUT_ROW_TEXELS,
                             .imageSubresource = layers,
                             .imageExtent = { COPY_W, COPY_H, 1 },
                          });
   /* The wait half of kgpBarrier(TRANSFER, COMPUTE). */
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                        1, &wrote, 0, NULL, 0, NULL);
   vkc_cmd_submit(c, cmd);
   VK_OK(vkQueueWaitIdle(c->queue));

   int ret = 0;
   if (kgp_test_offline()) {
      printf("vk copy: recorded and submitted (offline)\n");
   } else {
      ret = copy_check("vk copied back", ob.map, COPY_OUT_ROW_TEXELS);
      if (ret == 0)
         printf("vk copy: %u texels correct\n", COPY_W * COPY_H);
   }

   vkDestroyImage(c->dev, image, NULL);
   vkFreeMemory(c->dev, image_mem, NULL);
   vkc_buffer_destroy(c, &ob);
   vkc_buffer_destroy(c, &mb);
   vkc_buffer_destroy(c, &sb);
   return ret;
}

static VkShaderModule
shader_module(struct vkc *c, const uint32_t *spirv, size_t size)
{
   VkShaderModule module;
   VK_OK(vkCreateShaderModule(c->dev, &(VkShaderModuleCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size,
      .pCode = spirv,
   }, NULL, &module));
   return module;
}

/* The twin of tests/kgp_triangle.c: the same two shaders, the same scene
 * (tri_scene.h), through a Vulkan graphics pipeline and dynamic rendering.
 * The two root pointers are a 16-byte push-constant block.
 */
static int
run_triangle(struct vkc *c, bool indirect, int scene)
{
   /* "depth" here: has a depth target and more than one draw. */
   const bool depth = scene != TRI_SCENE_PLAIN;
   const bool stencil = scene == TRI_SCENE_STENCIL;
   const unsigned draws = scene == TRI_SCENE_DEPTH ? TRI_DEPTH_DRAWS :
                          stencil ? TRI_STENCIL_DRAWS : 1;

   struct vkc_buffer ab =
      vkc_buffer_create(c, sizeof(VkDrawIndexedIndirectCommand));
   *(VkDrawIndexedIndirectCommand *)ab.map = (VkDrawIndexedIndirectCommand) {
      .indexCount = TRI_INDEX_COUNT, .instanceCount = TRI_INSTANCE_COUNT,
   };
   struct vkc_buffer vb =
      vkc_buffer_create(c, TRI_VERTEX_COUNT * sizeof(struct tri_vertex));
   struct vkc_buffer ib =
      vkc_buffer_create(c, TRI_INDEX_COUNT * sizeof(uint32_t));
   struct vkc_buffer vrb =
      vkc_buffer_create(c, TRI_DEPTH_DRAWS * sizeof(struct tri_vs_root));
   struct vkc_buffer prb =
      vkc_buffer_create(c, TRI_DEPTH_DRAWS * sizeof(struct tri_ps_root));
   struct vkc_buffer ob = vkc_buffer_create(c, TRI_W * TRI_H * 4);
   tri_fill(vb.map, ib.map);
   *(struct tri_vs_root *)vrb.map = (struct tri_vs_root) {
      .vertices = vb.addr, .instance_step = TRI_INSTANCE_STEP,
   };
   *(struct tri_ps_root *)prb.map = (struct tri_ps_root) {
      .tint = TRI_TINT_RGBA,
   };
   for (unsigned d = 0; depth && d < TRI_DEPTH_DRAWS; d++) {
      struct tri_vs_root *vr = (struct tri_vs_root *)vrb.map + d;
      struct tri_ps_root *pr = (struct tri_ps_root *)prb.map + d;
      *vr = (struct tri_vs_root) {
         .vertices = vb.addr,
         .instance_step = tri_depth_draws[d].instance_step,
         .z = tri_depth_draws[d].z,
      };
      memcpy(pr->tint, tri_depth_draws[d].tint, sizeof(pr->tint));
   }
   memset(ob.map, 0, TRI_W * TRI_H * 4);

   /* The depth target of the "depth" variant. TRANSFER_DST keeps NVK from
    * giving it a ZCULL plane; KoreGP has no ZCULL at all (see the awk
    * script).
    */
   const VkFormat depth_format = stencil ? VK_FORMAT_D24_UNORM_S8_UINT
                                         : VK_FORMAT_D32_SFLOAT;
   const VkImageSubresourceRange depth_range = {
      .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT |
                    (stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0),
      .levelCount = 1, .layerCount = 1,
   };
   VkImage depth_image = VK_NULL_HANDLE;
   VkDeviceMemory depth_mem = VK_NULL_HANDLE;
   VkImageView depth_view = VK_NULL_HANDLE;
   if (depth) {
      VK_OK(vkCreateImage(c->dev, &(VkImageCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .imageType = VK_IMAGE_TYPE_2D,
         .format = depth_format,
         .extent = { TRI_W, TRI_H, 1 },
         .mipLevels = 1,
         .arrayLayers = 1,
         .samples = VK_SAMPLE_COUNT_1_BIT,
         .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                  VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      }, NULL, &depth_image));
      VkMemoryRequirements depth_reqs;
      vkGetImageMemoryRequirements(c->dev, depth_image, &depth_reqs);
      CHECK(depth_reqs.memoryTypeBits != 0);
      VK_OK(vkAllocateMemory(c->dev, &(VkMemoryAllocateInfo) {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = depth_reqs.size,
         .memoryTypeIndex = __builtin_ctz(depth_reqs.memoryTypeBits),
      }, NULL, &depth_mem));
      VK_OK(vkBindImageMemory(c->dev, depth_image, depth_mem, 0));
      VK_OK(vkCreateImageView(c->dev, &(VkImageViewCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = depth_image,
         .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = depth_format,
         .subresourceRange = depth_range,
      }, NULL, &depth_view));
   }

   const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
   VkImage image;
   VK_OK(vkCreateImage(c->dev, &(VkImageCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = { TRI_W, TRI_H, 1 },
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
   }, NULL, &image));
   VkMemoryRequirements reqs;
   vkGetImageMemoryRequirements(c->dev, image, &reqs);
   CHECK(reqs.memoryTypeBits != 0);
   VkDeviceMemory image_mem;
   VK_OK(vkAllocateMemory(c->dev, &(VkMemoryAllocateInfo) {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.size,
      .memoryTypeIndex = __builtin_ctz(reqs.memoryTypeBits),
   }, NULL, &image_mem));
   VK_OK(vkBindImageMemory(c->dev, image, image_mem, 0));

   const VkImageSubresourceRange range = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1,
   };
   VkImageView view;
   VK_OK(vkCreateImageView(c->dev, &(VkImageViewCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = format,
      .subresourceRange = range,
   }, NULL, &view));

   VkPipelineLayout layout;
   VK_OK(vkCreatePipelineLayout(c->dev, &(VkPipelineLayoutCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &(VkPushConstantRange) {
         .stageFlags = VK_SHADER_STAGE_VERTEX_BIT |
                       VK_SHADER_STAGE_FRAGMENT_BIT,
         .size = 2 * sizeof(VkDeviceAddress),
      },
   }, NULL, &layout));

   VkShaderModule vs = shader_module(c, tri_vs_spv, sizeof(tri_vs_spv));
   VkShaderModule ps = shader_module(c, tri_ps_spv, sizeof(tri_ps_spv));
   const VkPipelineShaderStageCreateInfo stages[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs, .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = ps, .pName = "main",
      },
   };
   const VkViewport viewport = { 0, 0, TRI_W, TRI_H, 0.0f, 1.0f };
   const VkRect2D area = { { 0, 0 }, { TRI_W, TRI_H } };

   /* One pipeline per draw of the depth variant, each with the state the
    * KoreGP test has set for that draw (tri_scene.h); pipelines[0] alone
    * otherwise, with everything off.
    */
   VkPipeline pipelines[TRI_DEPTH_DRAWS] = { VK_NULL_HANDLE };
   for (unsigned d = 0; d < draws; d++) {
   const bool depth_on = scene == TRI_SCENE_DEPTH && d < 2;
   const bool blend_on = scene == TRI_SCENE_DEPTH && d != 1;
   const VkStencilOpState stencil_state = {
      .passOp = stencil && d == 0 ? VK_STENCIL_OP_REPLACE : VK_STENCIL_OP_KEEP,
      .compareOp = !stencil ? VK_COMPARE_OP_NEVER :
                   d == 0 ? VK_COMPARE_OP_ALWAYS : VK_COMPARE_OP_EQUAL,
      .compareMask = stencil ? 0xff : 0,
      .writeMask = stencil ? 0xff : 0,
      .reference = stencil ? TRI_STENCIL_REF : 0,
   };
   VK_OK(vkCreateGraphicsPipelines(c->dev, VK_NULL_HANDLE, 1,
      &(VkGraphicsPipelineCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
         .pNext = &(VkPipelineRenderingCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &format,
            .depthAttachmentFormat = depth ? depth_format
                                           : VK_FORMAT_UNDEFINED,
            .stencilAttachmentFormat = stencil ? depth_format
                                               : VK_FORMAT_UNDEFINED,
         },
         .stageCount = 2,
         .pStages = stages,
         .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
         },
         .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
         },
         .pViewportState = &(VkPipelineViewportStateCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1, .pViewports = &viewport,
            .scissorCount = 1, .pScissors = &area,
         },
         .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_NONE,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .lineWidth = 1.0f,
         },
         .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
         },
         .pDepthStencilState = &(VkPipelineDepthStencilStateCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = depth_on,
            .depthWriteEnable = depth_on,
            .depthCompareOp = depth_on ? VK_COMPARE_OP_LESS
                                       : VK_COMPARE_OP_NEVER,
            .stencilTestEnable = stencil,
            .front = stencil_state,
            .back = stencil_state,
         },
         .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .attachmentCount = 1,
            .pAttachments = &(VkPipelineColorBlendAttachmentState) {
               .blendEnable = blend_on,
               .srcColorBlendFactor = blend_on ? VK_BLEND_FACTOR_ONE : 0,
               .dstColorBlendFactor = blend_on ? VK_BLEND_FACTOR_ONE : 0,
               .srcAlphaBlendFactor = blend_on ? VK_BLEND_FACTOR_ONE : 0,
               .dstAlphaBlendFactor = blend_on ? VK_BLEND_FACTOR_ONE : 0,
               .colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                 VK_COLOR_COMPONENT_G_BIT |
                                 VK_COLOR_COMPONENT_B_BIT |
                                 VK_COLOR_COMPONENT_A_BIT,
            },
         },
         .layout = layout,
      }, NULL, &pipelines[d]));
   }

   VkImageMemoryBarrier to_target = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = range,
   };
   const VkDeviceAddress roots[2] = { vrb.addr, prb.addr };

   VkCommandBuffer cmd = vkc_cmd_alloc(c);
   vkc_cmd_begin(cmd);
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                        0, NULL, 0, NULL, 1, &to_target);
   if (depth) {
      VkImageMemoryBarrier to_depth = to_target;
      to_depth.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      to_depth.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      to_depth.image = depth_image;
      to_depth.subresourceRange = depth_range;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0,
                           0, NULL, 0, NULL, 1, &to_depth);
   }
   /* kgpBeginRenderPass */
   vkCmdBeginRendering(cmd, &(VkRenderingInfo) {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = area,
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &(VkRenderingAttachmentInfo) {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = view,
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .clearValue.color.float32 = TRI_CLEAR_RGBA,
      },
      .pDepthAttachment = !depth ? NULL : &(VkRenderingAttachmentInfo) {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = depth_view,
         .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .clearValue.depthStencil.depth = TRI_DEPTH_CLEAR,
      },
      .pStencilAttachment = !stencil ? NULL : &(VkRenderingAttachmentInfo) {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = depth_view,
         .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .clearValue.depthStencil.stencil = TRI_STENCIL_CLEAR,
      },
   });
   /* kgpSetPipeline + kgpDrawIndexedInstanced. (The depth variant's later
    * draws: the first starts like the other variants', so that the stream
    * up to it compares.)
    */
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines[0]);
   vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT |
                      VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(roots), roots);
   vkCmdBindIndexBuffer(cmd, ib.buffer, 0, VK_INDEX_TYPE_UINT32);
   if (depth) {
      vkCmdDrawIndexed(cmd, TRI_INDEX_COUNT, TRI_INSTANCE_COUNT, 0, 0, 0);
      for (unsigned d = 1; d < draws; d++) {
         const VkDeviceAddress draw_roots[2] = {
            vrb.addr + d * sizeof(struct tri_vs_root),
            prb.addr + d * sizeof(struct tri_ps_root),
         };
         vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines[d]);
         vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT |
                            VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                            sizeof(draw_roots), draw_roots);
         vkCmdDrawIndexed(cmd, TRI_INDEX_COUNT, TRI_INSTANCE_COUNT, 0, 0, 0);
      }
   } else if (indirect) {
      vkCmdDrawIndexedIndirect(cmd, ab.buffer, 0, 1,
                               sizeof(VkDrawIndexedIndirectCommand));
   } else {
      vkCmdDrawIndexed(cmd, TRI_INDEX_COUNT, TRI_INSTANCE_COUNT, 0, 0, 0);
   }
   /* kgpEndRenderPass */
   vkCmdEndRendering(cmd);

   /* Read back. (On this queue NVK copies with internal shaders, so nothing
    * past the draw is comparable with KoreGP's copy-engine readback.)
    */
   VkImageMemoryBarrier to_source = to_target;
   to_source.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
   to_source.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
   to_source.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   to_source.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                        0, NULL, 0, NULL, 1, &to_source);
   vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          ob.buffer, 1,
                          &(VkBufferImageCopy) {
                             .imageSubresource = {
                                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                .layerCount = 1,
                             },
                             .imageExtent = { TRI_W, TRI_H, 1 },
                          });
   vkc_cmd_submit(c, cmd);
   VK_OK(vkQueueWaitIdle(c->queue));

   int ret = 0;
   if (kgp_test_offline()) {
      printf("vk triangle: recorded and submitted (offline)\n");
   } else {
      ret = tri_check("vk triangle", ob.map, TRI_W, scene);
      if (ret == 0)
         printf("vk triangle: %ux%u pixels correct\n", TRI_W, TRI_H);
   }

   for (unsigned d = 0; d < TRI_DEPTH_DRAWS; d++)
      vkDestroyPipeline(c->dev, pipelines[d], NULL);
   vkDestroyImageView(c->dev, depth_view, NULL);
   vkDestroyImage(c->dev, depth_image, NULL);
   vkFreeMemory(c->dev, depth_mem, NULL);
   vkDestroyShaderModule(c->dev, ps, NULL);
   vkDestroyShaderModule(c->dev, vs, NULL);
   vkDestroyPipelineLayout(c->dev, layout, NULL);
   vkDestroyImageView(c->dev, view, NULL);
   vkDestroyImage(c->dev, image, NULL);
   vkFreeMemory(c->dev, image_mem, NULL);
   vkc_buffer_destroy(c, &ob);
   vkc_buffer_destroy(c, &prb);
   vkc_buffer_destroy(c, &vrb);
   vkc_buffer_destroy(c, &ib);
   vkc_buffer_destroy(c, &vb);
   vkc_buffer_destroy(c, &ab);
   return ret;
}

int
main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr, "usage: %s saxpy|prefix_sum|scratch|indirect|"
              "copy [linear]|triangle [indirect|depth|stencil]\n", argv[0]);
      return 2;
   }
   const bool copy = strcmp(argv[1], "copy") == 0;

   struct vkc c;
   vkc_init(&c, copy);

   int ret = 2;
   if (copy)
      ret = run_copy(&c, argc > 2 && strcmp(argv[2], "linear") == 0);
   else if (strcmp(argv[1], "saxpy") == 0)
      ret = run_saxpy(&c);
   else if (strcmp(argv[1], "prefix_sum") == 0)
      ret = run_prefix_sum(&c);
   else if (strcmp(argv[1], "scratch") == 0)
      ret = run_scratch(&c);
   else if (strcmp(argv[1], "indirect") == 0)
      ret = run_indirect(&c);
   else if (strcmp(argv[1], "triangle") == 0)
      ret = run_triangle(&c, argc > 2 && strcmp(argv[2], "indirect") == 0,
                         argc > 2 && strcmp(argv[2], "depth") == 0
                            ? TRI_SCENE_DEPTH :
                         argc > 2 && strcmp(argv[2], "stencil") == 0
                            ? TRI_SCENE_STENCIL : TRI_SCENE_PLAIN);

   vkc_finish(&c);
   return ret;
}
