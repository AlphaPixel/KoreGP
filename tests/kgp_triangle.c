/* First triangles (plan N3, task R6): the scene of tri_scene.h, two
 * instances of an indexed quad drawn by a vertex-pulling vertex shader and
 * a tinting pixel shader into a cleared texture in GPU-only memory, read
 * back with the copy engine and checked pixel by pixel.
 *
 * Exercises: the 3D channel state, a graphics pipeline, the two raster root
 * pointers (delta D10), a render pass with a clear, the index buffer and
 * the draw macro, gl_VertexIndex and gl_InstanceIndex.
 *
 * "depth" (task R7) is the three-draw variant of tri_scene.h: a depth
 * target, and depth-stencil and blend state objects set, unset and set
 * again between draws. "stencil" is its two-draw variant: a depth-stencil
 * target, stencil written by one draw and tested by the next. "multi"
 * (task R9b) is one GPU-driven multi-draw with per-draw root data; it has
 * no Vulkan twin, there being no Vulkan spelling for it.
 *
 * Vulkan twin: vk_baseline triangle.
 */
#include "kgp_test.h"

#include "saxpy_spv.h"
#include "tri_ps_spv.h"
#include "tri_scene.h"
#include "tri_vs_spv.h"

#include <string.h>

#define SPAN(a) ((KgpByteSpan) { a, sizeof(a) })

int
main(int argc, char **argv)
{
   /* "indirect": the same draw with its arguments read by the GPU. */
   const bool indirect = argc > 1 && strcmp(argv[1], "indirect") == 0;
   const bool depth = argc > 1 && strcmp(argv[1], "depth") == 0;
   const bool stencil = argc > 1 && strcmp(argv[1], "stencil") == 0;
   const bool multi = argc > 1 && strcmp(argv[1], "multi") == 0;
   const int scene = depth ? TRI_SCENE_DEPTH :
                     stencil ? TRI_SCENE_STENCIL :
                     multi ? TRI_SCENE_MULTI : TRI_SCENE_PLAIN;
   const bool has_depth = depth || stencil;

   KgpDevice dev = kgpCreateDevice(0);
   CHECK(dev);
   KgpQueue queue = kgpGetQueue(dev);

   struct tri_vertex *vertices =
      kgpMalloc(dev, TRI_VERTEX_COUNT * sizeof(*vertices), 64,
                KGP_MEMORY_DEFAULT);
   uint32_t *indices =
      kgpMalloc(dev, TRI_INDEX_COUNT * sizeof(*indices), 64,
                KGP_MEMORY_DEFAULT);
   /* One pair of roots per draw; only "depth" draws more than once. */
   struct tri_vs_root *vs_root =
      kgpMalloc(dev, TRI_MAX_DRAWS * sizeof(*vs_root), 64,
                KGP_MEMORY_DEFAULT);
   struct tri_ps_root *ps_root =
      kgpMalloc(dev, TRI_MAX_DRAWS * sizeof(*ps_root), 64,
                KGP_MEMORY_DEFAULT);
   uint32_t *out = kgpMalloc(dev, TRI_W * TRI_H * 4, 64, KGP_MEMORY_READBACK);
   CHECK(vertices && indices && vs_root && ps_root && out);

   tri_fill(vertices, indices);
   *vs_root = (struct tri_vs_root) {
      .vertices = (uintptr_t)kgpHostToDevicePointer(dev, vertices),
      .instance_step = TRI_INSTANCE_STEP,
   };
   *ps_root = (struct tri_ps_root) { .tint = TRI_TINT_RGBA };
   /* The multi-draw's records, count and per-draw roots. CPU-written here;
    * the GPU reads all of them when the macro runs.
    */
   KgpDrawIndexedArgs *multi_args =
      kgpMalloc(dev, TRI_MULTI_RECORDS * sizeof(*multi_args), 64,
                KGP_MEMORY_DEFAULT);
   uint32_t *multi_count = kgpMalloc(dev, sizeof(*multi_count), 64,
                                     KGP_MEMORY_DEFAULT);
   CHECK(multi_args && multi_count);
   *multi_count = TRI_MULTI_COUNT;
   for (unsigned d = 0; d < TRI_MULTI_RECORDS; d++) {
      multi_args[d] = (KgpDrawIndexedArgs) {
         .indexCount = TRI_INDEX_COUNT,
         .instanceCount = tri_multi_draws[d].instance_count,
         .firstInstance = tri_multi_draws[d].first_instance,
      };
      if (!multi)
         continue;
      vs_root[d] = (struct tri_vs_root) {
         .vertices = vs_root[0].vertices,
         .instance_step = tri_multi_draws[d].instance_step,
      };
      memcpy(ps_root[d].tint, tri_multi_draws[d].tint, sizeof(ps_root[d].tint));
   }
   for (unsigned d = 0; has_depth && d < TRI_DEPTH_DRAWS; d++) {
      vs_root[d] = (struct tri_vs_root) {
         .vertices = vs_root[0].vertices,
         .instance_step = tri_depth_draws[d].instance_step,
         .z = tri_depth_draws[d].z,
      };
      memcpy(ps_root[d].tint, tri_depth_draws[d].tint, sizeof(ps_root[d].tint));
   }
   memset(out, 0, TRI_W * TRI_H * 4);

   /* The render target: a texture over GPU-only memory. */
   const KgpTextureDesc tex_desc = {
      .format = KGP_FORMAT_RGBA8_UNORM,
      .width = TRI_W,
      .height = TRI_H,
   };
   KgpSizeAlign sa;
   CHECK(kgpTextureSizeAlign(dev, &tex_desc, &sa));
   void *target_gpu = kgpMalloc(dev, sa.size, sa.align, KGP_MEMORY_GPU);
   CHECK(target_gpu);
   KgpTexture target = kgpCreateTexture(dev, &tex_desc, target_gpu);
   CHECK(target);

   /* The depth target: also a texture over GPU-only memory, which D24S8
    * requires (delta D4).
    */
   const KgpTextureDesc depth_desc = {
      .format = stencil ? KGP_FORMAT_D24_UNORM_S8_UINT : KGP_FORMAT_D32_FLOAT,
      .width = TRI_W,
      .height = TRI_H,
   };
   CHECK(kgpTextureSizeAlign(dev, &depth_desc, &sa));
   void *depth_gpu = kgpMalloc(dev, sa.size, sa.align, KGP_MEMORY_GPU);
   CHECK(depth_gpu);
   KgpTexture depth_tex = kgpCreateTexture(dev, &depth_desc, depth_gpu);
   CHECK(depth_tex);

   KgpDepthStencilState less = kgpCreateDepthStencilState(
      dev, &(KgpDepthStencilDesc) {
         .depthTest = true,
         .depthWrite = true,
         .depthCompare = KGP_COMPARE_LESS,
      });
   KgpBlendState additive = kgpCreateBlendState(
      dev, &(KgpBlendDesc) {
         .srcColor = KGP_BLEND_ONE, .dstColor = KGP_BLEND_ONE,
         .srcAlpha = KGP_BLEND_ONE, .dstAlpha = KGP_BLEND_ONE,
      });
   CHECK(less && additive);
   const KgpStencilFace mark = {
      .pass = KGP_STENCIL_REPLACE, .compare = KGP_COMPARE_ALWAYS,
   };
   const KgpStencilFace match = { .compare = KGP_COMPARE_EQUAL };
   KgpDepthStencilState stencil_mark = kgpCreateDepthStencilState(
      dev, &(KgpDepthStencilDesc) {
         .stencilTest = true,
         .stencilFront = mark, .stencilBack = mark,
         .stencilReadMask = 0xff, .stencilWriteMask = 0xff,
         .stencilReference = TRI_STENCIL_REF,
      });
   KgpDepthStencilState stencil_match = kgpCreateDepthStencilState(
      dev, &(KgpDepthStencilDesc) {
         .stencilTest = true,
         .stencilFront = match, .stencilBack = match,
         .stencilReadMask = 0xff, .stencilWriteMask = 0xff,
         .stencilReference = TRI_STENCIL_REF,
      });
   CHECK(stencil_mark && stencil_match);
   CHECK(!kgpCreateBlendState(dev, &(KgpBlendDesc) { .colorOp = 99 }));

   const KgpRasterDesc raster = {
      .topology = KGP_TOPOLOGY_TRIANGLE_LIST,
      .cull = KGP_CULL_NONE,
      .colorTargetCount = 1,
      .colorFormats = { KGP_FORMAT_RGBA8_UNORM },
      .depthFormat = has_depth ? depth_desc.format : KGP_FORMAT_NONE,
   };
   KgpPipeline pipeline = kgpCreateGraphicsPipeline(
      dev, SPAN(tri_vs_spv), SPAN(tri_ps_spv), &raster, NULL);
   KgpPipeline compute = kgpCreateComputePipeline(dev, SPAN(saxpy_spv), NULL);
   KgpSemaphore done = kgpCreateSemaphore(dev, 0);
   CHECK(pipeline && compute && done);

   KgpRenderPassDesc pass = {
      .colorTargetCount = 1,
      .colorTargets = { {
         .texture = target,
         .load = KGP_LOAD_CLEAR,
         .clearColor = TRI_CLEAR_RGBA,
      } },
   };
   const KgpRenderPassDesc pass_no_depth = pass;
   if (has_depth) {
      pass.depthTarget = (KgpDepthTarget) {
         .texture = depth_tex,
         .load = KGP_LOAD_CLEAR,
         .clearDepth = TRI_DEPTH_CLEAR,
         .clearStencil = TRI_STENCIL_CLEAR,
      };
   }
   void *vs_gpu = kgpHostToDevicePointer(dev, vs_root);
   void *ps_gpu = kgpHostToDevicePointer(dev, ps_root);
   void *indices_gpu = kgpHostToDevicePointer(dev, indices);

   KgpCommandBuffer cb = kgpStartCommandRecording(queue);
   CHECK(cb);
   kgpBeginRenderPass(cb, &pass);
   kgpSetPipeline(cb, pipeline);
   KgpDrawIndexedArgs *args = NULL;
   if (indirect) {
      args = kgpMalloc(dev, sizeof(*args), 64, KGP_MEMORY_DEFAULT);
      CHECK(args);
      *args = (KgpDrawIndexedArgs) {
         .indexCount = TRI_INDEX_COUNT, .instanceCount = TRI_INSTANCE_COUNT,
      };
      kgpDrawIndexedInstancedIndirect(cb, vs_gpu, ps_gpu, indices_gpu,
                                      kgpHostToDevicePointer(dev, args));
   } else if (depth) {
      for (unsigned d = 0; d < TRI_DEPTH_DRAWS; d++) {
         kgpSetDepthStencilState(cb, d < 2 ? less : NULL);
         kgpSetBlendState(cb, d != 1 ? additive : NULL);
         kgpDrawIndexedInstanced(cb, (char *)vs_gpu + d * sizeof(*vs_root),
                                 (char *)ps_gpu + d * sizeof(*ps_root),
                                 indices_gpu, TRI_INDEX_COUNT,
                                 TRI_INSTANCE_COUNT);
      }
   } else if (multi) {
      kgpDrawIndexedInstancedIndirectMulti(
         cb, vs_gpu, sizeof(*vs_root), ps_gpu, sizeof(*ps_root), indices_gpu,
         kgpHostToDevicePointer(dev, multi_args),
         kgpHostToDevicePointer(dev, multi_count), TRI_MULTI_MAX);
   } else if (stencil) {
      for (unsigned d = 0; d < TRI_STENCIL_DRAWS; d++) {
         kgpSetDepthStencilState(cb, d == 0 ? stencil_mark : stencil_match);
         kgpDrawIndexedInstanced(cb, (char *)vs_gpu + d * sizeof(*vs_root),
                                 (char *)ps_gpu + d * sizeof(*ps_root),
                                 indices_gpu, TRI_INDEX_COUNT,
                                 TRI_INSTANCE_COUNT);
      }
   } else {
      kgpDrawIndexedInstanced(cb, vs_gpu, ps_gpu, indices_gpu,
                              TRI_INDEX_COUNT, TRI_INSTANCE_COUNT);
   }
   kgpEndRenderPass(cb);
   kgpBarrier(cb, KGP_STAGE_RASTER_COLOR_OUT, KGP_STAGE_TRANSFER,
              KGP_HAZARD_NONE);
   kgpCopyFromTexture(cb, target, 0, 0, kgpHostToDevicePointer(dev, out),
                      TRI_W * 4);
   CHECK(kgpSubmit(queue, &cb, 1, done, 1));
   /* Only now: kgpFree defers against what has been submitted, not against
    * what is recorded. Freed before the submit, the arguments were unmapped
    * when the GPU read them (MMU fault, window of 2026-11-22).
    */
   kgpFree(dev, args);
   CHECK(kgpWaitSemaphore(dev, done, 1, KGP_TEST_TIMEOUT_NS));

   /* Misuse fails the command buffer, not the process. */
   fprintf(stderr, "(the errors logged by this test are expected)\n");
   KgpCommandBuffer bad = kgpStartCommandRecording(queue);
   CHECK(bad);
   kgpSetPipeline(bad, pipeline);                    /* no render pass */
   kgpDrawIndexedInstanced(bad, vs_gpu, ps_gpu, indices_gpu, 3, 1);
   CHECK(!kgpSubmit(queue, &bad, 1, NULL, 0));
   bad = kgpStartCommandRecording(queue);
   CHECK(bad);
   kgpBeginRenderPass(bad, &pass);                   /* a compute pipeline */
   kgpSetPipeline(bad, compute);
   kgpDrawIndexedInstanced(bad, vs_gpu, ps_gpu, indices_gpu, 3, 1);
   CHECK(!kgpSubmit(queue, &bad, 1, NULL, 0));
   bad = kgpStartCommandRecording(queue);
   CHECK(bad);
   kgpBeginRenderPass(bad, &pass);                   /* two passes open */
   kgpBeginRenderPass(bad, &pass);
   CHECK(!kgpSubmit(queue, &bad, 1, NULL, 0));
   bad = kgpStartCommandRecording(queue);
   CHECK(bad);
   kgpEndRenderPass(bad);                            /* none open */
   CHECK(!kgpSubmit(queue, &bad, 1, NULL, 0));
   bad = kgpStartCommandRecording(queue);
   CHECK(bad);
   kgpBeginRenderPass(bad, &pass_no_depth);          /* depth test, no depth */
   kgpSetPipeline(bad, pipeline);
   kgpSetDepthStencilState(bad, less);
   kgpDrawIndexedInstanced(bad, vs_gpu, ps_gpu, indices_gpu, 3, 1);
   CHECK(!kgpSubmit(queue, &bad, 1, NULL, 0));
   bad = kgpStartCommandRecording(queue);
   CHECK(bad);
   kgpBeginRenderPass(bad, &pass_no_depth);          /* stencil, no stencil */
   kgpSetPipeline(bad, pipeline);
   kgpSetDepthStencilState(bad, stencil_mark);
   kgpDrawIndexedInstanced(bad, vs_gpu, ps_gpu, indices_gpu, 3, 1);
   CHECK(!kgpSubmit(queue, &bad, 1, NULL, 0));
   /* A depth-stencil texture over mapped memory is refused. */
   const KgpTextureDesc ds_desc = {
      .format = KGP_FORMAT_D24_UNORM_S8_UINT, .width = TRI_W, .height = TRI_H,
   };
   CHECK(kgpTextureSizeAlign(dev, &ds_desc, &sa));
   void *ds_mapped = kgpMalloc(dev, sa.size, sa.align, KGP_MEMORY_DEFAULT);
   CHECK(ds_mapped);
   CHECK(!kgpCreateTexture(dev, &ds_desc,
                           kgpHostToDevicePointer(dev, ds_mapped)));
   kgpFree(dev, ds_mapped);

   int ret = 0;
   if (kgp_test_offline()) {
      printf("triangle: recorded and submitted (offline, results not "
             "checked)\n");
   } else {
      ret = tri_check("triangle", out, TRI_W, scene);
      if (ret == 0 && multi)
         printf("triangle: %ux%u pixels correct: three GPU-counted draws, "
                "each with its own arguments and root data\n", TRI_W, TRI_H);
      else if (ret == 0 && stencil)
         printf("triangle: %ux%u pixels correct: stencil clear, write and "
                "test\n", TRI_W, TRI_H);
      else if (ret == 0 && depth)
         printf("triangle: %ux%u pixels correct: depth test and write, "
                "additive blending, state changes between draws\n",
                TRI_W, TRI_H);
      else if (ret == 0)
         printf("triangle: %ux%u pixels correct: clear, two instances of an "
                "indexed quad, both root pointers\n", TRI_W, TRI_H);
   }

   kgpDestroySemaphore(dev, done);
   kgpDestroyPipeline(dev, compute);
   kgpDestroyPipeline(dev, pipeline);
   kgpFree(dev, multi_count);
   kgpFree(dev, multi_args);
   kgpDestroyBlendState(dev, additive);
   kgpDestroyDepthStencilState(dev, stencil_match);
   kgpDestroyDepthStencilState(dev, stencil_mark);
   kgpDestroyDepthStencilState(dev, less);
   kgpDestroyTexture(dev, depth_tex);
   kgpFree(dev, depth_gpu);
   kgpDestroyTexture(dev, target);
   kgpFree(dev, target_gpu);
   kgpFree(dev, out);
   kgpFree(dev, ps_root);
   kgpFree(dev, vs_root);
   kgpFree(dev, indices);
   kgpFree(dev, vertices);
   kgpDestroyDevice(dev);
   return ret;
}
