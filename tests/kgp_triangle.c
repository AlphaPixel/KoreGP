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
 * target, stencil written by one draw and tested by the next. "state" is
 * a two-draw variant for depth bias, write mask and blend constant. "mrt"
 * draws into two colour targets, the second with a write mask. "multi"
 * (task R9b) is one GPU-driven multi-draw with per-draw root data; it has
 * no Vulkan twin, there being no Vulkan spelling for it. "multigpu" is
 * that draw with its arguments, count and root data written by a compute
 * shader in the same command buffer (tri_scene.h). "bench" times the
 * recording of draws instead of checking one (tri_scene.h). A format's
 * name ("srgb", "rgba16f", "rgba32f", "r32f") draws the plain scene into a
 * target of that format. "mesh" draws the plain scene from a mesh shader,
 * one workgroup per quad, with a per-primitive output; "meshindirect" has
 * the GPU read its group counts. "msaa" draws into a four-sample target
 * and reads it back with a shader that averages the samples (tri_scene.h).
 *
 * Vulkan twin: vk_baseline triangle.
 */
#include "kgp_test.h"

#include "multi_gen_spv.h"
#include "saxpy_spv.h"
#include "tri_mesh_ps_spv.h"
#include "tri_mrt_ps_spv.h"
#include "tri_ms_spv.h"
#include "tri_ps_spv.h"
#include "tri_resolve_spv.h"
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
   const bool state = argc > 1 && strcmp(argv[1], "state") == 0;
   const bool mrt = argc > 1 && strcmp(argv[1], "mrt") == 0;
   const bool msaa = argc > 1 && strcmp(argv[1], "msaa") == 0;
   const bool meshind = argc > 1 && strcmp(argv[1], "meshindirect") == 0;
   const bool mesh = meshind || (argc > 1 && strcmp(argv[1], "mesh") == 0);
   const bool multigpu = argc > 1 && strcmp(argv[1], "multigpu") == 0;
   const bool multi = multigpu || (argc > 1 && strcmp(argv[1], "multi") == 0);
   const bool bench = argc > 1 && strcmp(argv[1], "bench") == 0;
   const int scene = depth ? TRI_SCENE_DEPTH :
                     stencil ? TRI_SCENE_STENCIL :
                     state ? TRI_SCENE_STATE :
                     multi ? TRI_SCENE_MULTI : TRI_SCENE_PLAIN;
   const bool has_depth = depth || stencil || state;
   const int tfmt = tri_target_from_arg(argc > 1 ? argv[1] : NULL);
   static const KGP_FORMAT target_formats[TRI_TARGET_COUNT] = {
      [TRI_TARGET_RGBA8] = KGP_FORMAT_RGBA8_UNORM,
      [TRI_TARGET_SRGB8] = KGP_FORMAT_RGBA8_SRGB,
      [TRI_TARGET_RGBA16F] = KGP_FORMAT_RGBA16_FLOAT,
      [TRI_TARGET_RGBA32F] = KGP_FORMAT_RGBA32_FLOAT,
      [TRI_TARGET_R32F] = KGP_FORMAT_R32_FLOAT,
   };
   const uint32_t bpp = tri_targets[tfmt].bpp;

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
   uint32_t *out = kgpMalloc(dev, TRI_W * TRI_H * bpp, 64,
                             KGP_MEMORY_READBACK);
   CHECK(vertices && indices && vs_root && ps_root && out);

   tri_fill(vertices, indices);
   for (unsigned i = 0; msaa && i < TRI_VERTEX_COUNT; i++)
      vertices[i].pos[0] += TRI_MSAA_SHIFT;
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
   /* "multigpu": what the compute shader is to write starts as zero. */
   const uint64_t vertices_gpu = vs_root[0].vertices;
   if (multigpu) {
      *multi_count = 0;
      memset(multi_args, 0, TRI_MULTI_COUNT * sizeof(*multi_args));
      memset(vs_root, 0, TRI_MULTI_COUNT * sizeof(*vs_root));
      memset(ps_root, 0, TRI_MULTI_COUNT * sizeof(*ps_root));
   }
   for (unsigned d = 0; (depth || stencil) && d < TRI_DEPTH_DRAWS; d++) {
      vs_root[d] = (struct tri_vs_root) {
         .vertices = vs_root[0].vertices,
         .instance_step = tri_depth_draws[d].instance_step,
         .z = tri_depth_draws[d].z,
      };
      memcpy(ps_root[d].tint, tri_depth_draws[d].tint, sizeof(ps_root[d].tint));
   }
   for (unsigned d = 0; state && d < TRI_STATE_DRAWS; d++) {
      vs_root[d] = (struct tri_vs_root) {
         .vertices = vertices_gpu,
         .instance_step = tri_state_draws[d].instance_step,
         .z = tri_state_draws[d].z,
      };
      memcpy(ps_root[d].tint, tri_state_draws[d].tint, sizeof(ps_root[d].tint));
   }
   for (unsigned d = 1; bench && d < TRI_BENCH_ROOTS; d++) {
      vs_root[d] = vs_root[0];
      ps_root[d] = ps_root[0];
   }
   memset(out, 0, TRI_W * TRI_H * bpp);

   /* The render target: a texture over GPU-only memory. */
   const KgpTextureDesc tex_desc = {
      .format = target_formats[tfmt],
      .width = TRI_W,
      .height = TRI_H,
      .sampleCount = msaa ? TRI_MSAA_SAMPLES : 0,
   };
   KgpSizeAlign sa;
   CHECK(kgpTextureSizeAlign(dev, &tex_desc, &sa));
   void *target_gpu = kgpMalloc(dev, sa.size, sa.align, KGP_MEMORY_GPU);
   CHECK(target_gpu);
   KgpTexture target = kgpCreateTexture(dev, &tex_desc, target_gpu);
   CHECK(target);
   /* The mrt variant's second target and its readback. */
   void *target1_gpu = kgpMalloc(dev, sa.size, sa.align, KGP_MEMORY_GPU);
   uint32_t *out1 = kgpMalloc(dev, TRI_W * TRI_H * 4, 64, KGP_MEMORY_READBACK);
   CHECK(target1_gpu && out1);
   KgpTexture target1 = kgpCreateTexture(dev, &tex_desc, target1_gpu);
   CHECK(target1);
   memset(out1, 0, TRI_W * TRI_H * 4);

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
   /* The state variant's draw 0 (tri_scene.h). */
   KgpDepthStencilState less_biased = kgpCreateDepthStencilState(
      dev, &(KgpDepthStencilDesc) {
         .depthTest = true,
         .depthWrite = true,
         .depthCompare = KGP_COMPARE_LESS,
         .depthBias = TRI_STATE_BIAS,
         .depthBiasSlope = TRI_STATE_BIAS_SLOPE,
      });
   KgpBlendState modulate = kgpCreateBlendState(
      dev, &(KgpBlendDesc) {
         .srcColor = KGP_BLEND_CONSTANT_COLOR, .dstColor = KGP_BLEND_ZERO,
         .srcAlpha = KGP_BLEND_CONSTANT_ALPHA, .dstAlpha = KGP_BLEND_ZERO,
         .constant = TRI_STATE_BLEND_CONSTANT,
         .colorWrite = KGP_COLOR_WRITE_R | KGP_COLOR_WRITE_B |
                       KGP_COLOR_WRITE_A,
      });
   CHECK(less_biased && modulate);
   CHECK(!kgpCreateBlendState(dev, &(KgpBlendDesc) {
            .colorWrite = KGP_COLOR_WRITE_NONE | KGP_COLOR_WRITE_R }));

   const KgpRasterDesc raster = {
      .topology = KGP_TOPOLOGY_TRIANGLE_LIST,
      .cull = KGP_CULL_NONE,
      .colorTargetCount = mrt ? 2 : 1,
      .colorFormats = { target_formats[tfmt], target_formats[tfmt] },
      .depthFormat = has_depth ? depth_desc.format : KGP_FORMAT_NONE,
      .sampleCount = msaa ? TRI_MSAA_SAMPLES : 0,
      .colorWrite = {
         [1] = KGP_COLOR_WRITE_R | KGP_COLOR_WRITE_G | KGP_COLOR_WRITE_A,
      },
   };
   KgpPipeline pipeline = mesh
      ? kgpCreateMeshPipeline(dev, SPAN(tri_ms_spv), SPAN(tri_mesh_ps_spv),
                              &raster, NULL)
      : kgpCreateGraphicsPipeline(
           dev, SPAN(tri_vs_spv),
           mrt ? SPAN(tri_mrt_ps_spv) : SPAN(tri_ps_spv), &raster, NULL);
   KgpRasterDesc bad_raster = raster;
   bad_raster.colorWrite[0] = KGP_COLOR_WRITE_NONE | KGP_COLOR_WRITE_A;
   CHECK(!kgpCreateGraphicsPipeline(dev, SPAN(tri_vs_spv), SPAN(tri_ps_spv),
                                    &bad_raster, NULL));
   KgpPipeline compute = kgpCreateComputePipeline(dev, SPAN(saxpy_spv), NULL);
   /* "msaa": the shader that reads the target back, and its heap of one. */
   KgpPipeline resolve =
      kgpCreateComputePipeline(dev, SPAN(tri_resolve_spv), NULL);
   KgpDescriptor256 *heap =
      kgpMalloc(dev, sizeof(*heap), 64, KGP_MEMORY_DEFAULT);
   struct tri_resolve_root *resolve_root =
      kgpMalloc(dev, sizeof(*resolve_root), 64, KGP_MEMORY_DEFAULT);
   CHECK(resolve && heap && resolve_root);
   CHECK(kgpTextureViewDescriptor(target, NULL, &heap[0]));
   *resolve_root = (struct tri_resolve_root) {
      .dst = (uintptr_t)kgpHostToDevicePointer(dev, out),
      .width = TRI_W,
      .samples = msaa ? TRI_MSAA_SAMPLES : 1,
   };
   KgpSemaphore done = kgpCreateSemaphore(dev, 0);
   CHECK(pipeline && compute && done);

   KgpRenderPassDesc pass = {
      .colorTargetCount = mrt ? 2 : 1,
      .colorTargets = {
         {
            .texture = target,
            .load = KGP_LOAD_CLEAR,
            .clearColor = TRI_CLEAR_RGBA,
         },
         {
            .texture = mrt ? target1 : NULL,
            .load = KGP_LOAD_CLEAR,
            .clearColor = TRI_MRT_CLEAR1_RGBA,
         },
      },
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
   int ret = 0;

   /* "multigpu": the candidates, and the shader that turns them into the
    * multi-draw's inputs.
    */
   struct tri_candidate *candidates = NULL;
   struct tri_gen_root *gen_root = NULL;
   KgpPipeline gen = NULL;
   if (multigpu) {
      candidates = kgpMalloc(dev, TRI_MULTI_RECORDS * sizeof(*candidates), 64,
                             KGP_MEMORY_DEFAULT);
      gen_root = kgpMalloc(dev, sizeof(*gen_root), 64, KGP_MEMORY_DEFAULT);
      gen = kgpCreateComputePipeline(dev, SPAN(multi_gen_spv), NULL);
      CHECK(candidates && gen_root && gen);
      for (unsigned c = 0; c < TRI_MULTI_RECORDS; c++) {
         const unsigned d = tri_gen_order[c];
         candidates[c] = (struct tri_candidate) {
            .instance_step = tri_multi_draws[d].instance_step,
            .instance_count = tri_multi_draws[d].instance_count,
            .first_instance = tri_multi_draws[d].first_instance,
            .visible = d < TRI_MULTI_COUNT,
         };
         memcpy(candidates[c].tint, tri_multi_draws[d].tint,
                sizeof(candidates[c].tint));
      }
      *gen_root = (struct tri_gen_root) {
         .candidates = (uintptr_t)kgpHostToDevicePointer(dev, candidates),
         .args = (uintptr_t)kgpHostToDevicePointer(dev, multi_args),
         .count = (uintptr_t)kgpHostToDevicePointer(dev, multi_count),
         .vs_roots = (uintptr_t)vs_gpu,
         .ps_roots = (uintptr_t)ps_gpu,
         .vertices = vertices_gpu,
         .candidate_count = TRI_MULTI_RECORDS,
         .index_count = TRI_INDEX_COUNT,
      };
   }

   if (bench) {
      const uint32_t cbs = argc > 2 ? strtoul(argv[2], NULL, 0) : 1000;
      const uint32_t n = argc > 3 ? strtoul(argv[3], NULL, 0) : 100;
      uint64_t record_ns = 0, submit_ns = 0;
      for (uint32_t i = 0; i < TRI_BENCH_WARMUP + cbs; i++) {
         const uint64_t t0 = tri_cpu_ns();
         KgpCommandBuffer bcb = kgpStartCommandRecording(queue);
         kgpBeginRenderPass(bcb, &pass);
         kgpSetPipeline(bcb, pipeline);
         for (uint32_t j = 0; j < n; j++) {
            const size_t r = j % TRI_BENCH_ROOTS;
            kgpDrawIndexedInstanced(bcb,
                                    (char *)vs_gpu + r * sizeof(*vs_root),
                                    (char *)ps_gpu + r * sizeof(*ps_root),
                                    indices_gpu, TRI_INDEX_COUNT,
                                    TRI_INSTANCE_COUNT);
         }
         kgpEndRenderPass(bcb);
         const uint64_t t1 = tri_cpu_ns();
         CHECK(kgpSubmit(queue, &bcb, 1, done, i + 1));
         const uint64_t t2 = tri_cpu_ns();
         CHECK(kgpWaitSemaphore(dev, done, i + 1, KGP_TEST_TIMEOUT_NS));
         if (i >= TRI_BENCH_WARMUP) {
            record_ns += t1 - t0;
            submit_ns += t2 - t1;
         }
      }
      tri_bench_report("kgp", cbs, n, record_ns, submit_ns);
      goto cleanup;
   }

   KgpCommandBuffer cb = kgpStartCommandRecording(queue);
   CHECK(cb);
   if (multigpu) {
      kgpSetPipeline(cb, gen);
      kgpDispatch(cb, kgpHostToDevicePointer(dev, gen_root),
                  (KgpUvec3) { 1, 1, 1 });
      /* The draw's arguments and count are read by the macro processor,
       * the root structs by the two shader stages.
       */
      kgpBarrier(cb, KGP_STAGE_COMPUTE,
                 KGP_STAGE_VERTEX_SHADER | KGP_STAGE_PIXEL_SHADER,
                 KGP_HAZARD_DRAW_ARGUMENTS);
   }
   kgpBeginRenderPass(cb, &pass);
   kgpSetPipeline(cb, pipeline);
   KgpDrawIndexedArgs *args = NULL;
   KgpUvec3 *mesh_groups = NULL;
   if (meshind) {
      mesh_groups = kgpMalloc(dev, sizeof(*mesh_groups), 64,
                              KGP_MEMORY_DEFAULT);
      CHECK(mesh_groups);
      *mesh_groups = (KgpUvec3) { TRI_INSTANCE_COUNT, 1, 1 };
      kgpDrawMeshletsIndirect(cb, vs_gpu, ps_gpu,
                              kgpHostToDevicePointer(dev, mesh_groups));
   } else if (mesh) {
      kgpDrawMeshlets(cb, vs_gpu, ps_gpu,
                      (KgpUvec3) { TRI_INSTANCE_COUNT, 1, 1 });
   } else if (indirect) {
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
   } else if (state) {
      for (unsigned d = 0; d < TRI_STATE_DRAWS; d++) {
         kgpSetDepthStencilState(cb, d == 0 ? less_biased : less);
         kgpSetBlendState(cb, d == 0 ? modulate : NULL);
         kgpDrawIndexedInstanced(cb, (char *)vs_gpu + d * sizeof(*vs_root),
                                 (char *)ps_gpu + d * sizeof(*ps_root),
                                 indices_gpu, TRI_INDEX_COUNT,
                                 TRI_INSTANCE_COUNT);
      }
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
   if (msaa) {
      kgpBarrier(cb, KGP_STAGE_RASTER_COLOR_OUT, KGP_STAGE_COMPUTE,
                 KGP_HAZARD_NONE);
      kgpSetActiveTextureHeapPtr(cb, kgpHostToDevicePointer(dev, heap), 1);
      kgpSetPipeline(cb, resolve);
      kgpDispatch(cb, kgpHostToDevicePointer(dev, resolve_root),
                  (KgpUvec3) { TRI_W / 8, TRI_H / 8, 1 });
   } else {
      kgpBarrier(cb, KGP_STAGE_RASTER_COLOR_OUT, KGP_STAGE_TRANSFER,
                 KGP_HAZARD_NONE);
      kgpCopyFromTexture(cb, target, 0, 0, kgpHostToDevicePointer(dev, out),
                         TRI_W * bpp);
   }
   if (mrt) {
      kgpCopyFromTexture(cb, target1, 0, 0,
                         kgpHostToDevicePointer(dev, out1), TRI_W * 4);
   }
   CHECK(kgpSubmit(queue, &cb, 1, done, 1));
   /* Only now: kgpFree defers against what has been submitted, not against
    * what is recorded. Freed before the submit, the arguments were unmapped
    * when the GPU read them (MMU fault, window of 2026-11-22).
    */
   kgpFree(dev, args);
   kgpFree(dev, mesh_groups);
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

   if (kgp_test_offline()) {
      printf("triangle: recorded and submitted (offline, results not "
             "checked)\n");
   } else {
      if (multigpu && *multi_count != TRI_MULTI_COUNT) {
         fprintf(stderr, "triangle: the compute shader left a draw count of "
                 "%u, expected %u\n", *multi_count, TRI_MULTI_COUNT);
         ret = 1;
      }
      if (ret == 0 && msaa)
         ret = tri_check_msaa("triangle", out);
      else if (ret == 0 && tfmt != TRI_TARGET_RGBA8)
         ret = tri_check_target("triangle", out, tfmt);
      else if (ret == 0)
         ret = tri_check("triangle", out, TRI_W, scene);
      if (ret == 0 && mrt)
         ret = tri_check_mrt1("triangle", out1);
      if (ret == 0 && multigpu)
         printf("triangle: %ux%u pixels correct: three draws out of four "
                "candidates, their arguments, count and root data written "
                "by a compute shader\n", TRI_W, TRI_H);
      else if (ret == 0 && multi)
         printf("triangle: %ux%u pixels correct: three GPU-counted draws, "
                "each with its own arguments and root data\n", TRI_W, TRI_H);
      else if (ret == 0 && state)
         printf("triangle: %ux%u pixels correct: depth bias, colour write "
                "mask and blend constant\n", TRI_W, TRI_H);
      else if (ret == 0 && mrt)
         printf("triangle: %ux%u pixels correct in each of two colour "
                "targets, the second through a write mask\n", TRI_W, TRI_H);
      else if (ret == 0 && stencil)
         printf("triangle: %ux%u pixels correct: stencil clear, write and "
                "test\n", TRI_W, TRI_H);
      else if (ret == 0 && depth)
         printf("triangle: %ux%u pixels correct: depth test and write, "
                "additive blending, state changes between draws\n",
                TRI_W, TRI_H);
      else if (ret == 0 && tfmt != TRI_TARGET_RGBA8)
         printf("triangle: %ux%u pixels correct in a %s target\n",
                TRI_W, TRI_H, tri_targets[tfmt].name);
      else if (ret == 0)
         printf("triangle: %ux%u pixels correct: clear, two instances of an "
                "indexed quad, both root pointers\n", TRI_W, TRI_H);
   }

cleanup:
   if (gen != NULL)
      kgpDestroyPipeline(dev, gen);
   kgpFree(dev, gen_root);
   kgpFree(dev, candidates);
   kgpDestroySemaphore(dev, done);
   kgpDestroyPipeline(dev, compute);
   kgpDestroyPipeline(dev, resolve);
   kgpFree(dev, resolve_root);
   kgpFree(dev, heap);
   kgpDestroyPipeline(dev, pipeline);
   kgpFree(dev, multi_count);
   kgpFree(dev, multi_args);
   kgpDestroyBlendState(dev, modulate);
   kgpDestroyDepthStencilState(dev, less_biased);
   kgpDestroyBlendState(dev, additive);
   kgpDestroyDepthStencilState(dev, stencil_match);
   kgpDestroyDepthStencilState(dev, stencil_mark);
   kgpDestroyDepthStencilState(dev, less);
   kgpDestroyTexture(dev, depth_tex);
   kgpFree(dev, depth_gpu);
   kgpDestroyTexture(dev, target1);
   kgpFree(dev, out1);
   kgpFree(dev, target1_gpu);
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
