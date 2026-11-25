/* The scene of tests/kgp_triangle.c and of its Vulkan twin (vk_baseline
 * triangle): one quad (two triangles, six indices, four vertices) drawn as
 * two instances into a cleared 64x64 RGBA8 target. Every edge falls on a
 * pixel boundary, so the expected image is exact.
 *
 *    clear          opaque black
 *    instance 0     x in [8, 24),  y in [16, 48)
 *    instance 1     x in [40, 56), y in [16, 48)   (shifted by the VS root)
 *    quad colour    vertex colour (1,1,0,1) x pixel-stage tint (1,0,1,1) = red
 *
 * A wrong vertex root shows as misplaced or missing quads, a wrong pixel
 * root as a wrong colour, a wrong instance index as one quad.
 *
 * The "depth" variant (kgp_triangle depth, vk_baseline triangle depth) adds
 * a depth target cleared to 1 and draws three times, changing depth-stencil
 * and blend state between draws of one pipeline:
 *
 *    draw 0   the scene above at z 0.25, depth test LESS and write,
 *             additive blending: red on black
 *    draw 1   green quads at z 0.75, x in [8, 24) and [24, 40), same depth
 *             state, no blending: the first is behind draw 0 and must not
 *             show, the second is visible
 *    draw 2   green quads where draw 0's are, no depth test, additive
 *             blending: red + green = yellow
 *
 *    result   x in [8, 24) yellow, [24, 40) green, [40, 56) yellow
 *
 * No depth test leaves [8, 24) green; no depth clear leaves draws 0 and 1
 * invisible; no blending in draw 2 leaves green where yellow should be.
 *
 * The "stencil" variant has a depth-stencil target with stencil cleared to
 * 0 and draws 0 and 1 of the above without depth test or blending:
 *
 *    draw 0   red quads; stencil always passes and is replaced by 1
 *    draw 1   green quads at [8, 24) and [24, 40); stencil must equal 1
 *
 *    result   x in [8, 24) green, [24, 40) black, [40, 56) red
 *
 * No stencil write leaves [8, 24) red; no stencil test makes [24, 40) green.
 *
 * The "state" variant is for depth bias, the colour write mask and the
 * blend constant. A float depth target cleared to 1, two draws, both at z
 * 0.5 with depth test LESS and write:
 *
 *    draw 0   yellow quads (tint white) at [8, 24) and [40, 56), with
 *             - a depth bias of +64 units: 0.5 and a little is stored
 *             - blending src * CONSTANT + dst * 0, the constant (1,1,0,1):
 *               the colour comes through unchanged
 *             - a write mask without green: what lands is red
 *    draw 1   green quads at [8, 24) and [24, 40), no bias, no blend state:
 *             at [8, 24) it is in front of draw 0 by exactly draw 0's bias
 *
 *    result   x in [8, 24) green, [24, 40) green, [40, 56) red
 *
 * No bias in draw 0 (or one left on in draw 1) leaves [8, 24) red; a write
 * mask ignored leaves [40, 56) yellow; a constant never set leaves it
 * transparent black.
 *
 * The "mrt" variant draws the plain scene into two colour targets at once,
 * with tri_mrt.frag: the first gets the plain scene, the second, cleared
 * to red, the complement of the first's colour (cyan in the quads) through
 * a per-target write mask without blue:
 *
 *    target 0   red quads on black
 *    target 1   green quads on red
 *
 * Outputs crossed over show as cyan-ish quads in target 0; a second target
 * never written stays red; a mask ignored leaves its quads cyan.
 */
#ifndef TRI_SCENE_H
#define TRI_SCENE_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define TRI_W 64u
#define TRI_H 64u
#define TRI_INDEX_COUNT 6u
#define TRI_VERTEX_COUNT 4u
#define TRI_INSTANCE_COUNT 2u

/* tri.vert: struct Vertex { vec2 pos; vec4 color; }, scalar layout */
struct tri_vertex {
   float pos[2];
   float color[4];
};

/* tri.vert's and tri.frag's root structs, padded to a cache line. */
struct tri_vs_root {
   uint64_t vertices;
   float instance_step;
   float z;
   uint32_t pad[12];
};
struct tri_ps_root {
   float tint[4];
   uint32_t pad[12];
};
_Static_assert(sizeof(struct tri_vs_root) == 64 &&
               sizeof(struct tri_ps_root) == 64, "root structs are 64 bytes");

#define TRI_INSTANCE_STEP 1.0f
#define TRI_CLEAR_RGBA { 0.0f, 0.0f, 0.0f, 1.0f }
#define TRI_TINT_RGBA { 1.0f, 0.0f, 1.0f, 1.0f }

enum {
   TRI_SCENE_PLAIN = 0, TRI_SCENE_DEPTH, TRI_SCENE_STENCIL, TRI_SCENE_MULTI,
   TRI_SCENE_STATE,
   TRI_SCENE_MRT,       /* target 0 is the plain scene */
};
#define TRI_MRT_CLEAR1_RGBA { 1.0f, 0.0f, 0.0f, 1.0f }

/* Room for any variant's per-draw roots. */
#define TRI_MAX_DRAWS 4u

/* The "multi" variant: one GPU-driven multi-draw. The count in memory is 3,
 * the maximum passed is 8, and a fourth record is there to show if the
 * count is ignored. No depth, no blending; later draws paint over earlier.
 *
 *    draw 0   2 instances, step 1.0, red      [8, 24) and [40, 56)
 *    draw 1   1 instance, the second (firstInstance 1), step 0.5, green
 *                                             [24, 40)
 *    draw 2   1 instance, step 1.0, yellow    [8, 24), over draw 0's
 *    (draw 3  1 instance, the second, step 0.5, red: must not run)
 *
 *    result   x in [8, 24) yellow, [24, 40) green, [40, 56) red
 */
#define TRI_MULTI_COUNT 3u
#define TRI_MULTI_MAX 8u
#define TRI_MULTI_RECORDS 4u
static const struct {
   float instance_step, tint[4];
   uint32_t instance_count, first_instance;
} tri_multi_draws[TRI_MULTI_RECORDS] = {
   { 1.0f, { 1.0f, 0.0f, 1.0f, 1.0f }, 2, 0 },
   { 0.5f, { 0.0f, 1.0f, 1.0f, 1.0f }, 1, 1 },
   { 1.0f, { 1.0f, 1.0f, 1.0f, 1.0f }, 1, 0 },
   { 0.5f, { 1.0f, 0.0f, 1.0f, 1.0f }, 1, 1 },
};

/* The "multigpu" variant: the same picture, with the multi-draw's count,
 * its first three records and their root structs written by a compute
 * shader (multi_gen.comp) earlier in the same command buffer. The shader
 * is given four candidates, the draws above in the order of
 * tri_gen_order with only the first TRI_MULTI_COUNT of the table visible,
 * and appends a draw for each visible one. The culled candidate sits
 * between visible ones and is the red draw over [24, 40): drawn, it shows
 * over draw 1's green. Record 3 is written by the CPU, as above, to show if
 * the count is ignored; everything the shader is to write starts as zero,
 * so nothing is drawn if it does not run or the draw does not see it.
 */
static const uint32_t tri_gen_order[TRI_MULTI_RECORDS] = { 0, 1, 3, 2 };

/* multi_gen.comp: Candidate and GenRoot, scalar layout */
struct tri_candidate {
   float instance_step;
   float tint[4];
   uint32_t instance_count;
   uint32_t first_instance;
   uint32_t visible;
};
struct tri_gen_root {
   uint64_t candidates;
   uint64_t args;
   uint64_t count;
   uint64_t vs_roots;
   uint64_t ps_roots;
   uint64_t vertices;
   uint32_t candidate_count;
   uint32_t index_count;
};
_Static_assert(sizeof(struct tri_candidate) == 32, "scalar layout");

/* The depth variant's three draws; the stencil variant uses the first two. */
#define TRI_DEPTH_DRAWS 3u
#define TRI_STENCIL_DRAWS 2u
#define TRI_DEPTH_CLEAR 1.0f
#define TRI_STENCIL_CLEAR 0
#define TRI_STENCIL_REF 1
static const struct {
   float instance_step, z, tint[4];
} tri_depth_draws[TRI_DEPTH_DRAWS] = {
   { 1.0f, 0.25f, TRI_TINT_RGBA },
   { 0.5f, 0.75f, { 0.0f, 1.0f, 1.0f, 1.0f } },
   { 1.0f, 0.75f, { 0.0f, 1.0f, 1.0f, 1.0f } },
};

/* "bench" (kgp_triangle bench, vk_baseline triangle bench, each followed
 * by [command buffers] [draws per command buffer]): the CPU cost of
 * recording draws, the raster counterpart of bench_dispatch.c and timed the
 * same way. Each command buffer is one render pass with a clear, the
 * pipeline set once and N draws of the plain scene, every draw with its own
 * pair of root pointers (cycling through TRI_BENCH_ROOTS). In Vulkan a draw
 * is vkCmdPushConstants + vkCmdDrawIndexed, the index buffer bound once.
 */
#define TRI_BENCH_ROOTS 3u
#define TRI_BENCH_WARMUP 16u

static inline uint64_t
tri_cpu_ns(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
   return (uint64_t)ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static inline void
tri_bench_report(const char *api, uint32_t cbs, uint32_t n,
                 uint64_t record_ns, uint64_t submit_ns)
{
   printf("%-4s %u command buffers x %u draws: "
          "record %.1f ns/draw, submit %.2f us/command buffer\n",
          api, cbs, n, (double)record_ns / ((double)cbs * n),
          (double)submit_ns / cbs / 1000.0);
}

/* The state variant's two draws, and draw 0's state. */
#define TRI_STATE_DRAWS 2u
#define TRI_STATE_BIAS 64.0f
#define TRI_STATE_BIAS_SLOPE 1.5f    /* the quads have no slope */
#define TRI_STATE_BLEND_CONSTANT { 1.0f, 1.0f, 0.0f, 1.0f }
static const struct {
   float instance_step, z, tint[4];
} tri_state_draws[TRI_STATE_DRAWS] = {
   { 1.0f, 0.5f, { 1.0f, 1.0f, 1.0f, 1.0f } },
   { 0.5f, 0.5f, { 0.0f, 1.0f, 1.0f, 1.0f } },
};

static inline void
tri_fill(struct tri_vertex *vertices, uint32_t *indices)
{
   static const float corners[TRI_VERTEX_COUNT][2] = {
      { -0.75f, -0.5f }, { -0.25f, -0.5f }, { -0.25f, 0.5f }, { -0.75f, 0.5f },
   };
   for (unsigned i = 0; i < TRI_VERTEX_COUNT; i++) {
      vertices[i] = (struct tri_vertex) {
         .pos = { corners[i][0], corners[i][1] },
         .color = { 1.0f, 1.0f, 0.0f, 1.0f },
      };
   }
   static const uint32_t quad[TRI_INDEX_COUNT] = { 0, 1, 2, 0, 2, 3 };
   for (unsigned i = 0; i < TRI_INDEX_COUNT; i++)
      indices[i] = quad[i];
}

/* RGBA8 as the little-endian uint32 a readback holds. */
static inline uint32_t
tri_expected(uint32_t x, uint32_t y, int scene)
{
   const int in_y = y >= 16 && y < 48;
   if (scene == TRI_SCENE_DEPTH) {
      if (!in_y || x < 8 || x >= 56)
         return 0xff000000u;
      return x >= 24 && x < 40 ? 0xff00ff00u : 0xff00ffffu;
   }
   if (scene == TRI_SCENE_MULTI) {
      if (!in_y || x < 8 || x >= 56)
         return 0xff000000u;
      return x < 24 ? 0xff00ffffu : x < 40 ? 0xff00ff00u : 0xff0000ffu;
   }
   if (scene == TRI_SCENE_STATE) {
      if (!in_y || x < 8 || x >= 56)
         return 0xff000000u;
      return x < 40 ? 0xff00ff00u : 0xff0000ffu;
   }
   if (scene == TRI_SCENE_STENCIL) {
      if (in_y && x >= 8 && x < 24)
         return 0xff00ff00u;
      return in_y && x >= 40 && x < 56 ? 0xff0000ffu : 0xff000000u;
   }
   const int in_x = (x >= 8 && x < 24) || (x >= 40 && x < 56);
   return in_x && in_y ? 0xff0000ffu : 0xff000000u;
}

/* The plain scene into a render target of another format (kgp_triangle
 * and vk_baseline triangle, followed by the name). Every value drawn is 0
 * or 1, so each format has one exact encoding of the picture.
 */
enum {
   TRI_TARGET_RGBA8 = 0, TRI_TARGET_SRGB8, TRI_TARGET_RGBA16F,
   TRI_TARGET_RGBA32F, TRI_TARGET_R32F, TRI_TARGET_COUNT,
};
static const struct {
   const char *name;
   uint32_t bpp;
} tri_targets[TRI_TARGET_COUNT] = {
   { "rgba8", 4 }, { "srgb", 4 }, { "rgba16f", 8 }, { "rgba32f", 16 },
   { "r32f", 4 },
};

/* TRI_TARGET_RGBA8 for anything that is not a format's name. */
static inline int
tri_target_from_arg(const char *arg)
{
   for (int t = 1; arg != NULL && t < TRI_TARGET_COUNT; t++) {
      if (strcmp(arg, tri_targets[t].name) == 0)
         return t;
   }
   return TRI_TARGET_RGBA8;
}

static inline int
tri_check_target(const char *what, const void *got, int target)
{
   const uint32_t bpp = tri_targets[target].bpp;

   for (uint32_t y = 0; y < TRI_H; y++) {
      for (uint32_t x = 0; x < TRI_W; x++) {
         const uint32_t rgba8 = tri_expected(x, y, TRI_SCENE_PLAIN);
         float c[4];
         for (unsigned i = 0; i < 4; i++)
            c[i] = (rgba8 >> (8 * i)) & 0xff ? 1.0f : 0.0f;

         uint8_t want[16];
         if (target == TRI_TARGET_RGBA16F) {
            for (unsigned i = 0; i < 4; i++) {
               const uint16_t half = c[i] != 0.0f ? 0x3c00 : 0;
               memcpy(want + 2 * i, &half, 2);
            }
         } else if (target == TRI_TARGET_RGBA32F ||
                    target == TRI_TARGET_R32F) {
            memcpy(want, c, bpp);
         } else {
            memcpy(want, &rgba8, 4);
         }

         const uint8_t *texel =
            (const uint8_t *)got + ((size_t)y * TRI_W + x) * bpp;
         if (memcmp(texel, want, bpp) != 0) {
            fprintf(stderr, "%s (%s): pixel (%u, %u) =", what,
                    tri_targets[target].name, x, y);
            for (uint32_t i = 0; i < bpp; i++)
               fprintf(stderr, " %02x", texel[i]);
            fprintf(stderr, ", expected");
            for (uint32_t i = 0; i < bpp; i++)
               fprintf(stderr, " %02x", want[i]);
            fputc('\n', stderr);
            return 1;
         }
      }
   }
   return 0;
}

/* The "msaa" variant: the plain scene into a four-sample target, moved
 * half a pixel to the right so that the quads' vertical edges cut through
 * pixels. With the standard sample locations two of the four samples lie
 * on each side of a pixel's centre line, so the edge columns resolve to
 * half red:
 *
 *    x = 8, 24, 40, 56   half red (0x7f or 0x80)
 *    (8, 24), (40, 56)   red
 *
 * One sample per pixel, or a target read as if it had one, gives no half
 * columns. KoreGP reads the target back with a shader that averages the
 * samples (tri_resolve.comp); the Vulkan twin uses a resolve attachment.
 */
#define TRI_MSAA_SAMPLES 4u
#define TRI_MSAA_SHIFT (1.0f / 64.0f)

/* tri_resolve.comp's root struct */
struct tri_resolve_root {
   uint64_t dst;
   uint32_t width;
   uint32_t samples;
   uint32_t pad[12];
};

static inline int
tri_check_msaa(const char *what, const uint32_t *got)
{
   for (uint32_t y = 0; y < TRI_H; y++) {
      for (uint32_t x = 0; x < TRI_W; x++) {
         const int in_y = y >= 16 && y < 48;
         const int edge = x == 8 || x == 24 || x == 40 || x == 56;
         const int inside = (x > 8 && x < 24) || (x > 40 && x < 56);
         const uint32_t px = got[y * TRI_W + x];
         const int ok = !in_y || !(edge || inside) ? px == 0xff000000u :
                        inside ? px == 0xff0000ffu :
                        px == 0xff00007fu || px == 0xff000080u;
         if (!ok) {
            fprintf(stderr, "%s: pixel (%u, %u) = %08x, expected %s\n",
                    what, x, y, px,
                    !in_y || !(edge || inside) ? "ff000000" :
                    inside ? "ff0000ff" : "ff00007f or ff000080");
            return 1;
         }
      }
   }
   return 0;
}

/* The mrt variant's second target. */
static inline int
tri_check_mrt1(const char *what, const uint32_t *got)
{
   for (uint32_t y = 0; y < TRI_H; y++) {
      for (uint32_t x = 0; x < TRI_W; x++) {
         const int quad = tri_expected(x, y, TRI_SCENE_PLAIN) != 0xff000000u;
         const uint32_t want = quad ? 0xff00ff00u : 0xff0000ffu;
         if (got[y * TRI_W + x] != want) {
            fprintf(stderr, "%s: second target, pixel (%u, %u) = %08x, "
                    "expected %08x\n", what, x, y, got[y * TRI_W + x], want);
            return 1;
         }
      }
   }
   return 0;
}

static inline int
tri_check(const char *what, const uint32_t *got, uint32_t row_texels,
          int scene)
{
   for (uint32_t y = 0; y < TRI_H; y++) {
      for (uint32_t x = 0; x < TRI_W; x++) {
         const uint32_t want = tri_expected(x, y, scene);
         if (got[y * row_texels + x] != want) {
            fprintf(stderr, "%s: pixel (%u, %u) = %08x, expected %08x\n",
                    what, x, y, got[y * row_texels + x], want);
            /* A window on hardware is scarce: show the whole image, every
             * second row. '.' clear colour, '#' red, 'g' green, 'y' yellow,
             * '0' untouched memory, '?' anything else.
             */
            for (uint32_t r = 0; r < TRI_H; r += 2) {
               for (uint32_t c = 0; c < TRI_W; c++) {
                  const uint32_t v = got[r * row_texels + c];
                  fputc(v == 0xff000000u ? '.' : v == 0xff0000ffu ? '#' :
                        v == 0xff00ff00u ? 'g' : v == 0xff00ffffu ? 'y' :
                        v == 0 ? '0' : '?', stderr);
               }
               fputc('\n', stderr);
            }
            return 1;
         }
      }
   }
   return 0;
}

#endif
