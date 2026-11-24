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
 */
#ifndef TRI_SCENE_H
#define TRI_SCENE_H

#include <stdint.h>
#include <stdio.h>

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
};

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
   if (scene == TRI_SCENE_STENCIL) {
      if (in_y && x >= 8 && x < 24)
         return 0xff00ff00u;
      return in_y && x >= 40 && x < 56 ? 0xff0000ffu : 0xff000000u;
   }
   const int in_x = (x >= 8 && x < 24) || (x >= 40 && x < 56);
   return in_x && in_y ? 0xff0000ffu : 0xff000000u;
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
