/* Shared by tests/kgp_copy.c and its Vulkan twin (vk_baseline copy), so the
 * two record the same copies and their pushbufs can be compared.
 */
#ifndef COPY_PATTERN_H
#define COPY_PATTERN_H

#include <stdint.h>
#include <stdio.h>

#define COPY_W 96u
#define COPY_H 40u
#define COPY_BYTES (COPY_W * COPY_H * 4)
/* The copy back out of the texture uses rows wider than the image. */
#define COPY_OUT_ROW_TEXELS (COPY_W + 8)
#define COPY_OUT_BYTES (COPY_OUT_ROW_TEXELS * COPY_H * 4)

static inline uint32_t
copy_texel(uint32_t x, uint32_t y)
{
   return (x & 255) | (y & 255) << 8 | ((x * 3 + y) & 255) << 16 |
          ((255 - y) & 255) << 24;
}

static inline void
copy_fill(uint32_t *texels)
{
   for (uint32_t y = 0; y < COPY_H; y++)
      for (uint32_t x = 0; x < COPY_W; x++)
         texels[y * COPY_W + x] = copy_texel(x, y);
}

static inline int
copy_check(const char *what, const uint32_t *got, uint32_t row_texels)
{
   for (uint32_t y = 0; y < COPY_H; y++) {
      for (uint32_t x = 0; x < COPY_W; x++) {
         if (got[y * row_texels + x] != copy_texel(x, y)) {
            fprintf(stderr, "copy: %s texel (%u, %u) = %08x, expected %08x\n",
                    what, x, y, got[y * row_texels + x], copy_texel(x, y));
            return 1;
         }
      }
   }
   return 0;
}

#endif
