/* CPU oracle for shaders/scratch.comp, shared by the KoreGP and Vulkan
 * versions of the test.
 */
#ifndef SCRATCH_ORACLE_H
#define SCRATCH_ORACLE_H

#include <stdint.h>

#define SCRATCH_COUNT 50000u
#define SCRATCH_GROUPS ((SCRATCH_COUNT + 63) / 64)

/* Matches ScratchRoot (scalar layout), padded to 64 bytes so an array of
 * them keeps every root pointer 64-byte aligned.
 */
struct scratch_root {
   uint64_t src;
   uint64_t dst;
   uint32_t count;
   uint32_t pad[11];
};
_Static_assert(sizeof(struct scratch_root) == 64, "root structs are 64 bytes");

static const uint32_t scratch_table[64] = {
     2,   3,   5,   7,  11,  13,  17,  19,
    23,  29,  31,  37,  41,  43,  47,  53,
    59,  61,  67,  71,  73,  79,  83,  89,
    97, 101, 103, 107, 109, 113, 127, 131,
   137, 139, 149, 151, 157, 163, 167, 173,
   179, 181, 191, 193, 197, 199, 211, 223,
   227, 229, 233, 239, 241, 251, 257, 263,
   269, 271, 277, 281, 283, 293, 307, 311,
};

static inline uint32_t
scratch_input(uint32_t i)
{
   return (i * 2246822519u) >> 12;
}

static inline uint32_t
scratch_expected(uint32_t i, uint32_t num_groups)
{
   const uint32_t s = scratch_input(i);

   uint32_t tmp[64];
   for (uint32_t k = 0; k < 64; k++)
      tmp[k] = s * scratch_table[(k + i) & 63] + k;

   uint32_t acc = num_groups;
   for (uint32_t k = 0; k < 64; k++)
      acc += tmp[(k * 7 + s) & 63];
   return acc;
}

#endif
