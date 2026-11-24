/*
 * Runs KoreGP's multi-draw macro (kgp_mme.c) on Mesa's simulator of the
 * Turing macro processor and prints every method it emits; the simulator
 * does the printing. tests/mme_multi_sim.sh compares the output with
 * tests/mme_multi_sim.expected.
 *
 * Three draws are asked for by the count in memory, with a maximum of eight
 * and four argument records present: a fourth draw in the output means the
 * count was not honoured.
 *
 * SPDX-License-Identifier: MIT
 */
#include "kgp_mme.h"
#include "kgp_private.h"

#include "mme_builder.h"
#include "mme_tu104.h"
#include "mme_tu104_sim.h"
#include "nvidia/classes/clc597.h"

#include <stdio.h>
#include <stdlib.h>

int
main(void)
{
   const struct nv_device_info info = { .cls_eng3d = TURING_A };

   struct mme_builder b;
   mme_builder_init(&b, &info);
   kgp_mme_draw_indexed_multi(&b);
   size_t size;
   uint32_t *macro = mme_builder_finish(&b, &size);
   if (macro == NULL)
      return 1;

   const uint32_t inst_count = size / (3 * sizeof(uint32_t));
   struct mme_tu104_inst *insts = calloc(inst_count, sizeof(*insts));
   mme_tu104_decode(insts, macro, inst_count);

   uint32_t count = 3;
   KgpDrawIndexedArgs args[4] = {
      { .indexCount = 6, .instanceCount = 2 },
      { .indexCount = 6, .instanceCount = 1, .firstInstance = 1 },
      /* firstInstance never returns to an earlier value here: the simulator
       * does not remember what a macro wrote to an ordinary register, so
       * the "only if it changed" test of kgp_mme_set_root0 would see 0.
       */
      { .indexCount = 3, .instanceCount = 1, .firstIndex = 3,
        .vertexOffset = 7, .firstInstance = 2 },
      { .indexCount = 99, .instanceCount = 99 },
   };
   struct mme_tu104_sim_mem mems[] = {
      { .addr = 0x1100000000ull, .data = args, .size = sizeof(args) },
      { .addr = 0x2200000000ull, .data = &count, .size = sizeof(count) },
   };
   /* Root pointers that carry into the high word on the second step. */
   const uint32_t params[] = {
      0x11, 0x00000000,          /* args */
      0x22, 0x00000000,          /* count */
      8,                         /* maximum */
      0x33, 0xffffff80, 0x40,    /* vertex roots, stride */
      0x44, 0x00001000, 0x100,   /* pixel roots, stride */
   };

   printf("%u instructions\n", inst_count);
   mme_tu104_sim(inst_count, insts, ARRAY_SIZE(params), params,
                 ARRAY_SIZE(mems), mems, info.cls_eng3d);

   free(insts);
   free(macro);
   return 0;
}
