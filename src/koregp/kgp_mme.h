/*
 * KoreGP's own macros for the 3D engine's macro processor (MME): the ones
 * that are not NVK's. They are loaded into the slots after NVK's.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef KGP_MME_H
#define KGP_MME_H

#include "nouveau/vulkan/nvk_mme.h"

enum kgp_mme {
   KGP_MME_DRAW_INDEXED_MULTI = NVK_MME_COUNT,
   KGP_MME_END,
};

/* Parameters, in order: the address of the KgpDrawIndexedArgs array (hi,
 * lo), the address of the draw count (hi, lo), the maximum draw count, the
 * vertex-stage root pointer of draw 0 (hi, lo) and its stride, the
 * pixel-stage root pointer of draw 0 (hi, lo) and its stride.
 */
void kgp_mme_draw_indexed_multi(struct mme_builder *b);

#endif /* KGP_MME_H */
