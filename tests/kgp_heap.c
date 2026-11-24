/* Textures and the texture heap (plan N2), end to end:
 *
 *  - two textures laid over kgpMalloc memory; one filled from the CPU, one
 *    by a shader through storage-image access;
 *  - their descriptors written into the heap BY A SHADER, then used by the
 *    next dispatches after a KGP_HAZARD_DESCRIPTORS barrier;
 *  - every texel read back through the heap and checked.
 *
 *    kgp_heap            the hardware's tiled layout
 *    kgp_heap linear     KGP_TEXTURE_LINEAR
 *
 * On hardware the two modes together are the probe for delta D4: tiled
 * textures here sit in ordinary memory, which NVK never does.
 */
#include "kgp_test.h"

#include "heap_copy_spv.h"
#include "heap_fetch_spv.h"
#include "heap_sample_spv.h"
#include "heap_store_spv.h"

#include <string.h>

#define W 96u
#define H 40u
#define HEAP_ENTRIES 16u
#define SLOT_A 5u
#define SLOT_B 9u

/* Root structs match the shaders (scalar layout), padded to 64 bytes. */
struct copy_root {
   uint64_t src;
   uint64_t heap;
   uint32_t src_index;
   uint32_t dst_index;
   uint32_t pad[10];
};
struct fetch_root {
   uint64_t dst;
   uint32_t texture_index;
   uint32_t width;
   uint32_t height;
   uint32_t pad[11];
};
struct store_root {
   uint32_t texture_index;
   uint32_t width;
   uint32_t height;
   uint32_t pad[13];
};
struct sample_root {
   uint64_t dst;
   uint32_t texture_index;
   uint32_t sampler_index;
   uint32_t width;
   uint32_t height;
   float offset_x;
   uint32_t pad[9];
};
_Static_assert(sizeof(struct copy_root) == 64 &&
               sizeof(struct fetch_root) == 64 &&
               sizeof(struct store_root) == 64 &&
               sizeof(struct sample_root) == 64, "root structs are 64 bytes");

#define SAMPLER_ENTRIES 4u
#define SAMPLER_NEAREST 1u
#define SAMPLER_LINEAR 2u

/* RGBA8 texels as the little-endian uint32 packUnorm4x8 produces. */
static uint32_t
uploaded_texel(uint32_t x, uint32_t y)
{
   return (x & 255) | (y & 255) << 8 | ((x + y) & 255) << 16 |
          ((255 - x) & 255) << 24;
}

static uint32_t
stored_texel(uint32_t x, uint32_t y)
{
   return (y & 255) | (x & 255) << 8 | ((x ^ y) & 255) << 16 | 128u << 24;
}

static int
check(const char *what, const uint32_t *got, uint32_t (*expect)(uint32_t, uint32_t))
{
   for (uint32_t y = 0; y < H; y++) {
      for (uint32_t x = 0; x < W; x++) {
         if (got[y * W + x] != expect(x, y)) {
            fprintf(stderr, "heap: %s texel (%u, %u) = %08x, expected %08x\n",
                    what, x, y, got[y * W + x], expect(x, y));
            return 1;
         }
      }
   }
   return 0;
}

/* Linear filtering half a texel to the right of each texel centre: every
 * channel must lie between that texel's value and its right neighbour's
 * (the edge clamps), whatever the hardware's rounding.
 */
static int
check_between(const uint32_t *got)
{
   for (uint32_t y = 0; y < H; y++) {
      for (uint32_t x = 0; x < W; x++) {
         const uint32_t a = uploaded_texel(x, y);
         const uint32_t b = uploaded_texel(x + 1 < W ? x + 1 : x, y);
         for (unsigned c = 0; c < 4; c++) {
            const uint32_t ca = (a >> (c * 8)) & 255, cb = (b >> (c * 8)) & 255;
            const uint32_t cg = (got[y * W + x] >> (c * 8)) & 255;
            if (cg < (ca < cb ? ca : cb) || cg > (ca > cb ? ca : cb)) {
               fprintf(stderr, "heap: linear sample (%u, %u) = %08x is not "
                       "between %08x and %08x\n", x, y, got[y * W + x], a, b);
               return 1;
            }
         }
      }
   }
   return 0;
}

int
main(int argc, char **argv)
{
   const bool linear = argc > 1 && strcmp(argv[1], "linear") == 0;
   const char *mode = linear ? "linear" : "tiled";

   KgpDevice dev = kgpCreateDevice(0);
   CHECK(dev);
   KgpQueue queue = kgpGetQueue(dev);

   /* Layout queries, checkable without a GPU. */
   const KgpTextureDesc desc = {
      .format = KGP_FORMAT_RGBA8_UNORM,
      .width = W,
      .height = H,
      .flags = linear ? KGP_TEXTURE_LINEAR : 0,
   };
   KgpSizeAlign sa;
   CHECK(kgpTextureSizeAlign(dev, &desc, &sa));
   CHECK(sa.size >= W * H * 4 && sa.align >= 32 &&
         (sa.align & (sa.align - 1)) == 0);
   printf("heap: %ux%u RGBA8 %s: %llu bytes, align %llu\n", W, H, mode,
          (unsigned long long)sa.size, (unsigned long long)sa.align);

   fprintf(stderr, "(the errors logged by this test are expected)\n");
   KgpSizeAlign bad_sa;
   KgpTextureDesc bad = desc;
   bad.format = (KGP_FORMAT)1234;
   CHECK(!kgpTextureSizeAlign(dev, &bad, &bad_sa));
   bad = desc;
   bad.width = 0;
   CHECK(!kgpTextureSizeAlign(dev, &bad, &bad_sa));
   bad = desc;
   bad.mipCount = 20;
   CHECK(!kgpTextureSizeAlign(dev, &bad, &bad_sa));

   /* Two textures, each over its own caller-owned memory. */
   void *mem_a = kgpMalloc(dev, sa.size, sa.align, KGP_MEMORY_DEFAULT);
   void *mem_b = kgpMalloc(dev, sa.size, sa.align, KGP_MEMORY_DEFAULT);
   CHECK(mem_a && mem_b);
   memset(mem_b, 0, sa.size);

   void *gpu_a = kgpHostToDevicePointer(dev, mem_a);
   void *gpu_b = kgpHostToDevicePointer(dev, mem_b);
   CHECK(!kgpCreateTexture(dev, &desc, (char *)gpu_a + 4));   /* misaligned */
   KgpTexture tex_a = kgpCreateTexture(dev, &desc, gpu_a);
   KgpTexture tex_b = kgpCreateTexture(dev, &desc, gpu_b);
   CHECK(tex_a && tex_b);

   uint32_t *pixels = malloc(W * H * sizeof(uint32_t));
   CHECK(pixels);
   for (uint32_t y = 0; y < H; y++)
      for (uint32_t x = 0; x < W; x++)
         pixels[y * W + x] = uploaded_texel(x, y);
   CHECK(kgpWriteTexture(tex_a, mem_a, 0, 0, pixels, W * 4));
   free(pixels);

   /* Descriptors go to a staging array first; a shader moves them. */
   KgpDescriptor256 *staging =
      kgpMalloc(dev, 2 * sizeof(*staging), 64, KGP_MEMORY_DEFAULT);
   KgpDescriptor256 *heap =
      kgpMalloc(dev, HEAP_ENTRIES * sizeof(*heap), 64, KGP_MEMORY_DEFAULT);
   CHECK(staging && heap);
   memset(heap, 0, HEAP_ENTRIES * sizeof(*heap));   /* null textures */

   KgpDescriptor256 desc_a, desc_b;
   CHECK(kgpTextureViewDescriptor(tex_a, NULL, &desc_a));
   CHECK(kgpTextureViewDescriptor(tex_b, NULL, &desc_b));
   CHECK(memcmp(&desc_a, &desc_b, sizeof(desc_a)) != 0);
   CHECK(!kgpTextureViewDescriptor(tex_a, &(KgpViewDesc) { .baseMip = 1 },
                                   &desc_b));
   CHECK(kgpTextureViewDescriptor(tex_b, NULL, &desc_b));
   staging[0] = desc_a;
   staging[1] = desc_b;

   /* The sampler heap is filled from the CPU: it is plain memory too. */
   KgpDescriptor256 *samplers =
      kgpMalloc(dev, SAMPLER_ENTRIES * sizeof(*samplers), 64,
                KGP_MEMORY_DEFAULT);
   CHECK(samplers);
   memset(samplers, 0, SAMPLER_ENTRIES * sizeof(*samplers));
   KgpDescriptor256 samp_nearest, samp_linear;
   CHECK(kgpSamplerDescriptor(dev, &(KgpSamplerDesc) {
      .addressU = KGP_ADDRESS_CLAMP_TO_EDGE,
      .addressV = KGP_ADDRESS_CLAMP_TO_EDGE,
   }, &samp_nearest));
   CHECK(kgpSamplerDescriptor(dev, &(KgpSamplerDesc) {
      .minFilter = KGP_FILTER_LINEAR,
      .magFilter = KGP_FILTER_LINEAR,
      .addressU = KGP_ADDRESS_CLAMP_TO_EDGE,
      .addressV = KGP_ADDRESS_CLAMP_TO_EDGE,
   }, &samp_linear));
   CHECK(memcmp(&samp_nearest, &samp_linear, sizeof(samp_linear)) != 0);
   CHECK(!kgpSamplerDescriptor(dev, &(KgpSamplerDesc) {
      .maxAnisotropy = 64.0f,
   }, &samplers[0]));
   samplers[SAMPLER_NEAREST] = samp_nearest;
   samplers[SAMPLER_LINEAR] = samp_linear;

   uint32_t *out_a = kgpMalloc(dev, W * H * 4, 64, KGP_MEMORY_READBACK);
   uint32_t *out_b = kgpMalloc(dev, W * H * 4, 64, KGP_MEMORY_READBACK);
   uint32_t *out_n = kgpMalloc(dev, W * H * 4, 64, KGP_MEMORY_READBACK);
   uint32_t *out_l = kgpMalloc(dev, W * H * 4, 64, KGP_MEMORY_READBACK);
   CHECK(out_a && out_b && out_n && out_l);
   memset(out_a, 0, W * H * 4);
   memset(out_b, 0, W * H * 4);
   memset(out_n, 0, W * H * 4);
   memset(out_l, 0, W * H * 4);

   struct sample_root *sample_roots =
      kgpMalloc(dev, 2 * sizeof(*sample_roots), 64, KGP_MEMORY_DEFAULT);
   CHECK(sample_roots);
   sample_roots[0] = (struct sample_root) {
      .dst = (uintptr_t)kgpHostToDevicePointer(dev, out_n),
      .texture_index = SLOT_A, .sampler_index = SAMPLER_NEAREST,
      .width = W, .height = H, .offset_x = 0.0f,
   };
   sample_roots[1] = (struct sample_root) {
      .dst = (uintptr_t)kgpHostToDevicePointer(dev, out_l),
      .texture_index = SLOT_A, .sampler_index = SAMPLER_LINEAR,
      .width = W, .height = H, .offset_x = 0.5f,
   };
   struct sample_root *sample_gpu = kgpHostToDevicePointer(dev, sample_roots);

   struct copy_root *copy_roots =
      kgpMalloc(dev, 2 * sizeof(*copy_roots), 64, KGP_MEMORY_DEFAULT);
   struct fetch_root *fetch_roots =
      kgpMalloc(dev, 2 * sizeof(*fetch_roots), 64, KGP_MEMORY_DEFAULT);
   struct store_root *store_root =
      kgpMalloc(dev, sizeof(*store_root), 64, KGP_MEMORY_DEFAULT);
   CHECK(copy_roots && fetch_roots && store_root);

   const uint64_t staging_gpu = (uintptr_t)kgpHostToDevicePointer(dev, staging);
   const uint64_t heap_gpu = (uintptr_t)kgpHostToDevicePointer(dev, heap);
   copy_roots[0] = (struct copy_root) {
      .src = staging_gpu, .heap = heap_gpu, .src_index = 0, .dst_index = SLOT_A,
   };
   copy_roots[1] = (struct copy_root) {
      .src = staging_gpu, .heap = heap_gpu, .src_index = 1, .dst_index = SLOT_B,
   };
   fetch_roots[0] = (struct fetch_root) {
      .dst = (uintptr_t)kgpHostToDevicePointer(dev, out_a),
      .texture_index = SLOT_A, .width = W, .height = H,
   };
   fetch_roots[1] = (struct fetch_root) {
      .dst = (uintptr_t)kgpHostToDevicePointer(dev, out_b),
      .texture_index = SLOT_B, .width = W, .height = H,
   };
   *store_root = (struct store_root) {
      .texture_index = SLOT_B, .width = W, .height = H,
   };
   struct copy_root *copy_gpu = kgpHostToDevicePointer(dev, copy_roots);
   struct fetch_root *fetch_gpu = kgpHostToDevicePointer(dev, fetch_roots);

   KgpPipeline copy = kgpCreateComputePipeline(
      dev, (KgpByteSpan) { heap_copy_spv, sizeof(heap_copy_spv) }, NULL);
   KgpPipeline fetch = kgpCreateComputePipeline(
      dev, (KgpByteSpan) { heap_fetch_spv, sizeof(heap_fetch_spv) }, NULL);
   KgpPipeline store = kgpCreateComputePipeline(
      dev, (KgpByteSpan) { heap_store_spv, sizeof(heap_store_spv) }, NULL);
   KgpPipeline sample = kgpCreateComputePipeline(
      dev, (KgpByteSpan) { heap_sample_spv, sizeof(heap_sample_spv) }, NULL);
   KgpSemaphore done = kgpCreateSemaphore(dev, 0);
   CHECK(copy && fetch && store && sample && done);

   const KgpUvec3 image_groups = { (W + 7) / 8, (H + 7) / 8, 1 };

   KgpCommandBuffer cb = kgpStartCommandRecording(queue);
   CHECK(cb);
   kgpSetActiveTextureHeapPtr(cb, (void *)(uintptr_t)heap_gpu, HEAP_ENTRIES);
   kgpSetActiveSamplerHeapPtr(cb, kgpHostToDevicePointer(dev, samplers),
                              SAMPLER_ENTRIES);

   /* The GPU fills in two heap entries... */
   kgpSetPipeline(cb, copy);
   kgpDispatch(cb, &copy_gpu[0], (KgpUvec3) { 1, 1, 1 });
   kgpDispatch(cb, &copy_gpu[1], (KgpUvec3) { 1, 1, 1 });
   kgpBarrier(cb, KGP_STAGE_COMPUTE, KGP_STAGE_COMPUTE, KGP_HAZARD_DESCRIPTORS);

   /* ...writes texture B through one of them... */
   kgpSetPipeline(cb, store);
   kgpDispatch(cb, kgpHostToDevicePointer(dev, store_root), image_groups);
   kgpBarrier(cb, KGP_STAGE_COMPUTE, KGP_STAGE_COMPUTE, KGP_HAZARD_NONE);

   /* ...and reads both textures back through them. */
   kgpSetPipeline(cb, fetch);
   kgpDispatch(cb, &fetch_gpu[0], image_groups);
   kgpDispatch(cb, &fetch_gpu[1], image_groups);

   /* Filtered sampling of texture A: nearest at texel centres, linear
    * half way to the next texel.
    */
   kgpSetPipeline(cb, sample);
   kgpDispatch(cb, &sample_gpu[0], image_groups);
   kgpDispatch(cb, &sample_gpu[1], image_groups);

   CHECK(kgpSubmit(queue, &cb, 1, done, 1));
   CHECK(kgpWaitSemaphore(dev, done, 1, KGP_TEST_TIMEOUT_NS));

   int ret = 0;
   if (kgp_test_offline()) {
      printf("heap: %s recorded and submitted (offline, results not checked)\n",
             mode);
   } else {
      CHECK(memcmp(&heap[SLOT_A], &desc_a, sizeof(desc_a)) == 0);
      CHECK(memcmp(&heap[SLOT_B], &desc_b, sizeof(desc_b)) == 0);
      ret |= check("CPU-uploaded", out_a, uploaded_texel);
      ret |= check("shader-stored", out_b, stored_texel);
      ret |= check("nearest-sampled", out_n, uploaded_texel);
      ret |= check_between(out_l);
      if (ret == 0)
         printf("heap: %s: descriptors written by a shader; fetch, storage "
                "write, nearest and linear sampling of %u texels correct\n",
                mode, W * H);
   }

   kgpDestroySemaphore(dev, done);
   kgpDestroyPipeline(dev, sample);
   kgpFree(dev, sample_roots);
   kgpFree(dev, out_l);
   kgpFree(dev, out_n);
   kgpFree(dev, samplers);
   kgpDestroyPipeline(dev, store);
   kgpDestroyPipeline(dev, fetch);
   kgpDestroyPipeline(dev, copy);
   kgpDestroyTexture(dev, tex_b);
   kgpDestroyTexture(dev, tex_a);
   kgpFree(dev, store_root);
   kgpFree(dev, fetch_roots);
   kgpFree(dev, copy_roots);
   kgpFree(dev, out_b);
   kgpFree(dev, out_a);
   kgpFree(dev, heap);
   kgpFree(dev, staging);
   kgpFree(dev, mem_b);
   kgpFree(dev, mem_a);
   kgpDestroyDevice(dev);
   return ret;
}
