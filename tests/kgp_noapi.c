/* The proposal's M2 exit test: the no_api example set, ported.
 *
 * no_api (github.com/UnNabbo/no_api, Jai, on Vulkan) transliterates the
 * prototype header of the "No Graphics API" post and has three examples.
 * Each is a variant here, call for call where KoreGP has the call:
 *
 *    kgp_noapi [triangle]   hello_triangle.jai: one indexed triangle, blue
 *                           on a red clear
 *    kgp_noapi compute      hello_compute.jai: a compute shader writes a
 *                           gradient into the backbuffer
 *    kgp_noapi playground   playground.jai: an image uploaded with a GPU
 *                           copy and put in the heap, the gradient, then a
 *                           triangle drawn over it, a frame semaphore
 *
 * A second argument "display" presents on the monitor (a window opened by
 * w0.sh present); without it the swapchain is headless. The first frame is
 * read back and checked.
 *
 * What is not the original's:
 *  - Shaders are GLSL, not Slang (tests/shaders/noapi.*), same logic.
 *  - no_api renders to a texture of its own and gpu_present copies it to
 *    the swapchain; here the swapchain's texture is the backbuffer.
 *  - no_api has a texture heap and a read-write texture heap; KoreGP has
 *    one heap, so the backbuffer is entry 1 of it.
 *  - The playground's image is generated, not loaded from a PNG. Like the
 *    original, nothing samples it.
 *  - No window, no input: a fixed number of frames.
 */
#include "kgp_test.h"

#include "noapi_cs_spv.h"
#include "noapi_ps_spv.h"
#include "noapi_vs_spv.h"

#include <string.h>

#define SPAN(a) ((KgpByteSpan) { a, sizeof(a) })

/* Headless size: a multiple of the compute shader's 16x16 workgroup. */
#define W 256
#define H 144
#define IMAGE_W 64
#define IMAGE_H 64

/* Vertex in noapi.vert (scalar layout). */
struct vertex {
   float position[3];
   float col[3];
   float uvs[2];
};

/* A BGRA8 texel as the bytes r, g, b, a. */
static void
unpack(uint32_t bgra, int rgba[4])
{
   rgba[0] = (bgra >> 16) & 0xff;
   rgba[1] = (bgra >> 8) & 0xff;
   rgba[2] = bgra & 0xff;
   rgba[3] = bgra >> 24;
}

static int
expect(const uint32_t *out, uint32_t width, uint32_t x, uint32_t y,
       int r, int g, int b, const char *what)
{
   int got[4];
   unpack(out[(size_t)y * width + x], got);
   /* One step of slack: the gradient is a float rounded to 8 bits. */
   if (abs(got[0] - r) > 1 || abs(got[1] - g) > 1 || abs(got[2] - b) > 1 ||
       got[3] != 255) {
      fprintf(stderr, "noapi: %s: pixel (%u, %u) is (%d, %d, %d, %d), "
              "expected (%d, %d, %d, 255)\n", what, x, y, got[0], got[1],
              got[2], got[3], r, g, b);
      return 1;
   }
   return 0;
}

/* compute.slang's value at a pixel it writes. */
static int
expect_gradient(const uint32_t *out, uint32_t width, uint32_t height,
                uint32_t x, uint32_t y)
{
   return expect(out, width, x, y,
                 (int)(255.0f * (float)x / (float)width + 0.5f),
                 (int)(255.0f * (float)y / (float)height + 0.5f), 0,
                 "gradient");
}

int
main(int argc, char **argv)
{
   const char *example = argc > 1 ? argv[1] : "triangle";
   const bool triangle = strcmp(example, "triangle") == 0;
   const bool compute = strcmp(example, "compute") == 0;
   const bool playground = strcmp(example, "playground") == 0;
   const bool display = argc > 2 && strcmp(argv[2], "display") == 0;
   if (!triangle && !compute && !playground) {
      fprintf(stderr, "usage: kgp_noapi triangle|compute|playground "
              "[display]\n");
      return 2;
   }
   const unsigned frames = display ? 120 : 3;

   KgpDevice dev = kgpCreateDevice(0);
   CHECK(dev);
   KgpQueue queue = kgpGetQueue(dev);

   KgpSwapchainInfo info;
   KgpSwapchain swapchain = kgpCreateSwapchain(
      dev, &(KgpSwapchainDesc) {
         .target = display ? KGP_PRESENT_DISPLAY : KGP_PRESENT_HEADLESS,
         .width = W, .height = H,
      }, &info);
   CHECK(swapchain);
   const uint32_t width = info.width, height = info.height;

   uint32_t *out = kgpMalloc(dev, (size_t)width * height * 4, 64,
                             KGP_MEMORY_READBACK);
   CHECK(out);
   memset(out, 0, (size_t)width * height * 4);

   /* Graphics: three vertices and three indices behind plain pointers. */
   KgpPipeline gfx = NULL;
   struct vertex *vertices = NULL;
   uint32_t *indices = NULL;
   if (triangle || playground) {
      gfx = kgpCreateGraphicsPipeline(
         dev, SPAN(noapi_vs_spv), SPAN(noapi_ps_spv),
         &(KgpRasterDesc) {
            .colorTargetCount = 1,
            .colorFormats = { info.format },
         }, NULL);
      vertices = kgpMalloc(dev, 3 * sizeof(*vertices), 64, KGP_MEMORY_DEFAULT);
      indices = kgpMalloc(dev, 3 * sizeof(*indices), 64, KGP_MEMORY_DEFAULT);
      CHECK(gfx && vertices && indices);
      static const float pos[3][2] = { { 0, -0.5f }, { 0.5f, 0.5f },
                                       { -0.5f, 0.5f } };
      /* hello_triangle: all blue. playground: yellow, red, magenta. */
      static const float play_col[3][3] = { { 1, 1, 0 }, { 1, 0, 0 },
                                            { 1, 0, 1 } };
      for (unsigned i = 0; i < 3; i++) {
         vertices[i] = (struct vertex) {
            .position = { pos[i][0], pos[i][1], 0 },
            .col = { playground ? play_col[i][0] : 0,
                     playground ? play_col[i][1] : 0,
                     playground ? play_col[i][2] : 1 },
         };
         indices[i] = i;
      }
   }

   /* Compute: the gradient, into heap entry 1. */
   KgpPipeline cs = NULL;
   KgpDescriptor256 *heap = NULL;
   if (compute || playground) {
      cs = kgpCreateComputePipeline(dev, SPAN(noapi_cs_spv), NULL);
      heap = kgpMalloc(dev, 2 * sizeof(*heap), 64, KGP_MEMORY_DEFAULT);
      CHECK(cs && heap);
      memset(heap, 0, 2 * sizeof(*heap));
   }
   void *heap_gpu = heap ? kgpHostToDevicePointer(dev, heap) : NULL;

   KgpSemaphore frame_semaphore = kgpCreateSemaphore(dev, 0);
   CHECK(frame_semaphore);
   uint64_t submitted = 0;

   /* playground: upload an image with a GPU copy and put it in the heap. */
   KgpTexture image = NULL;
   void *image_mem = NULL;
   uint32_t *upload = NULL;
   if (playground) {
      const KgpTextureDesc desc = {
         .format = KGP_FORMAT_RGBA8_UNORM,
         .width = IMAGE_W, .height = IMAGE_H,
      };
      KgpSizeAlign sa;
      CHECK(kgpTextureSizeAlign(dev, &desc, &sa));
      image_mem = kgpMalloc(dev, sa.size, sa.align, KGP_MEMORY_DEFAULT);
      upload = kgpMalloc(dev, IMAGE_W * IMAGE_H * 4, 64, KGP_MEMORY_DEFAULT);
      CHECK(image_mem && upload);
      image = kgpCreateTexture(dev, &desc,
                               kgpHostToDevicePointer(dev, image_mem));
      CHECK(image);
      for (uint32_t i = 0; i < IMAGE_W * IMAGE_H; i++)
         upload[i] = 0xff000000u | (i * 2654435761u >> 8);
      CHECK(kgpTextureViewDescriptor(image, NULL, &heap[0]));

      KgpCommandBuffer cb = kgpStartCommandRecording(queue);
      CHECK(cb);
      kgpCopyToTexture(cb, image, 0, 0, kgpHostToDevicePointer(dev, upload),
                       IMAGE_W * 4);
      kgpBarrier(cb, KGP_STAGE_TRANSFER,
                 KGP_STAGE_COMPUTE | KGP_STAGE_VERTEX_SHADER |
                 KGP_STAGE_PIXEL_SHADER, KGP_HAZARD_DESCRIPTORS);
      CHECK(kgpSubmit(queue, &cb, 1, frame_semaphore, ++submitted));
   }

   for (unsigned f = 0; f < frames; f++) {
      KgpTexture backbuffer = kgpAcquireSwapchainTexture(swapchain);
      CHECK(backbuffer);
      /* The heap entry is rewritten per frame: the swapchain decides which
       * of its textures this is. Nothing submitted is still reading it
       * (the wait below).
       */
      if (heap != NULL)
         CHECK(kgpTextureViewDescriptor(backbuffer, NULL, &heap[1]));

      KgpCommandBuffer cb = kgpStartCommandRecording(queue);
      CHECK(cb);
      if (cs != NULL) {
         kgpSetPipeline(cb, cs);
         kgpSetActiveTextureHeapPtr(cb, heap_gpu, 2);
         kgpDispatch(cb, NULL,
                     (KgpUvec3) { (width + 15) / 16, (height + 15) / 16, 1 });
      }
      if (playground) {
         kgpBarrier(cb, KGP_STAGE_COMPUTE,
                    KGP_STAGE_VERTEX_SHADER | KGP_STAGE_PIXEL_SHADER |
                    KGP_STAGE_RASTER_COLOR_OUT, KGP_HAZARD_DESCRIPTORS);
      }
      if (gfx != NULL) {
         kgpBeginRenderPass(cb, &(KgpRenderPassDesc) {
            .colorTargetCount = 1,
            .colorTargets = { {
               .texture = backbuffer,
               .load = playground ? KGP_LOAD_KEEP : KGP_LOAD_CLEAR,
               .clearColor = { 1, 0, 0, 1 },
            } },
         });
         kgpSetPipeline(cb, gfx);
         kgpDrawIndexedInstanced(cb, kgpHostToDevicePointer(dev, vertices),
                                 NULL, kgpHostToDevicePointer(dev, indices),
                                 3, 1);
         kgpEndRenderPass(cb);
      }
      if (f == 0) {
         kgpBarrier(cb, gfx ? KGP_STAGE_RASTER_COLOR_OUT : KGP_STAGE_COMPUTE,
                    KGP_STAGE_TRANSFER, KGP_HAZARD_NONE);
         kgpCopyFromTexture(cb, backbuffer, 0, 0,
                            kgpHostToDevicePointer(dev, out),
                            (size_t)width * 4);
      }
      CHECK(kgpSubmit(queue, &cb, 1, frame_semaphore, ++submitted));
      /* no_api waits for the frame before last; kgpPresent needs this one
       * finished.
       */
      CHECK(kgpWaitSemaphore(dev, frame_semaphore, submitted,
                             KGP_TEST_TIMEOUT_NS));
      CHECK(kgpPresent(swapchain));
   }

   int ret = 0;
   if (kgp_test_offline()) {
      printf("noapi %s: %u frames recorded and presented (offline, results "
             "not checked)\n", example, frames);
   } else {
      /* Inside the triangle, and near a corner outside it; the second on a
       * pixel the gradient writes (not the first row or column of a
       * workgroup).
       */
      const uint32_t cx = width / 2, cy = height / 2 + height / 8;
      const uint32_t ox = 21, oy = 19;
      if (triangle) {
         ret |= expect(out, width, cx, cy, 0, 0, 255, "triangle");
         ret |= expect(out, width, ox, oy, 255, 0, 0, "clear colour");
      } else if (compute) {
         ret |= expect_gradient(out, width, height, cx | 1, cy | 1);
         ret |= expect_gradient(out, width, height, ox, oy);
         ret |= expect_gradient(out, width, height, width - 2, height - 2);
      } else {
         /* Every vertex has red 1; green and blue vary across it. */
         int got[4];
         unpack(out[(size_t)cy * width + cx], got);
         if (got[0] != 255 || got[3] != 255) {
            fprintf(stderr, "noapi: playground: pixel (%u, %u) is (%d, %d, "
                    "%d, %d), expected the triangle (red 255)\n", cx, cy,
                    got[0], got[1], got[2], got[3]);
            ret = 1;
         }
         ret |= expect_gradient(out, width, height, ox, oy);
         ret |= expect_gradient(out, width, height, width - 2, height - 2);
      }
      if (ret == 0)
         printf("noapi %s: %ux%u backbuffer correct, %u frames presented%s\n",
                example, width, height, frames,
                display ? " on the display" : " (headless)");
   }

   kgpDestroySemaphore(dev, frame_semaphore);
   if (image != NULL)
      kgpDestroyTexture(dev, image);
   if (gfx != NULL)
      kgpDestroyPipeline(dev, gfx);
   if (cs != NULL)
      kgpDestroyPipeline(dev, cs);
   kgpDestroySwapchain(swapchain);
   kgpFree(dev, upload);
   kgpFree(dev, image_mem);
   kgpFree(dev, heap);
   kgpFree(dev, indices);
   kgpFree(dev, vertices);
   kgpFree(dev, out);
   kgpDestroyDevice(dev);
   return ret;
}
