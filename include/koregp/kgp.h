/*
 * KoreGP public API. C99.
 *
 * The compute subset of the proposal's header (N1): device, memory, compute
 * pipelines, command recording, barriers, submission, timeline semaphores;
 * N2: textures, samplers, their heaps, and GPU-side copies; and the start
 * of N3: graphics pipelines, colour render passes and indexed instanced
 * draws (docs/plan.md says what of raster is missing).
 *
 * Conventions
 *  - Handles are opaque pointers. NULL is never a valid handle.
 *  - Creation functions return NULL on failure; functions returning bool
 *    return false on failure. The reason is logged to stderr.
 *  - A "GPU pointer" is a GPU virtual address carried in a void*. It must
 *    not be dereferenced on the CPU. Convert a CPU pointer with
 *    kgpHostToDevicePointer once and keep both.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef KGP_H
#define KGP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__GNUC__)
#define KGP_API __attribute__((visibility("default")))
#else
#define KGP_API
#endif

typedef struct KgpDevice_T *KgpDevice;
typedef struct KgpQueue_T *KgpQueue;
typedef struct KgpPipeline_T *KgpPipeline;
typedef struct KgpCommandBuffer_T *KgpCommandBuffer;
typedef struct KgpSemaphore_T *KgpSemaphore;

typedef struct KgpByteSpan {
   const void *data;
   size_t size;
} KgpByteSpan;

typedef struct KgpUvec3 {
   uint32_t x, y, z;
} KgpUvec3;

typedef enum KGP_MEMORY {
   /* GPU memory, CPU-mapped write-combined. kgpMalloc returns the CPU
    * pointer. Placed in VRAM where the device can map it (see
    * KgpDeviceProperties::mappedVramBytes), otherwise in system memory.
    */
   KGP_MEMORY_DEFAULT = 0,
   /* VRAM with no CPU mapping. kgpMalloc returns the GPU pointer. */
   KGP_MEMORY_GPU,
   /* System memory, CPU-cached: fast to read back. Returns the CPU pointer. */
   KGP_MEMORY_READBACK,
} KGP_MEMORY;

typedef enum KGP_STAGE {
   KGP_STAGE_TRANSFER         = 1 << 0,
   KGP_STAGE_COMPUTE          = 1 << 1,
   KGP_STAGE_VERTEX_SHADER    = 1 << 2,
   KGP_STAGE_PIXEL_SHADER     = 1 << 3,
   KGP_STAGE_RASTER_COLOR_OUT = 1 << 4,
   KGP_STAGE_RASTER_DEPTH_OUT = 1 << 5,
} KGP_STAGE;

typedef enum KGP_HAZARD {
   KGP_HAZARD_NONE           = 0,
   /* The producer wrote indirect dispatch/draw arguments. */
   KGP_HAZARD_DRAW_ARGUMENTS = 1 << 0,
   /* The producer wrote texture or sampler descriptors. */
   KGP_HAZARD_DESCRIPTORS    = 1 << 1,
   /* The producer wrote depth/stencil from a non-raster stage. */
   KGP_HAZARD_DEPTH_STENCIL  = 1 << 2,
} KGP_HAZARD;

typedef struct KgpDeviceProperties {
   char     deviceName[64];
   uint32_t vendorId;
   uint32_t deviceId;
   uint32_t chipset;          /* vendor chip id, e.g. 0x164 for TU104 */
   uint64_t vramBytes;
   /* How much VRAM can be CPU-mapped at once without the kernel evicting
    * mappings. Equal to vramBytes with ReBAR/UMA; the BAR size otherwise.
    */
   uint64_t mappedVramBytes;
} KgpDeviceProperties;

/* Device ------------------------------------------------------------------ */

/* Opens the index'th compatible GPU. Each device has exactly one queue. */
KGP_API KgpDevice kgpCreateDevice(uint32_t index);
KGP_API void      kgpDestroyDevice(KgpDevice);
KGP_API void      kgpGetDeviceProperties(KgpDevice, KgpDeviceProperties *out);
KGP_API KgpQueue  kgpGetQueue(KgpDevice);

/* Memory ------------------------------------------------------------------ */

KGP_API void *kgpMalloc(KgpDevice, size_t bytes, size_t align, KGP_MEMORY type);
/* Destruction is deferred until everything submitted before the call has
 * finished on the GPU. Accepts what kgpMalloc returned.
 */
KGP_API void  kgpFree(KgpDevice, void *ptr);
/* Any pointer into a live kgpMalloc allocation -> its GPU pointer. NULL if
 * the pointer is not inside one. Not for the command path: call it once per
 * allocation and keep the result.
 */
KGP_API void *kgpHostToDevicePointer(KgpDevice, void *cpuPtr);

/* Textures ---------------------------------------------------------------- */

/* A texture is a layout over GPU memory the caller already owns. Creating
 * one allocates nothing on the GPU: ask kgpTextureSizeAlign what the layout
 * needs, kgpMalloc that, and hand the GPU pointer to kgpCreateTexture.
 *
 * Shaders reach textures by index into the texture heap, which is caller
 * memory too: an array of KgpDescriptor256 at a GPU pointer set with
 * kgpSetActiveTextureHeapPtr. Entry i is filled by copying in what
 * kgpTextureViewDescriptor wrote, from the CPU or from a shader. A zeroed
 * entry is a valid null texture.
 *
 * SPIR-V convention: the texture heap is a runtime array at set 0,
 * binding 0 (texture* types for sampling and fetching, image* types for
 * storage access; the same entry serves both); the sampler heap is a
 * runtime array at set 0, binding 1, filled the same way from
 * kgpSamplerDescriptor and set with kgpSetActiveSamplerHeapPtr. Fetches
 * (texelFetch) need no sampler.
 */
typedef struct KgpTexture_T *KgpTexture;

typedef struct KgpDescriptor256 {
   uint32_t bits[8];
} KgpDescriptor256;

typedef enum KGP_FORMAT {
   KGP_FORMAT_NONE = 0,
   KGP_FORMAT_RGBA8_UNORM,
   KGP_FORMAT_RGBA8_SRGB,
   KGP_FORMAT_R32_UINT,
   KGP_FORMAT_R32_FLOAT,
   KGP_FORMAT_RGBA16_FLOAT,
   KGP_FORMAT_RGBA32_FLOAT,
   KGP_FORMAT_D32_FLOAT,       /* depth; a render pass's depth target */
   /* Depth and stencil. Must lie in KGP_MEMORY_GPU memory, starting on a
    * 4096-byte boundary of its allocation.
    */
   KGP_FORMAT_D24_UNORM_S8_UINT,
   KGP_FORMAT_BGRA8_UNORM,     /* what a swapchain's textures are */
} KGP_FORMAT;

typedef enum KGP_TEXTURE_FLAGS {
   /* Row-major layout instead of the hardware's tiled one. 2D, one mip
    * level, no layers. Slower to sample; its memory needs nothing special.
    */
   KGP_TEXTURE_LINEAR = 1 << 0,
} KGP_TEXTURE_FLAGS;

typedef struct KgpTextureDesc {
   KGP_FORMAT format;
   uint32_t width;
   uint32_t height;
   uint32_t depth;        /* 0 or 1: not a 3D texture */
   uint32_t layerCount;   /* 0 or 1: not an array */
   uint32_t mipCount;     /* 0 means 1 */
   uint32_t flags;        /* KGP_TEXTURE_FLAGS */
} KgpTextureDesc;

typedef struct KgpSizeAlign {
   uint64_t size;
   uint64_t align;
} KgpSizeAlign;

/* Zero counts mean "to the end"; KGP_FORMAT_NONE means the texture's. */
typedef struct KgpViewDesc {
   KGP_FORMAT format;
   uint32_t baseMip;
   uint32_t mipCount;
   uint32_t baseLayer;
   uint32_t layerCount;
} KgpViewDesc;

KGP_API bool kgpTextureSizeAlign(KgpDevice, const KgpTextureDesc *,
                                 KgpSizeAlign *out);
KGP_API KgpTexture kgpCreateTexture(KgpDevice, const KgpTextureDesc *,
                                    void *gpuPtr);
/* Frees the handle only; the memory is the caller's. */
KGP_API void kgpDestroyTexture(KgpDevice, KgpTexture);
/* viewDesc may be NULL for the whole texture. */
KGP_API bool kgpTextureViewDescriptor(KgpTexture, const KgpViewDesc *viewDesc,
                                      KgpDescriptor256 *out);
/* CPU upload of one mip level of one layer from row-major pixels, for
 * texture memory the caller has CPU-mapped: textureCpu is the CPU pointer to
 * the memory whose GPU pointer was given to kgpCreateTexture.
 */
KGP_API bool kgpWriteTexture(KgpTexture, void *textureCpu, uint32_t mip,
                             uint32_t layer, const void *pixels,
                             size_t rowStride);

typedef enum KGP_FILTER {
   KGP_FILTER_NEAREST = 0,
   KGP_FILTER_LINEAR,
} KGP_FILTER;

typedef enum KGP_ADDRESS {
   KGP_ADDRESS_REPEAT = 0,
   KGP_ADDRESS_MIRRORED_REPEAT,
   KGP_ADDRESS_CLAMP_TO_EDGE,
   KGP_ADDRESS_CLAMP_TO_BORDER,
} KGP_ADDRESS;

/* Zero-initialized: nearest filtering, repeat, and only mip level 0
 * (maxLod is 0). Set maxLod to use the mip chain.
 */
typedef struct KgpSamplerDesc {
   KGP_FILTER minFilter;
   KGP_FILTER magFilter;
   KGP_FILTER mipFilter;
   KGP_ADDRESS addressU;
   KGP_ADDRESS addressV;
   KGP_ADDRESS addressW;
   float mipLodBias;
   float minLod;
   float maxLod;
   float maxAnisotropy;    /* 0 or 1: off; up to 16 */
   float borderColor[4];   /* for KGP_ADDRESS_CLAMP_TO_BORDER */
} KgpSamplerDesc;

/* Writes the 256-bit sampler entry for the sampler heap. */
KGP_API bool kgpSamplerDescriptor(KgpDevice, const KgpSamplerDesc *,
                                  KgpDescriptor256 *out);

/* Pipelines --------------------------------------------------------------- */

/* Values for a shader's specialization constants, by SPIR-V SpecId
 * (GLSL: layout(constant_id = N)). `value` holds the constant's bits in its
 * low end: a bool as 0 or 1, a float as its IEEE bit pattern, a 64-bit
 * constant whole. Constants not listed keep the module's default. An id
 * that no shader of the pipeline declares is an error.
 */
typedef struct KgpSpecConstant {
   uint32_t id;
   uint64_t value;
} KgpSpecConstant;

typedef struct KgpSpecialization {
   const KgpSpecConstant *constants;
   uint32_t count;
} KgpSpecialization;

/* SPIR-V 1.3+ compute shader, entry point "main". Root convention: a
 * push-constant block whose first member is a PhysicalStorageBuffer
 * pointer; kgpDispatch's dataGpu arrives there. `constants` may be NULL.
 */
KGP_API KgpPipeline kgpCreateComputePipeline(KgpDevice, KgpByteSpan spirv,
                                             const KgpSpecialization *constants);

typedef enum KGP_TOPOLOGY {
   KGP_TOPOLOGY_TRIANGLE_LIST = 0,
   KGP_TOPOLOGY_TRIANGLE_STRIP,
} KGP_TOPOLOGY;

/* Which winding is discarded. Counter-clockwise is the front face. */
typedef enum KGP_CULL {
   KGP_CULL_NONE = 0,
   KGP_CULL_CW,
   KGP_CULL_CCW,
} KGP_CULL;

#define KGP_MAX_COLOR_TARGETS 8

typedef struct KgpRasterDesc {
   KGP_TOPOLOGY topology;
   KGP_CULL cull;
   uint32_t colorTargetCount;                         /* 1 to 8 */
   KGP_FORMAT colorFormats[KGP_MAX_COLOR_TARGETS];
   KGP_FORMAT depthFormat;     /* KGP_FORMAT_NONE: drawn without a depth target */
} KgpRasterDesc;

/* A vertex and a pixel shader, SPIR-V 1.3+, entry points "main". There are
 * no vertex buffers: the vertex shader fetches what it needs through its
 * root pointer, by gl_VertexIndex and gl_InstanceIndex. Root convention for
 * raster: each stage declares a push-constant block holding one
 * PhysicalStorageBuffer pointer, at offset 0 in the vertex shader and at
 * offset 8 in the pixel shader; a draw supplies the two pointers.
 * `constants` may be NULL; one list serves both stages.
 */
KGP_API KgpPipeline kgpCreateGraphicsPipeline(KgpDevice, KgpByteSpan vs,
                                              KgpByteSpan ps,
                                              const KgpRasterDesc *,
                                              const KgpSpecialization *constants);
KGP_API void        kgpDestroyPipeline(KgpDevice, KgpPipeline);

/* Commands ---------------------------------------------------------------- */

/* Command buffers are transient: record, submit once, forget. A command
 * buffer that is never submitted is reclaimed by kgpDestroyDevice.
 */
KGP_API KgpCommandBuffer kgpStartCommandRecording(KgpQueue);
KGP_API void kgpSetPipeline(KgpCommandBuffer, KgpPipeline);
KGP_API void kgpDispatch(KgpCommandBuffer, void *dataGpu, KgpUvec3 groups);
/* The group counts are read by the GPU, at execution time, from a KgpUvec3
 * at groupsGpu (a GPU pointer, 4-byte aligned). If an earlier dispatch in
 * the same submission wrote them, put a kgpBarrier with
 * KGP_HAZARD_DRAW_ARGUMENTS between the two. The Y and Z counts must fit in
 * 16 bits; that cannot be checked here.
 */
KGP_API void kgpDispatchIndirect(KgpCommandBuffer, void *dataGpu,
                                 void *groupsGpu);
KGP_API void kgpBarrier(KgpCommandBuffer, KGP_STAGE src, KGP_STAGE dst,
                        KGP_HAZARD hazards);
/* Raster. A render pass names the textures drawn into: on every current GPU
 * that is state the CPU must program, so it takes KgpTexture handles. The
 * area is the whole of the targets, which must all have the same extent;
 * viewport and scissor cover it. Targets use the tiled layout. The depth
 * target is optional; a pipeline draws only into passes whose depth format
 * is its KgpRasterDesc::depthFormat. Clip-space z runs from 0 (near) to 1.
 */
typedef enum KGP_LOAD {
   KGP_LOAD_KEEP = 0,     /* draw over what the target holds */
   KGP_LOAD_CLEAR,
} KGP_LOAD;

typedef struct KgpColorTarget {
   KgpTexture texture;
   uint32_t mip;
   uint32_t layer;
   KGP_LOAD load;
   float clearColor[4];
} KgpColorTarget;

typedef struct KgpDepthTarget {
   KgpTexture texture;    /* NULL: no depth target */
   uint32_t mip;
   uint32_t layer;
   KGP_LOAD load;         /* of depth and, if the format has it, stencil */
   float clearDepth;
   uint8_t clearStencil;
} KgpDepthTarget;

typedef struct KgpRenderPassDesc {
   uint32_t colorTargetCount;                         /* 1 to 8 */
   KgpColorTarget colorTargets[KGP_MAX_COLOR_TARGETS];
   KgpDepthTarget depthTarget;
} KgpRenderPassDesc;

KGP_API void kgpBeginRenderPass(KgpCommandBuffer, const KgpRenderPassDesc *);
KGP_API void kgpEndRenderPass(KgpCommandBuffer);
/* Draws with the graphics pipeline last set, inside a render pass whose
 * target count is the pipeline's. vsData and psData are the two root
 * pointers. indicesGpu points at indexCount uint32 indices. Before something
 * reads what a render pass drew, put a kgpBarrier with src
 * KGP_STAGE_RASTER_COLOR_OUT after kgpEndRenderPass.
 */
KGP_API void kgpDrawIndexedInstanced(KgpCommandBuffer, void *vsData,
                                     void *psData, void *indicesGpu,
                                     uint32_t indexCount,
                                     uint32_t instanceCount);

/* What the GPU reads at argsGpu for an indirect draw: the layout Vulkan and
 * Direct3D use. vertexOffset is added to each index and reaches the shader
 * in gl_VertexIndex; firstInstance in gl_InstanceIndex.
 */
typedef struct KgpDrawIndexedArgs {
   uint32_t indexCount;
   uint32_t instanceCount;
   uint32_t firstIndex;
   int32_t vertexOffset;
   uint32_t firstInstance;
} KgpDrawIndexedArgs;

/* The same draw with its arguments read by the GPU at execution time. If
 * something earlier in the submission wrote them, put a kgpBarrier with
 * KGP_HAZARD_DRAW_ARGUMENTS between the two.
 */
KGP_API void kgpDrawIndexedInstancedIndirect(KgpCommandBuffer, void *vsData,
                                             void *psData, void *indicesGpu,
                                             void *argsGpu);

/* GPU-driven multi-draw with per-draw root data. At execution time the GPU
 * reads a uint32 draw count at countGpu (capped at maxDrawCount) and that
 * many KgpDrawIndexedArgs, packed, at argsGpu. Draw i runs with the root
 * pointers vsData + i * vsStride and psData + i * psStride, and with
 * gl_DrawID = i. All draws share the index buffer. The count, the
 * arguments and the root data may all be written by the GPU earlier in the
 * submission; put a kgpBarrier with KGP_HAZARD_DRAW_ARGUMENTS in between.
 */
KGP_API void kgpDrawIndexedInstancedIndirectMulti(KgpCommandBuffer,
                                                  void *vsData,
                                                  uint32_t vsStride,
                                                  void *psData,
                                                  uint32_t psStride,
                                                  void *indicesGpu,
                                                  void *argsGpu,
                                                  void *countGpu,
                                                  uint32_t maxDrawCount);

/* Depth-stencil and blend state: objects made once, set between draws. A
 * command buffer starts with neither set, which is: no depth test, no depth
 * write, no blending. Setting NULL returns to that. Both stay set across
 * pipelines and render passes of the command buffer. A state object must
 * outlive the recording of the command buffers it is set in, not their
 * execution.
 */
typedef enum KGP_COMPARE {
   KGP_COMPARE_NEVER = 0,
   KGP_COMPARE_LESS,
   KGP_COMPARE_EQUAL,
   KGP_COMPARE_LESS_EQUAL,
   KGP_COMPARE_GREATER,
   KGP_COMPARE_NOT_EQUAL,
   KGP_COMPARE_GREATER_EQUAL,
   KGP_COMPARE_ALWAYS,
} KGP_COMPARE;

typedef enum KGP_STENCIL_OP {
   KGP_STENCIL_KEEP = 0,
   KGP_STENCIL_ZERO,
   KGP_STENCIL_REPLACE,        /* with stencilReference */
   KGP_STENCIL_INCR_CLAMP,
   KGP_STENCIL_DECR_CLAMP,
   KGP_STENCIL_INVERT,
   KGP_STENCIL_INCR_WRAP,
   KGP_STENCIL_DECR_WRAP,
} KGP_STENCIL_OP;

typedef struct KgpStencilFace {
   KGP_STENCIL_OP fail;        /* stencil test failed */
   KGP_STENCIL_OP depthFail;   /* stencil passed, depth failed */
   KGP_STENCIL_OP pass;
   KGP_COMPARE compare;        /* passes when: reference <compare> stored */
} KgpStencilFace;

/* Testing or writing depth needs a render pass with a depth target; the
 * stencil test one whose depth target has stencil. A zeroed desc is
 * everything off.
 */
typedef struct KgpDepthStencilDesc {
   bool depthTest;
   bool depthWrite;
   KGP_COMPARE depthCompare;   /* passes when: incoming <compare> stored */
   bool stencilTest;
   KgpStencilFace stencilFront, stencilBack;
   uint8_t stencilReadMask;    /* applied to both sides of the compare */
   uint8_t stencilWriteMask;
   uint8_t stencilReference;
} KgpDepthStencilDesc;

typedef struct KgpDepthStencilState_T *KgpDepthStencilState;
KGP_API KgpDepthStencilState kgpCreateDepthStencilState(
   KgpDevice, const KgpDepthStencilDesc *);
KGP_API void kgpDestroyDepthStencilState(KgpDevice, KgpDepthStencilState);
KGP_API void kgpSetDepthStencilState(KgpCommandBuffer, KgpDepthStencilState);

typedef enum KGP_BLEND_FACTOR {
   KGP_BLEND_ZERO = 0,
   KGP_BLEND_ONE,
   KGP_BLEND_SRC_COLOR,
   KGP_BLEND_INV_SRC_COLOR,
   KGP_BLEND_DST_COLOR,
   KGP_BLEND_INV_DST_COLOR,
   KGP_BLEND_SRC_ALPHA,
   KGP_BLEND_INV_SRC_ALPHA,
   KGP_BLEND_DST_ALPHA,
   KGP_BLEND_INV_DST_ALPHA,
} KGP_BLEND_FACTOR;

typedef enum KGP_BLEND_OP {
   KGP_BLEND_OP_ADD = 0,       /* src * srcFactor + dst * dstFactor */
   KGP_BLEND_OP_SUBTRACT,      /* src * srcFactor - dst * dstFactor */
   KGP_BLEND_OP_REV_SUBTRACT,  /* dst * dstFactor - src * srcFactor */
   KGP_BLEND_OP_MIN,
   KGP_BLEND_OP_MAX,
} KGP_BLEND_OP;

/* One equation, applied to every colour target of the render pass. */
typedef struct KgpBlendDesc {
   KGP_BLEND_FACTOR srcColor, dstColor;
   KGP_BLEND_OP colorOp;
   KGP_BLEND_FACTOR srcAlpha, dstAlpha;
   KGP_BLEND_OP alphaOp;
} KgpBlendDesc;

typedef struct KgpBlendState_T *KgpBlendState;
KGP_API KgpBlendState kgpCreateBlendState(KgpDevice, const KgpBlendDesc *);
KGP_API void kgpDestroyBlendState(KgpDevice, KgpBlendState);
KGP_API void kgpSetBlendState(KgpCommandBuffer, KgpBlendState);

/* GPU-side copies. They run on the copy engine, in order with the
 * dispatches around them, and are the KGP_STAGE_TRANSFER of kgpBarrier: put
 * a barrier with src TRANSFER before anything that reads what a copy wrote
 * (a later copy included), and one with src COMPUTE before a copy that reads
 * what a shader wrote. Source and destination must not overlap.
 */
KGP_API void kgpMemCpy(KgpCommandBuffer, void *dstGpu, void *srcGpu,
                       uint64_t bytes);
/* One mip level of one layer (all depth slices of a 3D texture) from or to
 * row-major texels at a GPU pointer, rows `rowStride` bytes apart: the GPU
 * counterpart of kgpWriteTexture, for texture memory that is not CPU-mapped.
 * rowStride is a multiple of the texel size.
 */
KGP_API void kgpCopyToTexture(KgpCommandBuffer, KgpTexture, uint32_t mip,
                              uint32_t layer, void *srcGpu, size_t rowStride);
KGP_API void kgpCopyFromTexture(KgpCommandBuffer, KgpTexture, uint32_t mip,
                                uint32_t layer, void *dstGpu,
                                size_t rowStride);
/* Points shaders at the texture heap: `count` KgpDescriptor256 entries at
 * heapGpu. It stays in effect for later command buffers on the queue, but
 * relying on that is fragile: set it in each command buffer that uses
 * textures. At most 2^20 entries. If a shader wrote entries since they were
 * last used, put a kgpBarrier with KGP_HAZARD_DESCRIPTORS before the use.
 */
KGP_API void kgpSetActiveTextureHeapPtr(KgpCommandBuffer, void *heapGpu,
                                        uint32_t count);
/* The same for the sampler heap. At most 4096 entries. */
KGP_API void kgpSetActiveSamplerHeapPtr(KgpCommandBuffer, void *heapGpu,
                                        uint32_t count);

/* Submits n command buffers in order. If `signal` is non-NULL it reaches
 * `value` when they have all finished.
 */
KGP_API bool kgpSubmit(KgpQueue, const KgpCommandBuffer *cbs, uint32_t n,
                       KgpSemaphore signal, uint64_t value);

/* Timeline semaphores ------------------------------------------------------ */

KGP_API KgpSemaphore kgpCreateSemaphore(KgpDevice, uint64_t initialValue);
KGP_API void         kgpDestroySemaphore(KgpDevice, KgpSemaphore);
/* Blocks until the semaphore reaches `value` or `timeoutNs` elapses. */
KGP_API bool         kgpWaitSemaphore(KgpDevice, KgpSemaphore, uint64_t value,
                                      uint64_t timeoutNs);

/* Presentation ------------------------------------------------------------ */

typedef struct KgpSwapchain_T *KgpSwapchain;

typedef enum KGP_PRESENT {
   /* The first connected display, at its preferred mode, with no compositor
    * in between: the process must be able to become DRM master.
    */
   KGP_PRESENT_DISPLAY = 0,
   /* No display at all: frames are presented to nowhere. For tests. */
   KGP_PRESENT_HEADLESS,
} KGP_PRESENT;

typedef struct KgpSwapchainDesc {
   KGP_PRESENT target;
   uint32_t width;    /* headless only; a display dictates its mode's size */
   uint32_t height;
} KgpSwapchainDesc;

typedef struct KgpSwapchainInfo {
   KGP_FORMAT format;
   uint32_t width;
   uint32_t height;
} KgpSwapchainInfo;

/* The swapchain's images are ordinary textures as far as rendering and
 * copying go; they belong to the swapchain and their memory is not the
 * caller's. Destroy the swapchain before the device.
 */
KGP_API KgpSwapchain kgpCreateSwapchain(KgpDevice, const KgpSwapchainDesc *,
                                        KgpSwapchainInfo *infoOut);
KGP_API void         kgpDestroySwapchain(KgpSwapchain);
/* The texture to render the next frame into; one at a time. Blocks until
 * one is free.
 */
KGP_API KgpTexture   kgpAcquireSwapchainTexture(KgpSwapchain);
/* Shows the acquired texture. It does not wait for rendering: everything
 * submitted that writes the texture must have completed (kgpWaitSemaphore).
 */
KGP_API bool         kgpPresent(KgpSwapchain);

#ifdef __cplusplus
}
#endif

#endif /* KGP_H */
