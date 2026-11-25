/*
 * KoreGP internals. NVIDIA (NVK) backend only; see docs/nvk-mapping.md.
 * SPDX-License-Identifier: MIT
 */
#ifndef KGP_PRIVATE_H
#define KGP_PRIVATE_H

#include "koregp/kgp.h"

#include "nak.h"
#include "nil.h"
#include "nouveau/vulkan/nvkmd/nvkmd.h"
#include "nv_push.h"
#include "util/list.h"
#include "util/log.h"
#include "util/rb_tree.h"
#include "util/simple_mtx.h"
#include "util/u_dynarray.h"

/* nvkmd logs through vk_errorf(log_obj, ...). NULL is safe there
 * (vk_log.c:__vk_errorv falls back to mesa_log), so KoreGP creates no
 * Vulkan object at all. Delta D8.
 */
#define KGP_LOG_OBJ NULL

#define kgp_err(...) mesa_loge("koregp: " __VA_ARGS__)

/* Ancestor: NVK_CMD_MEM_SIZE (nvk_cmd_pool.h) */
#define KGP_CHUNK_SIZE_B (64 * 1024)

enum kgp_debug {
   KGP_DEBUG_PUSH = 1 << 0,   /* dump every submitted pushbuf to stderr */
   KGP_DEBUG_ASM  = 1 << 1,   /* dump NAK assembly at pipeline creation */
   KGP_DEBUG_NIR  = 1 << 2,   /* dump NIR before NAK */
   KGP_DEBUG_VALIDATE = 1 << 5,   /* KGP_DEBUG=1: kgp_validate.c */
};

/* The root convention (delta D5): bound as cbuf 0 on every dispatch.
 * kgp_nir_lower_root turns push-constant loads into loads from here.
 */
struct kgp_root_record {
   uint64_t root;              /* kgpDispatch dataGpu; push-constant offset 0 */
   uint32_t group_count[3];    /* gl_NumWorkGroups */
   /* Entries in the texture and the sampler heap as this command buffer
    * set them: read only by shaders compiled under validation
    * (kgp_nir_lower_heap), which check their heap indices against them.
    */
   uint32_t heap_counts[2];
   uint32_t pad[9];
};
#define KGP_ROOT_PUSH_SIZE_B 8 /* how much of it SPIR-V push constants may read */
#define KGP_CBUF_ROOT 0
#define KGP_CBUF_SHADER_DATA 1
/* Raster (delta D10): Turing's eight hardware root tables are constant
 * buffers 24 to 31 in every stage. Table 0 holds the draw parameters NVK's
 * macros maintain; the two root pointers go where NVK keeps Vulkan push
 * constants, the start of table 7. Ancestor: NVK_HW_ROOT_TABLE_FIRST_CB and
 * struct nvk_root_descriptor_table.
 */
#define KGP_GFX_CBUF_DRAW_PARAMS 24
#define KGP_GFX_CBUF_ROOT 31
#define KGP_GFX_ROOT_TABLE 7
/* Under validation the two heap counts follow the two root pointers. */
#define KGP_GFX_ROOT_HEAP_COUNTS_OFFSET_B 16
/* Ancestor: nvk_min_cbuf_alignment() on Turing+, NVK_MAX_CBUF_SIZE */
#define KGP_CBUF_ALIGN_B 64
/* What validation asks of a root pointer. The proposal names no number;
 * NAK loads through a root in 32-bit words.
 */
#define KGP_ROOT_ALIGN_B 4
#define KGP_MAX_CBUF_SIZE_B (64 * 1024)
static_assert(sizeof(struct kgp_root_record) == KGP_CBUF_ALIGN_B,
              "root record is exactly one cbuf alignment unit");

struct kgp_alloc {
   struct rb_node node;        /* KgpDevice_T::allocs[], keyed by [key, key+size) */
   struct list_head link;      /* KgpDevice_T::pending_free */
   struct nvkmd_mem *mem;
   uintptr_t key;              /* CPU pointer; the GPU VA for KGP_MEMORY_GPU */
   uint64_t size_B;
   uint64_t free_seq;
   uint64_t req_B;             /* what kgpMalloc was asked for; validation */
   uint64_t id;                /* never reused; validation */
};

enum kgp_chunk_kind {
   KGP_CHUNK_PUSH,             /* pushbuf: system memory */
   KGP_CHUNK_UPLOAD,           /* root records and QMDs: as local as possible */
   KGP_CHUNK_KINDS,
};

struct kgp_chunk {
   struct list_head link;
   struct nvkmd_mem *mem;
};

/* Where one constant buffer's address sits in the QMD. */
struct kgp_qmd_addr_patch {
   uint8_t lo_dw;              /* dword holding the low 32 bits */
   uint8_t hi_dw;              /* dword holding the remaining high bits */
   uint8_t hi_shift;           /* ... at this bit position */
   uint8_t addr_shift;         /* the address is stored shifted right */
};

struct KgpPipeline_T {
   struct nak_shader_info info;
   struct kgp_alloc *alloc;    /* code, then constant data */
   uint64_t code_addr;
   uint64_t data_addr;
   uint32_t data_size_B;       /* 0 if the shader has no constant data */

   /* The QMD (the compute launch descriptor) for this shader with every
    * field filled in but the two things a dispatch varies: the group
    * counts and the address of its root record. kgp_pipeline_fill_qmd
    * copies it and patches those. Possible because the cbuf bindings are
    * fixed by convention rather than by descriptor state.
    */
   uint32_t qmd[NAK_MAX_QMD_DWORDS];
   uint32_t qmd_size_B;
   bool qmd_patchable;         /* false: field layout not word-aligned */
   uint8_t qmd_size_dw;        /* dword index of the X,Y,Z group counts */
   struct kgp_qmd_addr_patch qmd_root_record;   /* cbuf 0 address */
   uint32_t max_groups_y;      /* the QMD's Y and Z count fields are narrow */
   uint32_t max_groups_z;

   /* Graphics pipelines (kgpCreateGraphicsPipeline). `info` is then the
    * vertex shader's and nothing of the QMD above is used. gfx_push is the
    * method block that binds both shaders and their constant data, built
    * once (ancestor: nvk_shader::push_dw).
    */
   bool graphics;
   bool mesh;                  /* graphics, with mesh (and task) shaders */
   struct nak_shader_info ps_info;
   KgpRasterDesc raster;
   uint32_t *gfx_push;
   uint32_t gfx_push_dw;
};

/* One shader of a graphics pipeline, as uploaded. */
struct kgp_gfx_shader {
   const struct nak_shader_info *info;
   uint64_t hdr_addr;          /* the program header; code follows it */
   uint32_t size_B;            /* header and code */
   uint64_t gs_hdr_addr;       /* mesh shader with per-primitive outputs */
   uint64_t data_addr;         /* constant data, bound as cbuf 1 of the stage */
   uint32_t data_size_B;
};

/* A layout over caller memory; kgp_texture.c. */
struct KgpTexture_T {
   struct KgpDevice_T *dev;
   struct nil_image image;
   uint64_t addr;
   uint64_t mem_addr;          /* the caller's pointer; validation */
   KGP_FORMAT format;
   struct nvkmd_va *alias_va;  /* mapped under its PTE kind: kgp_texture.c */
   struct nvkmd_mem *owned_mem;   /* swapchain textures: imported memory */
};

/* The depth-stencil methods, ready to copy into a command buffer. */
#define KGP_DS_PUSH_DW 40
struct KgpDepthStencilState_T {
   bool needs_depth_target;
   bool needs_stencil;
   /* Depth bias is raster state to the hardware and to NVK, emitted in
    * another place than the snippet below (kgp_push_depth_bias).
    */
   bool bias_enable;
   float bias, bias_slope, bias_clamp;
   uint32_t push_dw;
   uint32_t push[KGP_DS_PUSH_DW];
};

struct KgpBlendState_T {
   KgpBlendDesc desc;
   bool enable;                /* the equation is not src * 1 + dst * 0 */
   uint32_t write_mask;        /* four bits, R in bit 0 */
};

/* A KGP_COLOR_WRITE value as the hardware's four bits, R in bit 0. */
static inline uint32_t
kgp_color_write_bits(uint32_t color_write)
{
   return color_write == KGP_COLOR_WRITE_ALL ? 0xf :
          color_write == KGP_COLOR_WRITE_NONE ? 0 : color_write;
}

struct KgpCommandBuffer_T {
   struct list_head link;      /* KgpQueue_T::recording, then ::pending */
   struct KgpQueue_T *queue;
   uint64_t seq;               /* device timeline point, once submitted */
   bool failed;                /* an allocation failed; submit will refuse */
   bool validate;              /* KGP_DEBUG=1 */
   struct util_dynarray refs;  /* struct kgp_ref (kgp_validate.c) */

   struct list_head chunks[KGP_CHUNK_KINDS];   /* the current one is last */
   struct nv_push push;
   struct kgp_chunk *push_chunk;
   struct util_dynarray execs;                 /* struct nvkmd_ctx_exec */
   struct kgp_chunk *upload_chunk;
   uint32_t upload_offset;

   const struct KgpPipeline_T *pipeline;

   /* Raster. The only shadow state: whether this buffer has had its 3D
    * prologue, the render pass's extent and target count, and whether the
    * fixed-function state and shaders must be emitted before the next draw
    * (after kgpSetPipeline or kgpBeginRenderPass).
    */
   bool gfx_begun;
   bool gfx_dirty;
   bool ds_dirty;              /* only the depth-stencil state is due */
   bool blend_dirty;           /* only the blend state is due */
   const struct KgpDepthStencilState_T *ds_state;   /* NULL: all off */
   const struct KgpBlendState_T *blend_state;       /* NULL: no blending */
   bool rp_active;
   uint32_t rp_width;
   uint32_t rp_height;
   uint32_t rp_color_count;
   KGP_FORMAT rp_formats[KGP_MAX_COLOR_TARGETS];
   uint32_t rp_samples;        /* per pixel, of every target */
   KGP_FORMAT rp_depth_format; /* KGP_FORMAT_NONE: no depth target */
   /* The index buffer this command buffer last bound: an address and how
    * far the hardware may read. Zero: none yet.
    */
   uint64_t ib_addr;
   uint32_t ib_bytes;
   /* Entries in the heaps this command buffer set: textures, samplers.
    * Zero until it sets one, whatever an earlier command buffer left.
    */
   uint32_t heap_counts[2];
};

struct KgpQueue_T {
   struct KgpDevice_T *dev;
   struct nvkmd_ctx *ctx;
   simple_mtx_t submit_mutex;  /* serializes kgpSubmit; taken before dev->mutex */
   uint64_t slm_bytes_per_tpc; /* shader-local memory size the channel has */
   /* What the 3D init points the hardware at; ancestors nvk_queue::draw_cb0
    * and nvk_device::zero_page.
    */
   struct nvkmd_mem *draw_cb0;
   struct nvkmd_mem *zero_page;
   struct list_head free_chunks[KGP_CHUNK_KINDS];
   struct list_head recording;
   struct list_head pending;   /* submitted, oldest first */
};

struct KgpDevice_T {
   struct nvkmd_pdev *pdev;
   struct nvkmd_dev *nvkmd;
   struct nak_compiler *nak;
   int drm_fd;
   uint32_t debug;
   /* The DRM shim never signals anything; it asks userspace to treat every
    * fence as complete (NOUVEAU_DISABLE_FENCES).
    */
   bool no_fences;
   /* A submit failed. The channel is dead, so nothing is in flight. */
   bool lost;

   /* Guards everything below and the queue's lists. Never taken while
    * recording unless a command buffer needs another chunk.
    */
   simple_mtx_t mutex;

   /* [0] keyed by CPU pointer, [1] by GPU VA (KGP_MEMORY_GPU). Two trees
    * because the two address spaces can overlap numerically.
    */
   struct rb_tree allocs[2];
   struct list_head pending_free;
   uint64_t next_alloc_id;

   /* One timeline for the whole device. Every submit signals the next
    * point; chunks and freed allocations retire against it.
    */
   uint32_t timeline;
   uint64_t submit_seq;
   uint64_t done_seq;

   /* Shader-local memory. Only grows. Ancestor: nvk_slm_area. */
   struct nvkmd_mem *slm_mem;
   uint64_t slm_bytes_per_tpc;
   uint64_t slm_bytes_per_warp;
   struct util_dynarray old_slm_mems;          /* struct nvkmd_mem * */

   struct KgpQueue_T queue;
};

static inline const struct nv_device_info *
kgp_dev_info(const struct KgpDevice_T *dev)
{
   return &dev->pdev->dev_info;
}

/* kgp_device.c */
void kgp_device_poll_locked(struct KgpDevice_T *dev);

/* kgp_memory.c */
struct kgp_alloc *kgp_alloc_create_mapped(struct KgpDevice_T *dev,
                                          uint64_t size_B,
                                          enum nvkmd_mem_flags flags,
                                          enum nvkmd_mem_map_flags map_flags);
void kgp_alloc_free_deferred(struct KgpDevice_T *dev, struct kgp_alloc *a);
struct KgpTexture_T *kgp_texture_over_mem(struct KgpDevice_T *dev,
                                          const struct nil_image *image,
                                          KGP_FORMAT format,
                                          struct nvkmd_mem *mem,
                                          uint64_t offset_B);
struct nvkmd_mem *kgp_gpu_only_mem(struct KgpDevice_T *dev, uint64_t addr,
                                   uint64_t *offset_B_out);
void kgp_memory_reap_locked(struct KgpDevice_T *dev);
void kgp_memory_finish(struct KgpDevice_T *dev);
const struct kgp_alloc *kgp_alloc_at_gpu_locked(struct KgpDevice_T *dev,
                                                uint64_t addr);

/* kgp_validate.c */
bool kgp_validate_range(struct KgpDevice_T *dev, struct util_dynarray *refs,
                        const char *what, uint64_t addr, uint64_t size_B,
                        uint32_t align_B);
bool kgp_cb_validate_ptr(struct KgpCommandBuffer_T *cb, const char *what,
                         uint64_t addr, uint64_t size_B, uint32_t align_B);
bool kgp_cb_validate_texture(struct KgpCommandBuffer_T *cb, const char *what,
                             const struct KgpTexture_T *tex);
bool kgp_cb_validate_barrier(struct KgpCommandBuffer_T *cb, uint32_t src,
                             uint32_t dst, uint32_t hazards);
bool kgp_cb_validate_submit(struct KgpCommandBuffer_T *cb, uint32_t index);

/* False: the pointer failed validation and the command buffer is poisoned.
 * `size_B` is what the call will touch from `ptr` on, 1 where only the GPU
 * will know.
 */
static inline bool
kgp_check_ptr(struct KgpCommandBuffer_T *cb, const char *what,
              const void *ptr, uint64_t size_B, uint32_t align_B)
{
   return likely(!cb->validate) ||
          kgp_cb_validate_ptr(cb, what, (uintptr_t)ptr, size_B, align_B);
}

static inline bool
kgp_check_texture(struct KgpCommandBuffer_T *cb, const char *what,
                  const struct KgpTexture_T *tex)
{
   return likely(!cb->validate) || kgp_cb_validate_texture(cb, what, tex);
}

/* kgp_pipeline.c */
bool kgp_device_ensure_slm(struct KgpDevice_T *dev,
                           uint32_t slm_bytes_per_lane,
                           uint32_t crs_bytes_per_warp);
void kgp_pipeline_fill_qmd(const struct nv_device_info *info,
                           const struct KgpPipeline_T *pipeline,
                           uint32_t *qmd, uint64_t root_record_addr,
                           uint32_t x, uint32_t y, uint32_t z);

/* kgp_nir_lower_heap.c. The heap convention's fixed SPIR-V bindings. */
#define KGP_HEAP_BINDING_TEXTURES 0
#define KGP_HEAP_BINDING_SAMPLERS 1
#define KGP_MAX_HEAP_TEXTURES (1u << 20)   /* the handle's index widths */
#define KGP_MAX_HEAP_SAMPLERS (1u << 12)
struct nir_shader;
bool kgp_nir_lower_heap(struct nir_shader *nir, bool bounds_check,
                        bool *bad_out);

/* kgp_nir_lower_root.c */
struct nir_shader;
bool kgp_nir_lower_root(struct nir_shader *nir, bool has_task,
                        bool *bad_push_out);

/* kgp_draw.c */
#define KGP_DRAW_CB0_MEM_SIZE_B 4096
/* Ancestor: the stack buffer of nvk_queue_init_context_state. */
#define KGP_INIT_PUSH_DW (4096 + 1024)
void kgp_push_own_macros(const struct nv_device_info *info,
                         struct nv_push *p, uint32_t mme_pos);
bool kgp_push_draw_state_init(const struct nv_device_info *info,
                              struct nv_push *p, uint64_t cb0_addr,
                              uint64_t zero_addr, uint32_t *mme_end_out);
bool kgp_gfx_pipeline_build_push(const struct nv_device_info *info,
                                 struct KgpPipeline_T *pipeline,
                                 const struct kgp_gfx_shader *shaders,
                                 unsigned count);

/* kgp_cmd.c */
bool kgp_queue_init(struct KgpDevice_T *dev);
void kgp_queue_finish(struct KgpDevice_T *dev);
void kgp_queue_reap_locked(struct KgpQueue_T *queue);
struct KgpCommandBuffer_T *kgp_queue_slm_update(struct KgpQueue_T *queue);
struct nv_push *kgp_cb_push(struct KgpCommandBuffer_T *cb, uint32_t dw_count);
void kgp_cb_flush_push(struct KgpCommandBuffer_T *cb);

#endif /* KGP_PRIVATE_H */
