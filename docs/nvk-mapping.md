# KoreGP on NVK: mapping the proposal onto NVIDIA

The proposal (`../KoreGP-website/README.md`, Draft 0.2) is written in AMD
terms — PM4, SGPRs, ACO, addrlib, amdgpu — and schedules NVK as milestone
M4. This test implementation starts with NVK because the hardware we have is
a TU104. This document restates each section of the proposal in NVIDIA terms
and lists where the proposal has to bend.

Every source reference is to Mesa **26.3.1** (`~/koregp/mesa` on the host),
paths relative to `src/nouveau/`. Items marked *(verify)* are read from the
source but not yet exercised.

## Vocabulary

| Proposal (AMD) | NVIDIA / NVK | Where |
|---|---|---|
| RADV | NVK | `vulkan/` |
| ACO | NAK (Rust, NIR in) | `compiler/`, C API `compiler/nak.h` |
| addrlib / `ac_surface` | NIL | `nil/` |
| PM4 packets | pushbuf methods (`P_MTHD`, `P_IMMD`) on class objects | `headers/nv_push.h`, generated `nv_push_cl*.h` |
| GFX10.3 register classes | TURING_A `C597` (3D), TURING_COMPUTE_A `C5C0`, TURING_DMA_COPY_A `C5B5` | `headers/nvidia/classes/` |
| libdrm_amdgpu winsys | `nouveau_ws_*` + the `nvkmd` abstraction | `winsys/`, `vulkan/nvkmd/` |
| SGPR preload | constant buffers (cbufs) bound by address; `nir_ldc_nv` | `vulkan/nvk_nir_lower_descriptors.c` |
| 256-bit image descriptor | 32-byte texture header (TIC) in the texture header pool | `nil_descriptor`, `vulkan/nvk_descriptor_types.h` |
| DISPATCH_DIRECT | QMD (≤384 B, 256-aligned) + `SEND_PCAS` | `vulkan/nvk_cmd_dispatch.c`, `nak_fill_qmd` |
| SDMA / CP DMA | DMA copy class (`90B5` family), inline-to-memory | `vulkan/nvk_cmd_copy.c` |
| RGP / SQTT | none. `vk_push_print` pushbuf dumps | `headers/nv_push.h` |
| `VM_ALWAYS_VALID` BOs | `NOUVEAU_GEM_DOMAIN_NO_SHARE` BOs | `winsys/nouveau_bo.c` |

## §5.1 Memory

**The kernel fact holds, more strongly than on AMD.** The proposal's whole
pointer-only design hangs on submitting without a buffer list. nouveau's
current uAPI has no buffer list to leave empty: `DRM_NOUVEAU_EXEC` takes an
array of `{va, va_len}` pushes and syncobjs, nothing else
(`nvkmd_ctx_exec { addr, size_B }`). Residency comes from `VM_BIND` plus
`NO_SHARE` BOs, which share the VM's reservation object — the nouveau
counterpart of amdgpu's always-valid registration, and NVK's default for every
non-exported allocation. Requires the VM_BIND uAPI (kernel ≥ 6.6; the host
runs 7.0).

`kgpMalloc` is `nvkmd_dev_alloc_mapped_mem` / `nvkmd_dev_alloc_mem`, which
already does GEM-new + VA alloc + bind in one call.

| KGP type | Proposal | nvkmd flags |
|---|---|---|
| default | ReBAR VRAM, CPU-mapped WC | `LOCAL \| CAN_MAP` (VRAM, may migrate to GART; WC map through BAR1) |
| `KGP_MEMORY_GPU` | VRAM, unmapped | `VRAM` |
| `KGP_MEMORY_READBACK` | cached GTT | `GART \| CAN_MAP` |

**Delta D1 — BAR1 is 256 MB.** On this card CPU-mapped VRAM goes through a
256 MB aperture; nouveau makes that work by evicting mappings, at a cost.
The proposal already says the fallback is "reported as a device property";
on NVIDIA the property has to be a size (`mappedVramBytes`), not a boolean,
and the M0 benchmark must record which placement it measured. Implemented:
`KgpDeviceProperties::mappedVramBytes`. The BAR is advertised as resizable and
the kernel can resize it between the two drivers; trying that is an optional,
separately agreed step of W0 (`test-host.md`, "BAR1 resize").

`KGP_MEMORY_GPU` has no CPU pointer to return, so `kgpMalloc` returns the GPU
pointer for that type, as the blog's prototype does. CPU pointers and GPU
pointers are different address spaces sharing one `void*`; the implementation
keeps a lookup tree for each.

`kgpHostToDevicePointer`: nvkmd keeps a VA-keyed lookup
(`nvkmd_dev_lookup_mem_by_va`) usable for `KGP_DEBUG` liveness checks; the
CPU-pointer-keyed direction needs our own small tree, as the proposal says.

`kgpFree` deferred against the timeline: unchanged.

## §5.2 Textures and the heap

This is where NVIDIA fits the proposal better than AMD does. The hardware's
native texture binding model *is* an index into a table in memory.

- The **texture header pool** is an array of 32-byte TIC entries at an address
  set by `SET_TEX_HEADER_POOL_A/B/C` (address, max index). The **sampler
  pool** is the same for 32-byte TSC entries (`SET_TEX_SAMPLER_POOL_*`). NVK
  sets both once per queue (`vulkan/nvk_queue.c`).
- A shader's texture handle is `image_index:20 | sampler_index:12`
  (`nvk_sampled_image_descriptor`), passed to NAK as
  `nir_tex_src_texture_handle`. Storage images use the same pool on
  Maxwell and later.
- `kgpSetActiveTextureHeapPtr` is `SET_TEX_HEADER_POOL` pointed at user
  memory. `kgpTextureViewDescriptor` is NIL's TIC fill writing into the
  caller's `KgpDescriptor256` — the size matches exactly.
- NonUniform indexing needs no waterfall: the handle is data.
- `kgpTextureSizeAlign` / `kgpCreateTexture`: `nil_image_init`; allocation is
  already separate from layout in NIL.

**Delta D2 — two heaps, with sizes.** NVIDIA has a separate sampler pool
capped at 4096 entries, and the hardware is told each pool's entry count.
The API needs a sampler-heap setter and both setters need a count. Image
indices are 20 bits (1 M entries).

**Delta D3 — sampler descriptors are 256-bit here**, not 128. Make the
sampler blob type `KgpDescriptor256` too; AMD uses half of it.

D2 and D3 are implemented as written: `kgpSetActiveTextureHeapPtr` and
`kgpSetActiveSamplerHeapPtr` both take an entry count (at most 2^20 and 4096,
the two halves of the hardware's 32-bit texture handle), and
`kgpTextureViewDescriptor` and `kgpSamplerDescriptor` both write a
`KgpDescriptor256`. The shader side (`kgp_nir_lower_heap.c`) is about forty
lines of real work: the array index *is* the handle, so there is nothing to
load.

**Delta D4 — tiled memory.** NVIDIA wants the page-table entries of image
memory tagged with the surface's PTE kind (`nvkmd_dev_alloc_tiled_mem`,
`pte_kind`, `tile_mode`). "A texture is a layout over memory the caller
already owns" therefore has a condition: memory destined for a
depth/compressed/MSAA surface must be allocated with that kind.
`kgpTextureSizeAlign` has to return the required kind and `kgpMalloc` has to
accept it (or `KGP_MEMORY_GPU` allocations take a texture desc hint).

Correction after reading the source for N2: "generic" is not kind 0. On
Turing every tiled colour image gets kind 6, ordinary allocations are kind 0,
and NVK bridges the two by mapping the image's pages a second time through a
kind-6 VA range (the kind is a property of the mapping, not of the memory).
Whether a colour texture works at a kind-0 address is not answered by the
source. Hardware answers it (2026-11-21, TU104, GSP 570.144): **it does** —
a tiled RGBA8 texture in plain `kgpMalloc` memory fetches, stores and
samples correctly. So for colour D4 asks nothing of the API. It stays open
for depth and multisampled surfaces, which is where N3 meets it.

`KGP_HAZARD_DESCRIPTORS`: the hardware caches headers; NVK's barrier code has
`NVK_BARRIER_INVALIDATE_TEX_DATA` for it.

Render targets as the retained residue: same on NVIDIA (`SET_COLOR_TARGET_*`,
`SET_ZT_*` are CPU-emitted method state).

## §5.3 Pipelines

Compile path: `spirv_to_nir` → `nak_preprocess_nir` → KoreGP lowering →
`nak_compile_shader` (which runs `nak_postprocess_nir` itself) →
`nak_shader_bin` written to GPU memory through a transient mapping. On Volta
and later the QMD and the 3D program binds take absolute shader addresses, so
no program-region base is needed for TU104. Implemented in `kgp_pipeline.c`;
the device is refused below Turing.

Two things come along from NVK that the proposal does not mention: constant
data the compiler could not turn into immediates lives after the shader code
and is bound as constant buffer 1, and shader-local (scratch) memory is a
device-wide area sized by the largest shader, reprogrammed at submit when it
has grown.

**Delta D5 — the root convention.** There are no SGPRs. The proposal does not
pin down how SPIR-V spells "the root pointer"; this implementation fixes it
as the one spelling that also works on the layered Vulkan backend (§9) and
that sgpu already uses:

> A push-constant block whose first member is a `PhysicalStorageBuffer`
> pointer. Compute reads it at offset 0. For raster, the vertex-side stage
> reads offset 0 and the pixel stage reads offset 8.

Natively:

1. **v0 (N1, implemented):** the driver writes a 64-byte per-dispatch *root
   record* (`struct kgp_root_record`: root pointer, group counts) and binds
   it as cbuf 0. `kgp_nir_lower_root.c` (80 lines) turns `load_push_constant`
   into `nir_ldc_nv(cbuf 0, …)`; the dereference is an ordinary 64-bit global
   load (`phys_ssbo_addr_format = nir_address_format_64bit_global`, as NVK
   configures it). One indirection. A push-constant block larger than the one
   pointer is rejected at pipeline creation, as are descriptor-set resources.
2. **v1 (was implemented behind `KGP_DEBUG=rootv1`; correct on hardware,
   measured in three windows, not adopted, and removed on 2026-11-24:
   `results-n1.md`):** bind
   the *user's root struct itself* as cbuf 2. The QMD takes `{index, addr,
   size}` per cbuf, so reads of root fields become `c2[offset]` with no
   pointer chase — and on NVIDIA a cbuf value can be an instruction operand,
   so they stop being loads at all: saxpy drops from 25 instructions to 16
   (`results-n1.md`). This is the NVIDIA counterpart of the SGPR-preload
   optimization. Cost: root pointers must be 64-byte aligned on Turing
   (`nvk_min_cbuf_alignment`), a cbuf window is at most 64 KiB, only
   `readonly` root blocks qualify, and the constant cache has to be
   invalidated (D7). The proposal's validation layer already lists
   "root-pointer alignment".

**Heap convention in SPIR-V:** `RuntimeDescriptorArray` at fixed bindings —
set 0 binding 0 textures, binding 1 samplers. Same on the layered backend.
The lowering turns the array index straight into the 20-bit/12-bit handle.

**Specialization**, including pointer-valued constants patched as literals
before optimization: NIR-level, vendor-neutral, unchanged. Here: a list of
`{SpecId, 64-bit value}` pairs handed to `spirv_to_nir`, so a pointer-valued
constant is a 64-bit one. Scalar constants only.

No pipeline cache: unchanged. NVK's `vk_pipeline_cache` is simply not used.

## §5.4 Commands

Recording appends methods to a pushbuf chunk (`struct nv_push`); submission is
`nvkmd_ctx_exec` with one `{addr, size}` per chunk.

`kgpDispatch(cb, dataGpu, groups)` on Turing:

1. Write the root record into the upload area.
2. `nak_fill_qmd` — shader address, group counts, cbuf table — into a
   256-aligned slot.
3. `SEND_PCAS_A(qmd_addr >> 8)` + `SEND_SIGNALING_PCAS_B(invalidate, schedule)`.

NVK already has this shape with the Vulkan state stripped out:
`nvk_cmd_dispatch_with_root` (used for its internal shaders). That function
and `nvk_cmd_upload_qmd` are the extraction ancestors for `kgp_cmd.c`.

**Delta D6 — a dispatch writes 320 bytes of memory**, not a register and a
packet: a 256-byte QMD and the 64-byte root record, in one upload. That is
the hardware's dispatch model, not a layering artifact, and the Vulkan driver
pays more of it (its root table is several times larger); the comparison
stays fair. With root v1 the root record disappears and only the QMD remains.

**Finding D9 — the pipeline can carry its QMD.** NVK builds the QMD on every
dispatch because which constant buffers it names depends on the bound
descriptor state. Under KoreGP's conventions the cbuf table is fixed, so the
whole QMD except the group counts and the root record's address is known at
pipeline creation. `kgpDispatch` copies a template and patches five words,
using layout queries NAK already exports for indirect dispatch. Measured
offline, this is most of a ~7× reduction in CPU per recorded dispatch
(`results-n1.md`). It is the NVIDIA counterpart of "PSO creation is one
compiler invocation and dispatch is a register and a packet": not an API
change, but a consequence of the API to state in the proposal.

Indirect dispatch: NVK patches the QMD's size fields from the argument buffer
with an inline-to-memory copy (`nvk_cmd_compute_indirect_copy`) whose data the
command fetcher reads straight from the caller's memory. Implemented the same
way as `kgpDispatchIndirect(cb, dataGpu, groupsGpu)`: two 12-byte GPU-side
copies, one into the QMD and one into the root record for `gl_NumWorkGroups`.
Because the fetcher runs ahead of execution, arguments written by an earlier
dispatch need `KGP_HAZARD_DRAW_ARGUMENTS` on the barrier between them — the
proposal's "command-processor prefetch stall", here a `SET_REFERENCE`.

Draws (N3), from a read of `nvk_cmd_draw.c`, `nvk_shader.c` and
`nvk_cmd_clear.c` at the pin:

- **A draw is a macro call.** On Turing NVK has no draw outside the GPU's
  macro processor (MME): `nvk_CmdDrawIndexed` is `CALL_MME_MACRO(DRAW_INDEXED)`
  with six parameters, and the macro emits one run of four methods starting
  at `SET_DRAW_CONTROL_A` (hardware instancing, no begin/end). The index
  buffer bind and a clear are macros too. KoreGP uploads NVK's 31 macros
  unchanged, by calling `nvk_build_mme()`, and calls them; indirect and
  multi-draw are macros that read the argument buffer on the GPU.
- **The 3D init is large and mostly unexplained defaults** (4014 methods
  with the macro code). Which of them the hardware needs is not knowable
  from the source, so it is taken whole (`kgp_draw.c`) and checked against
  NVK's (`init3d_matches_nvk`).
- **Shader binding is a prebuilt method block per shader**: program address
  (absolute, with a 128-byte header NAK generates in front of the code),
  register count, binding group; for the pixel shader also early-Z and
  sample-shading state. Unused stages get an explicit disable. There is no
  method describing the interface between vertex and pixel shader: it is in
  the two headers.
- **Constant buffers**: five groups (one per stage) of sixteen slots, bound
  by address with four methods and no macro. Addresses must be 64-byte
  aligned, sizes at most 64 KiB.
- **Fixed-function state** is re-emitted in full at the first draw of every
  command buffer, group by group, almost all plain methods.

**Delta D10 — the graphics root is two words in a hardware root table.**
Turing's 3D class has eight 256-byte *root tables* that every stage reads as
constant buffers 24 to 31, written inline from the pushbuf
(`SET_ROOT_TABLE_SELECTOR` + data). NVK keeps its whole root descriptor
table there, push constants in the last one. So for raster the proposal's
"root SGPRs for both stages" has a direct NVIDIA form, closer than compute's
(D5, D6): `kgpDrawIndexedInstanced(cb, vsData, psData, …)` writes the two
pointers into root table 7 with one five-word packet, and the shaders read
them as `c[31][0]` and `c[31][8]`. No upload, no memory write, no per-draw
constant-buffer bind. It is also exactly where NVK puts a 16-byte Vulkan
push-constant block, which keeps D5's promise that the same SPIR-V runs on
the layered backend and gives the pushbuf comparison a twin.

Copies (`kgp_copy.c`): `kgpMemCpy`, `kgpCopyToTexture` and
`kgpCopyFromTexture` drive the DMA copy class directly, which is the
proposal's "dedicated hardware path" (its SDMA). The class understands the
tiled layout, so a texture copy is one `LAUNCH_DMA` per layer or depth slice
with the block size NIL chose. This is where KoreGP and NVK part ways in
Mesa 26.3: on a queue that has compute, NVK copies with internal compute
shaders (`nvk_cmd_meta.c`) and reaches the copy class only on its
transfer-only queue. KoreGP's one channel carries compute and copy
together; the kernel binds the copy class to its subchannel when the channel
is made, so there is no init to push. That a transfer stage is a different
engine, not a shader in disguise, is also why `kgpBarrier` treats it
differently from NVK's main queue (§5.5).

**Depth, and the two state objects (R7).** A depth target is programmed
as NVK does it (`SET_ZT_*`), from a `KgpTexture` in `KGP_FORMAT_D32_FLOAT`.
The proposal's depth-stencil state object is exactly its "register-write
packet snippet": `kgpCreateDepthStencilState` builds NVK's whole
depth-stencil group (`nvk_flush_ds_state` with every bit dirty) once, and
a draw copies it. The blend state object is not a snippet yet:
NVK's blend methods are per colour target, so their number is the render
pass's, and they are emitted from the stored equation. Neither is sent by
`kgpSet*State`; like NVK's dynamic state they go out with the next draw,
whole after a pipeline or render pass change, alone otherwise. That is the
"three state snippets" of shadow state §5.4 allows, and it keeps the stream
comparable with NVK's line for line.

Three things the read turned up:

- NVK's clip control depends on the depth format: with a floating-point
  depth target the extra clamp of depth to 0..1 is off
  (`nvk_flush_rs_state`, `z_clamp_zero_one`). KoreGP follows it.
- ZCULL, the coarse depth test. With a recent nouveau NVK sets up a ZCULL
  region whenever a pass clears its depth target, even for an image without
  ZCULL storage. KoreGP does not (the region select stays at "none", which
  is what NVK does on kernels that do not report the ZCULL geometry). It is
  an optimization, and the one place where the two streams differ by
  design; the comparison drops those methods on both sides.
- Delta D4 again. NIL gives Turing depth formats their own PTE kinds, not
  the generic one colour gets. Measured: a `D32_FLOAT` depth target works
  at a kind-0 address like colour does; a `D24_UNORM_S8_UINT` one does not
  (the hardware raises "3D-Z KIND Violation" and the channel is killed). So
  for that format `kgpCreateTexture` does what NVK does for every tiled
  image: it maps the pages a second time through a VA range tagged with
  NIL's kind and uses that as the texture's address. The cost to the API is
  that such a texture must lie in `KGP_MEMORY_GPU` memory and start on a
  page of its allocation, and that its address is not the caller's pointer:
  the proposal's "a texture is a layout over any memory" does not hold for
  this format on this hardware.

**GPU-driven multi-draw with per-draw root data (R9b).** This is the one
draw the proposal has and Vulkan does not, so there is no NVK entry point to
extract; but NVK's indirect draws already run on the 3D engine's macro
processor, which can read GPU memory and write root tables, and that is all
it takes. `kgpDrawIndexedInstancedIndirectMulti` is one call of a macro of
KoreGP's own (`kgp_mme.c`, slot 31, after NVK's 31): it reads the draw count
and each `KgpDrawIndexedArgs` from memory as NVK's indirect-count macro
does, and before each draw writes the two stage root pointers, stepped by
the caller's strides, into the hardware root table the shaders read (delta
D10). So the proposal's "indirect SGPR loads" have a direct counterpart
here, with no CPU involvement per draw and no extra memory read in the
shader. The cost is per draw on the macro processor: four root-table words
on top of what an NVK indirect draw does.

The macro is pushed after the rest of the 3D init, not with NVK's macros,
so that the init stays NVK's word for word up to that point.

Per-queue init is `nvk_push_dispatch_state_init` (compute, ~60 lines) and
`nvk_push_draw_state_init` (3D, large). Compute-only bring-up needs only the
first.

## §5.5 Barriers

NVK already reduces every Vulkan barrier to ten flags
(`enum nvk_barrier`, `vulkan/nvk_cmd_buffer.c`) and ignores the resource lists
beyond that. `kgpBarrier` (`kgp_barrier.c`, compute and copy subchannels so far) keys
those flags directly on the stage pair. With no access masks to narrow it, a
shader stage is assumed to have written, and to be about to read, anything a
shader can; for compute → compute this produces exactly the three methods NVK
emits for a `SHADER_WRITE` → `SHADER_READ` memory barrier.

| KGP | NVK flag | Methods |
|---|---|---|
| any shader-write → shader-read | `FLUSH_SHADER_DATA` | `INVALIDATE_SHADER_CACHES {data, flush_data}` (implies wait-for-idle) |
| otherwise ordering only | `WFI` | `WAIT_FOR_IDLE` on the last subchannel |
| `KGP_STAGE_TRANSFER` as producer | `WFI`, copy-subchannel form | an empty, non-pipelined `LAUNCH_DMA` on the copy engine |
| `KGP_STAGE_TRANSFER` as consumer | — | nothing: the copy engine has no cache of what shaders wrote |
| `KGP_HAZARD_DESCRIPTORS` | `INVALIDATE_TEX_DATA` | texture header / sampler cache invalidate |
| `KGP_HAZARD_DRAW_ARGUMENTS` | `INVALIDATE_MME_DATA`, `INVALIDATE_QMD_DATA` | MME / QMD data cache invalidate |
| root data rewritten in place | `INVALIDATE_CONSTANT` | constant buffer cache invalidate |
| CPU ↔ GPU through GART | `HOST_WFI_*_SYSMEM` | `MEM_OP_D` L2 sysmem flush/invalidate |

No layout transitions on this path either, with one footnote: NVK's
`nvk_cmd_image_layout_transition` does exactly one thing, zero-initialising
ZCULL data the first time a depth image is used, because loading garbage ZCULL
kills the context. KoreGP either leaves ZCULL off (N3 default) or clears it in
`kgpCreateTexture`.

**Delta D7 — cbuf caching is a hazard only under root v1.** Under v0 the one
constant buffer a shader sees is the driver's own root record, written fresh
for each dispatch exactly as NVK writes its root table, and NVK does not
invalidate the constant cache for that; the user's root struct is read with
global loads, which the ordinary shader-data flush covers. So v0 needs no new
hazard bit and `kgpBarrier` does not touch the constant cache. Once the user's
root struct is itself bound as a constant buffer (v1), rewriting it in place
between dispatches does need the invalidate: either a `KGP_HAZARD_ROOT_DATA`
bit or `nak_set_invalidate_cache` per QMD. N1.H decided it: v1 was not
adopted, so the hazard bit is not needed.

Split barriers (written 2026-11-24, correct on the GPU in the thirteenth
window): the proposal's
"RELEASE_MEM and WAIT_REG_MEM against a raw user VA" has a direct
counterpart in the channel's semaphore methods (class C36F, `SEM_ADDR`,
`SEM_PAYLOAD`, `SEM_EXECUTE`), which NVK does not use. `kgpSignalAfter` is
the producer half of `kgpBarrier` followed by a release with wait-for-idle:
the GPU stores a 64-bit payload at the address, or the unsigned maximum of
it and what is there (the method's reduction operation). `kgpWaitBefore`
is an acquire, which holds the channel until the memory equals the payload
or is at least it, followed by the consumer half. The memory is ordinary
`kgpMalloc` memory, so the CPU can release a waiting GPU by writing it.
What the hardware asks: an address below 2^40, 4-byte aligned (the API
says 8, for the 64-bit payload). The proposal's "caller's atomic op and
compare" is here two of each (set and max; equal and greater-or-equal);
the method has more.

Mesh shaders (written 2026-11-24, correct on the GPU in the thirteenth
window): NVK has them on
Turing and KoreGP takes them as they are. A mesh shader without a task
shader is bound in the vertex shader's slot and binding group, so root
table 7 serves it as it serves a vertex shader; its group counts are in
root table 0, where NVK's mesh draw macros put them and where
`gl_NumWorkGroups` is read. Per-primitive outputs need a second program
header for the geometry stage, which NAK generates. No task shader:
`kgpDrawMeshlets` takes the group counts a task shader would have
computed.

MSAA (written 2026-11-24, correct on the GPU in the thirteenth window): the sample count is a
property of the texture's layout (NIL) and of the render pass; sample
locations are Vulkan's standard ones, sent as NVK sends them. Delta D4
for multisampled textures is not asked: they are mapped under their PTE
kind, as NVK maps them. The proposal has no resolve; here a shader reads
the samples through the heap.

## §5.6 Presentation

Unchanged in idea, and built as written (`kgp_present.c`, R10): a hidden NVK
device owns the `VkSwapchainKHR`; each swapchain image's memory is exported
as a dma-buf, imported into KoreGP's device (`nvkmd_dev_import_dma_buf`) and
wrapped as a `KgpTexture` whose layout is the `nil_image` NVK computed, so
"the handle crosses cleanly" is literal: one NIL on both sides. The texture
is mapped under its PTE kind, like the D24S8 case of delta D4.

What is particular to this implementation:

- No Vulkan loader. NVK is linked into `libkoregp.so` and called through
  Mesa's dispatch tables. The cost: the library now contains all of NVK
  and exports its three `vk_icd*` symbols, though nothing loads it as an
  ICD.
- Two surfaces: `VK_KHR_display` (the first connected display at its first
  mode, no compositor, the process becomes DRM master) and
  `VK_EXT_headless_surface` for tests. The private Mesa build has no
  Wayland or X11 platform, and the test host has no compositor running
  during a window anyway.
- The swapchain format is `B8G8R8A8_UNORM`, the one both surfaces offer;
  hence `KGP_FORMAT_BGRA8_UNORM`.
- `kgpPresent` does not wait for KoreGP's rendering; the caller waits on
  its semaphore first. Handing a sync object to the hidden device is the
  upgrade.
- Offline, two opens of the DRM shim have to share a buffer:
  `mesa-patches/0004` gives the shim dma-buf export and import.

The API is four calls (`kgpCreateSwapchain`, `kgpAcquireSwapchainTexture`,
`kgpPresent`, `kgpDestroySwapchain`); the proposal names none.

## §6 API deltas, collected

| # | Change to the header | Reason |
|---|---|---|
| D1 | device property `mappedVramBytes` | BAR1 aperture is a size |
| D2 | `kgpSetActiveSamplerHeapPtr`; both heap setters take a count | separate sampler pool, pools have sizes |
| D3 | sampler descriptor is `KgpDescriptor256` | TSC is 32 bytes |
| D4 | none for colour textures (hardware, 2026-11-21); for depth/MSAA, possibly: size/align query returns a memory kind and allocation accepts it | PTE kind |
| D5 | SPIR-V root ABI fixed as push-constant-held pointer (the 64-byte root alignment was v1's, which was not adopted) | no SGPRs |
| D7 | none: `KGP_HAZARD_ROOT_DATA` would have been needed only with root v1 | constant-buffer cache |
| D10 | none: raster root pointers are push-constant offsets 0 (vertex) and 8 (pixel), as D5 already says | hardware root tables |

Smaller choices the N1 header made where the proposal's sketch was silent:
`kgpSubmit` and `kgpWaitSemaphore` return `bool`; `kgpCreateDevice` takes a
device index and each device has one queue (`kgpGetQueue`);
`kgpSetPipeline` is a separate call, as in the blog; `kgpDestroyPipeline`
and `kgpFree` both defer against the device timeline.

These belong in the proposal once N1/N2 confirm them on hardware. Nothing here contradicts
the proposal's claim that the model is vendor-neutral; D2–D4 are exactly the
"two RDNAs in a trenchcoat" findings M4 was meant to produce.

## §7 Validation and tooling

- `KGP_DEBUG=1` checks: vendor-neutral, and written (`kgp_validate.c`,
  test `kgp_validate`), the NIR-injected bounds checks on heap indexing
  included since 2026-11-24 (`kgp_nir_lower_heap.c`; correct on
  the GPU, twelfth window). A pipeline created under validation compares each heap index
  with the number of entries the command buffer gave the heap, and skips
  the access when it is past the end: a read gives zero, a write is
  dropped. The two counts reach a compute shader in the root record and a
  raster shader behind its root pointers, so a command buffer that uses
  textures under validation has to set its heaps itself; relying on an
  earlier command buffer's reads as an empty heap. On this hardware the
  texture header pool has a size register too (`SET_TEX_HEADER_POOL_C`);
  what the GPU does with an index past it is not known here, and the
  check does not depend on it. Two things differ from the proposal's
  wording:
  - There is no dispatch table to interpose; the entry points are plain
    exported functions. Each one that takes a GPU pointer has one branch
    on a flag of the command buffer, and the mode costs nothing more when
    off (dispatch recording stays at 49 ns offline; 60 ns with it on).
  - "Root-pointer alignment" has no number in the proposal. The check is
    4 bytes (NAK loads through a root in 32-bit words).
  What it checks: every GPU pointer argument is inside live `kgpMalloc`
  memory, with the bytes the call will touch where the CPU knows them
  (copies, heaps of `count` entries, direct index counts, textures); the
  stage and hazard masks of `kgpBarrier`; and at `kgpSubmit`, that nothing
  the command buffer was recorded against has been freed since (each
  allocation has an id, so a new allocation at the old address does not
  hide it). What it cannot see: pointers stored in memory, anything the
  GPU computes, destroyed handles.
- Profiling: no RGP. Pushbuf dumps via `vk_push_print`; NAK assembly dumps via
  `nak_compile_shader(dump_asm)`.
- The DRM shim gives a no-hardware run mode (see `test-host.md`), which is what
  makes golden-file tests of command emission possible.

## How KoreGP links against NVK

`nvkmd`, `nouveau_ws`, NIL, NAK, the class headers and the MME builder are all
below NVK's Vulkan object model. `src/koregp/` uses them unmodified:

- link `libnvk.a` (static; only referenced objects are pulled in) and its
  dependency list from `vulkan/meson.build`;
- call `nvkmd_*`, `nil_*`, `nak_*`, `nv_push` directly;
- copy-and-cut from `nvk_cmd_dispatch.c`, `nvk_cmd_draw.c`, `nvk_cmd_buffer.c`
  (barriers), `nvk_queue.c` (pool setup) into `kgp_*.c`, keeping each file
  diffable against its ancestor, as the proposal requires.

**Delta D8 — the audit rule needs a carve-out.** `nvkmd` returns `VkResult`
and logs through `vk_errorf(log_obj, …)`. Reusing it unmodified means
`libkoregp` links Mesa's Vulkan *runtime utilities* for types and logging, with
no Vulkan objects created. Settled at N0: `log_obj = NULL` is safe
(`vk_log.c` falls back to plain logging), and the one place nvkmd wants
`vk_sync` objects (signals on submit) only reads a syncobj handle out of them,
so `kgp_sync.c` passes two stack structs. No `vk_instance`, `vk_device` or
any other Vulkan object exists in a KoreGP process. A purist build would wrap
or fork nvkmd; not worth it for a test implementation.

Nothing is patched in Mesa except the build hook (`subdir('koregp')` + an
option) and one winsys probe that asserted on the DRM shim. Both live in
`mesa-patches/`. The repository root is symlinked into the tree as
`src/koregp`.
