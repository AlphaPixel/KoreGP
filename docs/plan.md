# Implementation plan: KoreGP test implementation on NVK

Scope: prove the proposal's central claim on the hardware we have — a native
frontend below the Vulkan entry points, on a TU104 through Mesa's NVK
internals — up to the proposal's M0 exit test, with M1/M2 sketched. Spec:
`../KoreGP-website/README.md`. NVIDIA restatement: `nvk-mapping.md`. Host
rules: `test-host.md`. Measurements: `results-n1.md`.

Milestones here are named so they cannot be confused with the proposal's:

| Here | Proposal | What | State |
|---|---|---|---|
| W0 | — | driver-swap shakedown on the host | **done** 2026-11-21: dead-man and swap both clean. (The first window, 2026-11-20, needed manual recovery; scripts corrected since.) BAR1 resize not attempted |
| N0 | — | skeleton: build hook, device open, offline tier | done |
| N1 | M0 | compute | **done**: exit criteria met on the TU104 (`results-n1.md`). Root v1 measured twice and not adopted; v0 stays |
| N2 | M1 | textures and the heap | **done on hardware** (2026-11-22): heap test in both layouts, GPU-side copies (T7) in both layouts; D4 answered for colour |
| N3 | M2 | raster | **first triangle correct on hardware** (2026-11-22, R1–R6), as is a heap texture sampled in a pixel shader (R8). Indirect draw (R9a) correct on hardware too. Depth and the depth-stencil and blend state objects (R7) correct on hardware (fourth window). GPU-driven multi-draw (R9b) correct on hardware (fifth window, 2026-11-23). Stencil (R7b) correct on hardware only with the texture mapped under its PTE kind, which is now what the code does. Presentation (R10) correct on hardware to a headless swapchain (seventh window); on the monitor not yet run |
| — | M3, M4, §9 | not planned here (see "Deferred") | |

## State on 2026-11-23

On hardware, eleven windows (`results-n1.md`). The eleventh (2026-11-23
19:17) presented 120 frames on the monitor through `kgp_present display`,
which completes R10 as far as a test can check it. The seventh (2026-11-23
11:46) ran specialization constants and presentation to a headless
swapchain on the GPU for the first time; both are correct. The fifth (2026-11-23 09:13)
ran multi-draw (R9b) and stencil (R7b) for the first time: multi-draw is
correct; stencil failed with its texture at an ordinary address ("3D-Z KIND
Violation") and is correct with the texture mapped under its PTE kind, so
D4 holds for colour and `D32_FLOAT` and not for `D24_UNORM_S8_UINT`. The
mapping was made the rule for that format after the window, and the sixth
window (2026-11-23 09:50) passed every step with it. The fourth passed every step
of the suite as it then stood, depth (R7) included. The first (2026-11-21, compute
only channel) passed every step: Vulkan twins, the N1 tests, root v1, the
heap test in both layouts, dispatch cost, `strace`. The second (2026-11-22,
channel with 3D, compute and copy engines) passed all of that again and,
for the first time: GPU-side copies in both layouts (T7), graphics
pipelines, the first triangle (R1–R6) and a textured draw (R8). One step
failed, the indirect triangle (R9a), on a bug in the test: it freed the
draw arguments before submitting. Fixed, and the third window (2026-11-22
12:44) passed it. That window also showed the second's benchmark outlier to
be an order effect: the first GPU benchmark after a swap is slow under
either root convention.

The fourth window (2026-11-22 16:25) ran R7, depth and the two state
objects, for the first time, and it is correct with the depth texture in
ordinary memory: D4 is answered for depth as for colour. Unverified on
hardware is only what the tests do not reach (copies of non-zero mips,
layers and 3D textures; formats other than RGBA8 as render targets).

Next, in dependency order:

- R10 (presentation), decided 2026-11-23: on the real monitor. The code
  is written and passes offline against a headless swapchain (48 offline
  tests). On hardware, in this order, each needing its own go-ahead:
  1. done (seventh window): `kgp_present` headless on the GPU, which
     checks that a swapchain image shared in from the hidden device is
     the memory KoreGP rendered to.
  2. done (eighth window, 2026-11-23 15:00): the display shakedown
     (`w0.sh display`). nouveau drove the monitor at 3840x2160 with the
     console on its framebuffer, unloaded cleanly, and nvidia came back.
  3. `w0.sh present`: the suite in a display window, with
     `kgp_present display` putting the scene on the monitor. Medium.
     Tried once (ninth window, 2026-11-23 16:54): the suite passed with
     modesetting on, but the swapchain was refused before anything was
     shown. Tried again (tenth window, 18:39) with the same result, which
     ruled out the first suspect (node ownership). The cause read from
     Mesa's source: its display WSI needs atomic modesetting, which
     nouveau exposes only when loaded with `atomic=1`. Done with that
     (eleventh window, 19:17): 120 frames presented on the display, the
     first read back and correct, host restored.
- Specialization constants: done, correct on hardware (seventh window).
- The `KGP_DEBUG=1` validation mode is written and passes offline
  (`kgp_validate`; the whole offline suite also passes with it on, with no
  false alarm). Not written: the NIR-injected bounds checks on heap
  indexing. `kgp_validate` passed on the GPU in the ninth window.

Done before the window:

- Host inventoried; safety model written (`test-host.md`).
- Toolchain user-local; Mesa pinned at `mesa-26.3.1`, built NVK-only and
  headless into private prefixes. System Vulkan still reports the NVIDIA
  proprietary driver.
- N0 and the offline half of N1: `libkoregp.so` builds inside Mesa's Meson,
  opens the device, allocates, compiles SPIR-V through NAK with the root
  convention, records direct and indirect dispatches, submits. 22 offline
  tests pass, four of them requiring KoreGP's pushbuf to equal NVK's
  (`results-n1.md`); every test program is also clean under
  AddressSanitizer.
- Most of N2: textures as layouts over caller memory, the 256-bit texture
  and sampler descriptor writers, both heaps, the shader-side heap indexing
  and CPU upload, exercised offline by `tests/kgp_heap.c`.
- Dispatch-cost benchmark, measured offline: about 7× less CPU per recorded
  dispatch than Vulkan on the same driver.
- Swap scripts corrected after the first window showed what reading and a
  dry run could not: the restore path's unload of nouveau did nothing, and
  reloading nvidia carelessly sets Ubuntu's udev rules looping.

## Repository layout

```
meson.build             Mesa subdir file: libkoregp.so, then tests/
include/koregp/kgp.h    public header (N1 subset)
src/koregp/             the frontend
tests/                  test programs, shaders, Vulkan twins, benchmark
mesa-patches/           the only changes to upstream Mesa
scripts/push.sh         workstation -> host copy
scripts/host/           everything that runs on the host
docs/                   this plan, the mapping, the host rules, results
```

Mesa is not vendored. `MESA_TAG` in `scripts/host/env.sh` is the pin.
`build-mesa.sh` applies `mesa-patches/` and symlinks **this repository's
root** as `mesa/src/koregp`; the build-hook patch adds
`option('koregp')` and `subdir('koregp')`. Bumping the pin is a deliberate
task: rebuild, rerun the offline tier (the NVK comparison tests will say what
changed underneath), then one window for the hardware tier.

`mesa-patches/`:

| Patch | Why |
|---|---|
| `0001-koregp-build-hook` | `-Dkoregp=true` builds `src/koregp` |
| `0002-nouveau-winsys-vdec-probe` | the winsys asserted while probing for a video-decode class the DRM shim does not have; upstreamable as is |
| `0003-nouveau-shim-transfer-queue` | the DRM shim reports nouveau 1.4.1 instead of 1.3.1, so NVK exposes its transfer-only queue offline; the copy tests' Vulkan twin needs it |

Build flavors (`KGP_FLAVOR`, `env.sh`), each with its own build dir and
prefix: `debug` (asserts on; what the tests run on), `release` (measuring
only) and `asan` (AddressSanitizer + UBSan, offline suite only).

The build stays Meson because it lives inside Mesa's build. A CMake package
config for consumers is §9 material and deferred.

## Two test tiers

| Tier | Runs on | Needs a window | Proves |
|---|---|---|---|
| **offline** (`koregp-offline`) | DRM shim, nvidia still bound | no | it builds, opens a device, allocates, compiles, and pushes what NVK pushes for the same work |
| **hardware** (`koregp-hardware`) | real nouveau | yes (ring 3) | it computes the right answer, and how fast |

The same executables serve both; `KGP_TEST_OFFLINE` makes them skip result
checks. Everything that can be an offline test must be one. Windows are for
running an already-green offline build, not for debugging compile errors.

The offline tier's strongest tool is differential: for each KoreGP test
there is a Vulkan twin using the same SPIR-V, and
`tests/push_matches_nvk.sh` requires the two compute-subchannel pushbuf
dumps to be identical. New command emission should come with a twin.

## W0 — swap shakedown

Blocks: every hardware-tier run. Blocked by: **the owner's go-ahead, with the
owner able to reach the machine**, because a failed restore means a reboot
and the reboot path is fragile. Risk: medium.

**First window, 2026-11-20.** The swap to nouveau worked (GSP firmware
570.144, 8192 MiB VRAM, headless). Every test then failed for one reason
that had nothing to do with KoreGP: the test user could not open the render
node. The restore failed on two script bugs and the attempts to fix it live
set off a load/unload loop in Ubuntu's udev rules; getting back to the
baseline took manual steps by the owner and no reboot. The full account and
the rules it produced are in `test-host.md`, "The first window". The
scripts were corrected and the order below changed: **`deadman` first** (a
three-minute window whose only job is to prove the restore), then `swap`.

**Second window, 2026-11-21.** Step 1 passed: the dead-man fired and the
corrected restore brought nvidia, `nvidia-persistenced` and gdm back
unattended. Step 2 passed at 07:54: swap, every step of the hardware suite,
swap back, host state equal to before. W0's exit is met. Step 3 was not
attempted; nothing so far needs more than the 256 MiB aperture.

Scripted, so a window is one command and a few minutes:

| Step | Command | Passes when |
|---|---|---|
| 1 | `scripts/host/w0.sh deadman` | nvidia and gdm are back with nobody touching anything |
| 2 | `scripts/host/w0.sh swap` | swaps, runs `hardware-suite.sh`, swaps back; host state after equals before |
| 3 (optional) | `scripts/host/w0.sh bar1 13` | as step 1 with BAR1 at 8 GiB; BAR1 is back at 256 MiB afterwards |

`hardware-suite.sh` logs to `~/koregp/logs/<time>/` and runs, in order:
nouveau's dmesg lines; `vulkaninfo`; `kgp_info`; the four Vulkan twins; the
KoreGP tests one at a time (reject, saxpy, prefix sum, scratch, indirect); the
same under root v1; the N2 texture probe (`kgp_heap linear`, then
`kgp_heap`); the copy tests (Vulkan twin, then `kgp_copy linear`, then
`kgp_copy`); the raster tests (`kgp_gfx_pipeline`, the Vulkan triangles,
then `kgp_triangle`, its indirect form and `kgp_textured`); the `strace`
form of the no-bind-per-submit check; the
dispatch-cost and GPU-time benchmarks on the release build. It keeps going after a failure, and
if a KoreGP test failed it reruns saxpy with diagnostics
(`KGP_DEBUG=sync`). Only the Vulkan twins and the five
N1 tests decide N1; the root-v1 and texture steps are reported separately.

Step 2 therefore also runs the N1 hardware exit test. If the Vulkan twins
pass and a KoreGP test fails, the bug is in KoreGP and the window was still
worth having.

Step 3 carries extra risk and is not needed for N1: see "BAR1" in
`test-host.md`.

Exit: steps 1 and 2 clean; the unknowns table in `test-host.md` answered from
the logs; defaults in `to-nouveau.sh` adjusted to match (GSP args, PCI reset);
`results-n1.md` updated with hardware numbers.

If a restore fails: stop, do not reboot, report. Do not run the restore
script repeatedly; that is what escalated the first window.

## N0 — skeleton (done)

| Task | Result |
|---|---|
| N0.1 build hook | `mesa-patches/0001`; root `meson.build` builds `libkoregp.so` against `libnvk` (static; only nvkmd and what it references is pulled in) |
| N0.2 header | `include/koregp/kgp.h`, N1 subset |
| N0.3 device | `kgp_device.c`. D8 settled: `log_obj = NULL` is safe in `vk_errorf` |
| N0.4 shim | the shim already advertises the compute class; what failed was the video-decode probe asserting. `mesa-patches/0002` |
| N0.5 harness | plain C executables under `meson test`, suites `koregp-offline` / `koregp-hardware`; pushbuf dumps through nvkmd's own `NVK_DEBUG_PUSH_DUMP` path (`KGP_DEBUG=push`) |

## N1 — compute (proposal M0)

| Task | File | State |
|---|---|---|
| N1.A memory | `kgp_memory.c` | done. Three memory types; pointer trees; frees and command memory retire against one device timeline |
| N1.B pipeline | `kgp_pipeline.c`, `kgp_nir_lower_root.c` | done. Root convention v0 (D5); constant data as cbuf 1; shader-local memory sized as NVK sizes it; QMD template |
| N1.C commands | `kgp_cmd.c`, `kgp_barrier.c` | done. Chunk recycling on timeline retirement; dispatch and indirect dispatch; stage-pair barriers |
| N1.D sync/submit | `kgp_sync.c` | done. Timeline syncobj per semaphore; `kgpSubmit` = `nvkmd_ctx_exec` + signal |
| N1.E tests | `tests/kgp_{saxpy,prefix_sum,scratch,indirect,reject}.c` | done; pass on hardware |
| N1.F baseline | `tests/vk_baseline.c` | done; pass on hardware |
| N1.G benchmark | `tests/bench_dispatch.c` | measured offline and on hardware |
| N1.H root v1 | `kgp_nir_lower_root.c`, `tests/bench_gpu.c` | implemented behind `KGP_DEBUG=rootv1`; correct on hardware; a third fewer shader instructions; measured in two windows: equal or 5–9 % faster on saxpy, 25–81 % slower on the loop shader. **Not adopted; v0 stays** |

The exit is met (2026-11-21): saxpy, prefix sum, scratch and indirect
dispatch correct on hardware, the benchmark repeated there, and `strace`
showing submits are `DRM_NOUVEAU_EXEC` with no bind.

N1.H is answered as far as this card can answer it. `bench_gpu` was meant
to decide whether v1 becomes the default, and with it D5's 64-byte
alignment rule and D7's hazard bit. The first window showed v1 costs
nothing on a memory-bound saxpy. The second ran a `loop` mode
(`tests/shaders/rootloop.comp`: 53 instructions under v0 with the root
loads inside the loop, 26 under v1 with none) and the same work cut into
1024 dispatches: v1 is 5–9 % faster on saxpy and 25 % (one dispatch) to
81 % (1024 dispatches) slower on the loop shader. v0 stays the default and
D5's alignment rule and D7's hazard bit are not needed for it. The v1 code
is still in the tree behind `KGP_DEBUG=rootv1`; removing it is the owner's
call. Open: why v1 is slower in a single dispatch of the loop shader. The
third window repeated the measurement with the same result.

Specialization constants were added on 2026-11-23: the proposal does not
say how a caller's struct maps to SPIR-V spec ids, so the `constants`
argument is a list of `{SpecId, value}` pairs (`KgpSpecialization`), one
list for all stages of a pipeline, an unknown id refused. The `KGP_DEBUG=1`
validation mode of the proposal's §7 was added the same day
(`nvk-mapping.md` §7), without its shader-side bounds checks.

The risks listed here before the first hardware run did not materialize: a
compute-only channel works (NVK never allocates one; the `KGP_DEBUG=nvkchan`
fallback was not needed), shader-local memory works, and nothing the pushbuf
comparison cannot see (QMD, shader binary, placement) was wrong. Still
untested: mapped VRAM beyond a few MiB through the 256 MiB BAR1. A killed
channel (second window, the indirect test's own fault) did not disturb the
tests after it or the restore.

## N2 — textures and the heap (proposal M1)

Risk: medium. Started ahead of N1's hardware run, on the parts that do not
depend on it. Task list from a read of the Mesa 26.3.1 source (paths under
`src/nouveau/`):

| Task | What | Ancestor | Risk | State |
|---|---|---|---|---|
| T1 | `kgp_texture.c`: desc → `nil_image_init`; `kgpTextureSizeAlign`, `kgpCreateTexture` | `nvk_image_init` (`vulkan/nvk_image.c`) | low | done offline (2D, arrays, 3D, mips; six formats) |
| T2 | `kgpTextureViewDescriptor`: `nil_image_view_descriptor` into caller memory | `nvk_image_view_init_no_alloc` | low | done offline |
| T3 | `kgp_sampler.c`: `kgpSamplerDescriptor`; `kgpSetActiveSamplerHeapPtr` | `nvk_sampler_get_header` (static, so copied with its helpers) | low | done offline; NAK emits bindless `tex` |
| T4 | `kgpSetActiveTextureHeapPtr`, with a count | `nvk_queue_heap_state_bind_image_heap` (`vulkan/nvk_queue.c`) | low | done offline |
| T5 | `kgp_nir_lower_heap`: array index → handle (image 19:0, sampler 31:20) | `lower_tex`, `lower_image_intrin` | medium | done offline; NAK emits bindless `tld` / `sust` |
| T6 | `kgpWriteTexture`: CPU upload into mapped texture memory | `nil_copy_linear_to_tiled`, `nvk_host_copy.c` | low | done offline |
| T7 | GPU copies; the channel gains the copy engine | `nouveau_copy_rect`, `nvk_cmd_copy_memory_ce` (`vulkan/nvk_cmd_copy.c`) | medium | done offline (`kgp_copy.c`, `tests/kgp_copy.c`); pushes what NVK's transfer queue pushes; **correct on hardware in both layouts (2026-11-22)** |
| T8 | PTE-kind alias mapping — only if the probe in T9 fails | `nvk_image_plane_alloc_va`, `nvk_image_plane_bind` | high | **not needed for colour** (T9 passed). Reopens for depth and multisampled surfaces in N3 |
| T9 | hardware probe, then the exit test | — | high | **passed** 2026-11-21, linear and tiled |

Exit: a compute shader writes descriptors into the heap; the next dispatch
samples through them. `tests/kgp_heap.c` is that test: a shader copies two
texture descriptors into the heap, a second fills one texture through
storage-image access, a third reads both textures back with texel fetches,
and a fourth samples one with a nearest and a linear sampler from the
sampler heap (filled from the CPU). It passes on hardware in both layouts,
every texel checked. What is left of N2 is T7's hardware run, and hardware
coverage beyond one 96×40 RGBA8 level: other formats, mips, arrays and 3D are exercised only
offline.

Not covered by a Vulkan twin: there is no Vulkan spelling of "a shader
writes a descriptor", so the pushbuf comparison does not apply to this test.
What NVK and KoreGP share here is NIL itself.

**T7, as built.** Three calls: `kgpMemCpy`, and `kgpCopyToTexture` /
`kgpCopyFromTexture` for one whole mip level of one layer. They use the copy
engine directly, which is the proposal's "dedicated hardware path". NVK does
the same only on its transfer-only queue; on a queue with compute it copies
with internal shaders, so the twin (`vk_baseline copy`) records on the
transfer queue and `push_matches_nvk_copy` compares the copy subchannel: 50
methods tiled, 36 linear, identical. The proposal names only the first two
calls; the third came with the shared code and is what lets the test read a
texture back.

`kgpBarrier` changed with it: `KGP_STAGE_TRANSFER` now means the copy
engine. As a producer it waits for the engine (NVK's transfer-queue form);
as a consumer it invalidates nothing. Before, a transfer stage was treated
like a shader stage, which nothing used.

`tests/kgp_copy.c` chains the three copies through GPU-only memory and then
has a shader fetch every texel, so the hardware run shows the copy engine
writes what the texture unit reads. Not exercised anywhere yet: copies of a
non-zero mip level or layer, and 3D textures.

**D4, answered for colour (2026-11-21): it works.** A tiled RGBA8 texture at
an ordinary kind-0 address fetches, stores and samples correctly on the
TU104 under GSP, so "a texture is a layout over memory the caller already
owns" holds as the proposal states it, and the alternatives below are not
needed. The question as it stood before the window:

On Turing NIL gives every tiled colour
image PTE kind 6 ("generic memory"), not 0, and NVK never places such an
image at an ordinary address: it maps the same pages a second time through a
VA range tagged with that kind and uses the alias as the image address
(`vulkan/nvk_image.c`, `nvk_image_plane_bind`). The source does not say what
happens if the kind is left at 0. So the first hardware step of N2 (T9) is a
probe, and `hardware-suite.sh` runs it at the end of the first window:
`kgp_heap linear` (the untiled layout, to which NIL itself gives kind 0) and
then `kgp_heap` (tiled textures over ordinary `kgpMalloc` memory).

- If it works, "a texture is a layout over memory the caller already owns"
  holds as the proposal states it.
- If it does not, either `kgpCreateTexture` creates the alias (the texture's
  address then differs from `gpuPtr`, `gpuPtr` and size must be page-aligned,
  and textures need a destroy call), or `kgpMalloc` gains a "texture memory"
  type that allocates with the kind up front. One kind covers every Turing
  colour format; depth formats have their own.

Other things the read turned up: at most 4096 samplers and 2^20 textures per
heap (the handle's bit split); texture memory alignment is at least 4096 and
the descriptor needs the base 512-byte aligned; NIL's descriptor fill aborts
on bad input rather than returning an error, so views must be validated
first; a zero-filled heap entry is a valid null descriptor.

## N3 — raster (proposal M2)

Risk: high — `nvk_cmd_draw.c` is 6 kLOC of state and the MME macros are the
draw path. Started; the task list comes from a read of the Mesa 26.3.1
source (`nvk-mapping.md` §5.4 has what it found).

The approach, decided by that read: **stay method-compatible with NVK.**
The hardware's reset state is not documented in the source, a draw on
Turing exists only as a macro, and windows are scarce, so KoreGP sends what
NVK sends for the equivalent Vulkan calls and the offline tier checks it,
as for compute. The macros are not extracted at all: `nvk_build_mme()` is
called for all 31, in NVK's slots.

| Task | What | Ancestor | Risk | State |
|---|---|---|---|---|
| R1 | the channel gains the 3D engine; its init push | `nvk_push_draw_state_init`, `nvk_queue_init_context_state` | medium | **done, runs on hardware**: `kgp_draw.c`; 4014 methods identical to the head of NVK's 3D stream (`init3d_matches_nvk`) |
| R2 | graphics pipeline: VS + PS through NAK, the root convention for two stages (D10), upload with the 128-byte program header, the per-shader bind snippet | `nvk_compile_shaders`, `nvk_shader_upload`, `nvk_shader_fill_push` (`nvk_shader.c`) | high | **done, runs on hardware**: `kgpCreateGraphicsPipeline`; the test shaders read their roots as `c[31][0]` and `c[31][8]` and the instance index from root table 0. The bind block is built but nothing emits it yet, so it is unchecked against NVK until R6 |
| R3 | raster state from `KgpRasterDesc`: topology, cull, front face, one viewport and scissor, sample state, depth off, colour write on and blend off | `nvk_flush_{vi,ia,vp,dr,rs,fsr,ms,ds,cb}_state` | medium | **done, runs on hardware**: emitted at the first draw after a pipeline or render pass change, in NVK's group order. Not yet a prebuilt snippet |
| R4 | render pass begin/end on `KgpTexture` colour targets, with clear; the per-command-buffer 3D prologue | `nvk_CmdBeginRendering`, `nvk_CmdClearAttachments`, `nvk_cmd_buffer_begin_graphics` | medium | **done, runs on hardware**: `kgpBeginRenderPass` / `kgpEndRenderPass`, tiled 2D targets, one sample |
| R5 | `kgpDrawIndexedInstanced`: the two root pointers into a hardware root table, index buffer bind, the draw macro | `nvk_CmdBindIndexBuffer3KHR`, `nvk_CmdDrawIndexed`, `nvk_cmd_flush_gfx_root_desc` | medium | **done, runs on hardware**: 18 pushbuf words per draw, no memory written |
| R6 | first triangle: test, Vulkan twin (dynamic rendering), 3D pushbuf comparison; read back with `kgpCopyFromTexture` | — | high | **done: correct on hardware (2026-11-22).** `tests/kgp_triangle.c`, `vk_baseline triangle`; `triangle_matches_nvk`: 326 3D methods from render pass to draw identical to NVK's, addresses masked |
| R7 | depth: depth target, depth-stencil state objects, blend state objects | `nvk_flush_ds_state`, `nvk_flush_cb_state`, the depth half of `BeginRendering` | high (delta D4 for depth: the PTE kind question again, and ZCULL) | **done: correct on hardware (2026-11-22, fourth window), in ordinary memory, so D4 is answered for depth too**: `KGP_FORMAT_D32_FLOAT`, `KgpRenderPassDesc::depthTarget` with clear, `kgpCreate/Set/DestroyDepthStencilState`, `kgpCreate/Set/DestroyBlendState`; `kgp_triangle depth` (three draws, state changed between them) with twin; 338 methods from render pass to first draw identical to NVK's, ZCULL aside. No ZCULL |
| R7b | stencil: a depth-stencil format, stencil state in the state object, stencil clear | the stencil half of `nvk_flush_ds_state` | low once R7 has run on hardware | **done: correct on hardware (2026-11-23, fifth window) with the texture mapped under its PTE kind; at an ordinary address the hardware raises "3D-Z KIND Violation"**: `KGP_FORMAT_D24_UNORM_S8_UINT`, per-face ops and compare, masks and reference in `KgpDepthStencilDesc`, `clearStencil`; `kgp_triangle stencil` (one draw marks, the next tests) with twin; 339 methods identical to NVK's, ZCULL aside. `kgpCreateTexture` maps this format a second time with NIL's kind, so it must lie in `KGP_MEMORY_GPU` memory on a page boundary (sixth window: passes as a plain step) |
| R8 | textures in pixel shaders: the heaps bound on the 3D engine too | `nvk_queue_heap_state_bind_*` | low | **done: correct on hardware (2026-11-22)**: both heap setters program both engines; `kgpBarrier` handles the 3D engine as producer and consumer; `tests/kgp_textured.c` (no twin: the heap has no Vulkan spelling) |
| R9a | indirect draw | `nvk_CmdDrawIndexedIndirect2KHR` | low | **done: correct on hardware (2026-11-22, third window)**: `kgpDrawIndexedInstancedIndirect`, NVK's macro; `kgp_triangle indirect`, 325 methods identical to NVK's. The index buffer's size is unknown to an indirect draw, so the hardware limit is set to the maximum |
| R9b | GPU-driven multi-draw with per-draw root data | needs a macro of our own, the first that is not NVK's | high | **done: correct on hardware (2026-11-23, fifth window)**: `kgpDrawIndexedInstancedIndirectMulti` (count and arguments read by the GPU, root pointers stepped per draw, `gl_DrawID`); the macro is `kgp_mme.c`, assembled from the Turing branches of NVK's draw macros and loaded into slot 31; `mme_multi_sim` runs it on Mesa's macro simulator against a reviewed expected output; `kgp_triangle multi` is the hardware test. No Vulkan twin. Count, arguments and roots are CPU-written in the test; a compute shader producing them (the proposal's demo) is not written |
| R10 | presentation shim (`kgp_present.c`, hidden NVK device) | — | medium; the first point at which the host's display is involved | **headless (seventh window) and display (eleventh window, nouveau loaded with `atomic=1`) correct on hardware**: `kgpCreateSwapchain` / `kgpAcquireSwapchainTexture` / `kgpPresent`; display (`VK_KHR_display`) and headless surfaces; swapchain images shared as dma-bufs and wrapped as `KGP_FORMAT_BGRA8_UNORM` textures; `kgp_present` (headless, read back and checked) and `kgp_present display`. Needs `mesa-patches/0004` offline. The display path runs only on hardware (the shim has no KMS), and needs nouveau loaded with `atomic=1` |

Dependency order: R1 → R2 → R5 → R6, with R3 and R4 parallel to R2. R6 is
the first hardware exit (a triangle read back correctly). R7, R8 and R9 are
independent of each other after R6; R10 is last.

What the pushbuf comparison can and cannot do here: macro calls hide the
methods a macro emits (they depend on scratch state left by earlier
submits), so the comparison is of macro calls and their parameters, which
is the same thing as long as both sides call the same macros in the same
order. NVK emits every dynamic-state group at the first draw of a command
buffer, in a fixed order, after the push constants and the index buffer;
KoreGP does the same at the first draw after `kgpSetPipeline` or
`kgpBeginRenderPass`, which is what lets the two streams be compared line
for line. The comparison was written twin first: NVK's stream for the
Vulkan triangle was dumped and used as the specification.

What the comparison does not cover, and the hardware run of 2026-11-22 tested
for the first time: the two shader binaries and their program headers
(KoreGP's lowering differs from NVK's by design: roots from `c[31]`, no
descriptor sets), the interface between the two stages that the headers
describe, and everything after the draw (the wait before the readback, and
the copy engine reading a render target).

Limits of what is built, all refused with an error rather than guessed at:
one sample; no depth bias or depth bounds; two depth formats (`D32_FLOAT`,
`D24_UNORM_S8_UINT`, the latter only in GPU-only memory); one stencil reference and one pair of masks for both
faces; targets must be tiled 2D textures; viewport and scissor are the whole
target; one blend equation for all targets, no blend constant, no write
masks; indices are `uint32`; vertex and pixel stages only. A depth-stencil
state that tests or writes depth is refused at the draw in a pass without
a depth target. A pipeline drawn into targets of another format than
it was created for is refused at the draw.

R1 changed the channel a second time after the first hardware window: it has
NVK's engine set (3D, compute, copy), which is the configuration NVK itself
ran on this card in that window. `KGP_DEBUG=nvkchan` is gone, being the
default. libkoregp.so now links most of NVK (the macro builders live in
`nvk_cmd_draw.c`), about 98 MB with debug info.

## Deferred

- Mesh shaders, multi-draw-indirect, split barriers, validation layer beyond
  basic argument checks, SDK packaging (proposal M3).
- AMD/RADV backend (the proposal's reference target; needs AMD hardware).
- The layered Vulkan ICD, ICD loader, CMake config, vcpkg port (§9). The
  shared SPIR-V root ABI (D5) is chosen so this stays possible: the test
  shaders already run unchanged through Vulkan.
- Slang prelude. Tests use GLSL buffer references; add Slang when there is a
  user.

## Feeding back to the proposal

Deltas D1–D9 in `nvk-mapping.md` are candidate edits to the proposal's §5–§6.
They are recorded here, not applied to `KoreGP-website`, until hardware
confirms them.
