# N1 results (compute, proposal M0)

State on 2026-11-21. **N1's exit criteria are met on the TU104.** The
hardware results come from one nouveau window (2026-11-21 07:54, nouveau
with GSP firmware 570.144; logs in `~/koregp/logs/20261121-075415` on the
host). The offline results further down were measured on Mesa's nouveau DRM
shim with the proprietary driver still bound, and are kept because they are
what can be re-run without a window.

A second window (2026-11-22 08:35, logs in `~/koregp/logs/20261122-083504`)
reran everything on the channel as it is now and ran copies and raster for
the first time; see "Second window" below.

An earlier window (2026-11-20 23:46) ran no test; see `test-host.md`.

Pin: Mesa 26.3.1, host kernel 7.0.0-38, Ryzen 7 2700X, RTX 2070 SUPER.

## Exit criteria

| Criterion | Offline | Hardware |
|---|---|---|
| saxpy and prefix sum correct | compile, record and submit cleanly | **correct**, as are scratch and indirect dispatch |
| dispatch CPU cost at parity or better against NVK-Vulkan | better, see below | **better**: 83 – 110 ns against 429 – 610 ns per recorded dispatch |
| a submit carries only pushbuf address ranges; no per-submit bind | VA binds independent of submit count | **confirmed by `strace`**: 8 `VM_BIND` whether 8 or 800 command buffers are submitted |

## On hardware (window of 2026-11-21)

`scripts/host/hardware-suite.sh`, every step passed, no nouveau or GSP
message in the kernel log beyond start-up. Tests ran on the debug build with
result checks on (each compares the GPU's output with a CPU computation);
benchmarks on the release build.

| Step | Result |
|---|---|
| NVK through Vulkan: `vk_saxpy`, `vk_prefix_sum`, `vk_scratch`, `vk_indirect` | pass |
| `kgp_reject`, `kgp_saxpy`, `kgp_prefix_sum`, `kgp_scratch`, `kgp_indirect` | pass |
| the four GPU tests under root convention v1 | pass |
| `kgp_heap linear`, `kgp_heap` (N2: textures through the heap) | pass, both layouts |

Three risks listed in `plan.md` for the first run are settled by this:

- **A compute-only channel works.** KoreGP allocates the compute engine
  alone, which NVK never does; the fallback (`KGP_DEBUG=nvkchan`) was not
  needed.
- **Shader-local memory works** as set up (`kgp_scratch`).
- **A tiled colour texture works in ordinary memory** (delta D4). The tiled
  `kgp_heap` puts 96×40 RGBA8 textures at plain `kgpMalloc` addresses, PTE
  kind 0, where NVK would use a kind-6 alias; CPU upload, texel fetch,
  storage-image write, nearest and linear sampling all return the right
  texels. Not covered: other formats, mips, arrays, 3D, and depth or
  multisampled surfaces, which have their own kinds.

### Dispatch cost

`bench_dispatch`, release build, CPU time of recording only (see "Dispatch
cost" below for the method), alternating the two APIs:

| Run | KoreGP, ns per dispatch | NVK through Vulkan, ns per dispatch |
|---|---|---|
| 2000 buffers × 100 dispatches | 87.2 | 609.9 |
| same | 98.6 | 457.6 |
| same | 109.7 | 448.5 |
| 200 buffers × 1000 dispatches | 83.3 | 429.2 |

Between four and seven times less CPU per dispatch across the runs, about
five in the longest. Both sides cost about
twice what they do on the shim, where upload memory is ordinary RAM.
Other jobs were running on the host during the window, so the spread
between runs is wider than offline. Instruction counts were not taken in the
window; the offline figures stand for those.

Submission costs the same through either API, 8 – 13 µs per command buffer:
it is the kernel's `EXEC` path, which KoreGP does not change.

### What a submit is, to the kernel

`strace` of the benchmark with 8 and with 800 command buffers:

| | `VM_BIND` | `EXEC` |
|---|---|---|
| 8 command buffers | 8 | 27 |
| 800 command buffers | 8 | 819 |

792 more command buffers, 792 more `EXEC` calls, no additional bind and no
other nouveau ioctl whose count changes.

### GPU time of the two root conventions

`bench_gpu`: one saxpy over 16,777,216 elements, wall clock from submit to
the semaphore, 21 runs each, twice.

| Root convention | Best | Median |
|---|---|---|
| v0 | 0.530 ms | 0.532 ms |
| v1 (`KGP_DEBUG=rootv1`) | 0.529 – 0.530 ms | 0.532 ms |

No difference. That is about 380 GB/s of memory traffic (two reads and a
write of 64 MiB each), against a nominal 448 GB/s for this card's memory bus
(my arithmetic from its 256-bit, 14 Gbit/s GDDR6), so this shader is limited
by memory and the nine instructions v1 removes do not show. The benchmark
answers "does v1 cost GPU time" (no) but cannot answer "does v1 save any".
`bench_gpu` does not check its output; the correctness of the same shader is
what `kgp_saxpy` establishes.

## Second window (2026-11-22)

The channel now has the 3D, compute and copy engines (the first window
verified compute alone), so the whole suite ran again. 19 of 20 steps
passed; the one failure was a bug in the test, not in the library.

| Step | Result |
|---|---|
| everything the first window ran (Vulkan twins, N1 tests, root v1, heap in both layouts) | pass |
| `vk_copy`, `vk_copy_linear` (NVK's transfer queue) | pass |
| `kgp_copy_linear`, `kgp_copy` (T7: `kgpMemCpy`, copy to and from a texture, both layouts) | **pass** |
| `kgp_gfx_pipeline` | pass |
| `vk_triangle`, `vk_triangle_indirect` | pass |
| `kgp_triangle` (R6: clear, two instanced indexed quads, both root pointers, readback through the copy engine) | **pass**, all 64×64 pixels |
| `kgp_textured` (R8: a heap texture sampled in the pixel shader) | **pass** |
| `kgp_triangle_indirect` (R9a) | **fail: test bug**, below |

What this settles: the 3D init, the shader binaries and program headers
with roots from `c[31]`, the interface between the two stages, the wait
before the readback and the copy engine reading a render target all work on
the TU104. None of that was visible to the offline comparison.

The indirect failure: the test freed its draw-arguments buffer after
recording the draw and before `kgpSubmit`. `kgpFree` defers against what has
been *submitted*; nothing was in flight, so the buffer was unmapped at once
and the 3D engine's read of it faulted (`fault_addr 0x3ffffdf000`, which is
the buffer's address in the `KGP_DEBUG=vm` log; channel killed, image left
untouched). The library did what `kgp.h` says. The test now frees after the
submit. The indirect draw itself is therefore still unverified on hardware.
The episode also answers an open question from N1: a killed channel does
not affect later tests in the same window (`kgp_textured`, the benchmarks
and the restore all ran clean after it).

`strace`: 12 `VM_BIND` for 8 and for 800 command buffers (four more buffers
at queue creation than before), 27 and 819 `EXEC`.

Dispatch cost, same method as above:

| Run | KoreGP, ns per dispatch | NVK through Vulkan, ns per dispatch |
|---|---|---|
| 2000 buffers × 100 dispatches | 127.6 | 791.7 |
| same | 122.7 | 788.3 |
| same | 144.4 | 788.0 |
| 200 buffers × 1000 dispatches | 108.4 | 789.7 |

Both sides are slower than in the first window by a similar factor (the
host's load differs between windows); the ratio is 5.5 to 7.3. Submission
7 – 11 µs either way.

### Root conventions, second measurement

`bench_gpu`, best / median in ms, 21 runs each:

| Workload | v0 | v1 |
|---|---|---|
| saxpy, 16M elements, 1 dispatch | 5.027 / 5.038 | 0.812 / 0.843 |
| saxpy, 4M elements, 1 dispatch | 0.211 / 0.213 | 0.201 / 0.202 |
| saxpy, 16M elements, 1024 dispatches | 0.696 / 0.723 | 0.636 / 0.665 |
| loop, 1M elements, 1 dispatch | 0.176 / 0.178 | 0.220 / 0.221 |
| loop, 1M elements, 1024 dispatches | 0.245 / 0.249 | 0.444 / 0.448 |

- The first row's v0 figure is an unexplained outlier: the same work took
  0.530 ms in the first window and takes 0.696 ms here when cut into 1024
  dispatches. It was the first GPU benchmark of the window, so order is as
  likely a cause as the convention. The suite now repeats that workload
  last to tell the two apart. Not a basis for any conclusion. (The third
  window showed it is the order.)
- On saxpy, v1 is 5 to 9 % faster.
- On the shader built to favour it (root fields read inside a loop, 26
  instructions against 53), v1 is 25 % *slower* in one dispatch and 81 %
  slower in 1024. The second matches the expected cost of invalidating the
  constant cache on every dispatch; the first is not explained and was not
  investigated.

Nothing here argues for making v1 the default, and one result argues
against. v0 stays.

## Third window (2026-11-22 12:44)

Logs in `~/koregp/logs/20261122-124415`. **Every step passed**, including
`kgp_triangle_indirect` with the test's free moved after the submit, which
closes R9a: the indirect draw is correct on hardware. No fault or killed
channel in the kernel log.

`strace` as before (12 `VM_BIND` either way, 27 and 819 `EXEC`). Dispatch
cost: 109 – 127 ns against 444 – 469 ns over 2000 × 100, 61 ns against
687 ns over 200 × 1000; submission 8 – 12 µs either way.

`bench_gpu`, best / median in ms, in the order run:

| Workload | v0 | v1 |
|---|---|---|
| saxpy, 16M elements, 1 dispatch (first) | 4.916 / 4.927 | 5.061 / 5.088 |
| saxpy, 4M elements, 1 dispatch | 0.167 / 0.169 | 0.166 / 0.170 |
| saxpy, 16M elements, 1024 dispatches | 0.616 / 0.628 | 0.517 / 0.522 |
| loop, 1M elements, 1 dispatch | 0.153 / 0.156 | 0.184 / 0.190 |
| loop, 1M elements, 1024 dispatches | 0.227 / 0.232 | 0.386 / 0.388 |
| saxpy, 16M elements, 1 dispatch (again, last) | 0.599 / 0.601 | 0.593 / 0.598 |

The second window's outlier is an order effect, not a convention effect:
the first GPU benchmark of a window runs about eight times slower under
either convention, and the same workload run last is back at 0.6 ms. The
cause was not investigated (the card's clocks after the swap are the
obvious suspect; the first window did not show it). First-run figures are
to be discarded.

The root-convention picture is the same as in the second window: equal on
a single saxpy dispatch, v1 16 % faster on saxpy in 1024 dispatches, 20 %
slower on the loop shader in one dispatch and 70 % slower in 1024. v0
stays.

## What the offline suite establishes

`meson test --suite koregp-offline`, 38 tests, all passing: the 15 below,
four of them repeated under root convention v1, ten for N2 (`kgp_heap`
and `kgp_copy` in their tiled, linear and root-v1 forms, and for the copies
two Vulkan twins and two pushbuf comparisons), and five for N3
(`init3d_matches_nvk`, `kgp_gfx_pipeline`, `kgp_triangle`, `vk_triangle`,
`triangle_matches_nvk`, and the indirect and textured variants); see
`plan.md`. The hardware
window above ran the 22 that existed then, on a channel with the compute
engine only; everything added since, and the channel as it is now, has not
run on hardware.

| Test | Shows |
|---|---|
| `kgp_info` | device open through nvkmd with no Vulkan object (D8) |
| `kgp_saxpy`, `kgp_prefix_sum`, `kgp_scratch`, `kgp_indirect` | SPIR-V → NIR → NAK with the root convention; memory, pointers, pipelines, recording, barriers, indirect dispatch, multi-buffer submit, semaphores all run |
| `kgp_reject` | bad SPIR-V, an oversized push-constant block, descriptor-set resources, bad sizes, pointers and submits are refused with an error, not a crash |
| `kgp_validate` | (Since 2026-11-24 also the shader-side heap bounds check, correct on the GPU in the twelfth window.) Under `KGP_DEBUG=1`: a CPU pointer as a root, a misaligned root, copies one byte past an allocation, a heap larger than its memory, barriers with empty or unknown stages, a texture over memory too small for it and a command buffer whose memory was freed before submit are all refused; correct use of the same calls is not |
| `vk_saxpy`, `vk_prefix_sum`, `vk_scratch`, `vk_indirect` | the Vulkan twins build and run on the same NVK |
| `push_matches_nvk_*` (4) | **KoreGP pushes exactly what NVK pushes** |
| `no_bind_per_submit` | 4 VA binds whether 8 or 800 command buffers are submitted |
| `kgp_triangle state`, `vk_triangle_state`, `triangle_matches_nvk_state` | added 2026-11-24, offline only so far. Depth bias, a colour write mask and a blend with constant factors, set for the first draw and unset for the second; from the render pass to the first draw KoreGP's methods are NVK's, the bias, the factors, the mask and the constant included |
| `kgp_copy mip`, `layer`, `3d`, their `vk_copy_*` twins and `push_matches_nvk_copy_*` | added 2026-11-24, offline only so far. The copy chain on mip level 1 of a three-level texture, on layer 2 of a three-layer array and on all four slices of a 3D texture. The copy engine's methods are NVK's in each case (50, 50 and 146 of them); addresses are masked in that comparison, so where a level or a layer starts is checked only on hardware, by a shader reading the texture through its own type (`texture2DArray`, `texture3D`, or a one-level view) |
| `kgp_triangle srgb`, `rgba16f`, `rgba32f`, `r32f`, their `vk_triangle_*` twins and `triangle_matches_nvk_*` | added 2026-11-24, offline only so far. The plain triangle scene into a render target of each format, read back at its own texel size; from the render pass to the draw KoreGP's methods are NVK's, the target format among them |
| `kgp_triangle mrt`, `vk_triangle_mrt`, `triangle_matches_nvk_mrt` | added 2026-11-24, offline only so far, and the first test to draw into more than one colour target. Two targets, a pixel shader with two outputs, the second target through a per-target write mask (`KgpRasterDesc::colorWrite`); KoreGP's methods are NVK's, the six unused target slots, the target selection and the combined mask among them |
| `kgp_triangle multigpu` | added 2026-11-24, offline only so far. A compute shader turns four candidate draws into the three records, the count and the per-draw root structs of a multi-draw later in the same command buffer; recorded and submitted. No twin: Vulkan has no spelling for per-draw root data |

Every dispatch in the suite also checks, by assertion, that the QMD built
from the pipeline's template is byte-identical to the one NAK builds.

### The differential check

`tests/push_matches_nvk.sh` runs one test through KoreGP and its twin through
Vulkan, dumps both pushbuf streams, keeps the compute subchannel, and diffs
them. Method for method and value for value they are identical:

| Test | Compute methods | Covers |
|---|---|---|
| saxpy | 18 | channel init, per-buffer prologue, dispatch |
| prefix_sum | 98 | 17 dispatches with a compute→compute barrier between each |
| scratch | 28 | shader-local memory setup (same sizes), two buffers in one submit |
| indirect | 40 | a dispatch whose group counts an earlier dispatch wrote: the barrier for indirect arguments, the two GPU-side copies, the pushbuf continuing in caller memory |

Masked, because they legitimately differ: QMD and shader-local-memory
addresses, the destinations of the indirect copies, and the texture/sampler
pool location (NVK points the pools at its descriptor tables; KoreGP has no
heap until N2 and programs a NULL pool, which is what NVK does when it has
nothing to bind). One method is dropped from both sides: the channel-level
`SET_REFERENCE` in the indirect barrier, which NVK sends on its 3D subchannel
and a compute-only KoreGP channel sends on the compute one.

This does not show the shaders compute the right answer — nothing executes on
the shim — but it removes "KoreGP sends the GPU something NVK would not" as a
cause when the first hardware run happens. The QMD contents and the shader
binaries are not in the pushbuf and are not compared; they differ by design
(different root data layout).

## Dispatch cost, offline

`tests/bench_dispatch.c`: N command buffers of M dispatches each of the saxpy
shader over zero elements. Recording (begin … last dispatch) is timed with the
thread CPU clock, separately from submission; waiting is not timed. The
Vulkan side is `vkCmdPushConstants` + `vkCmdDispatch` per dispatch with the
pipeline bound once — the Vulkan spelling of `kgpDispatch(cb, root, groups)`.

Release build (`KGP_FLAVOR=release`: `-Dbuildtype=release -Db_ndebug=true`),
same Mesa tree, same machine, one API per process.

| | Record, ns per dispatch | User instructions per dispatch (marginal) |
|---|---|---|
| NVK through Vulkan | 330 – 350 | 3308 |
| KoreGP, QMD filled per dispatch like NVK | 270 – 280 | not measured |
| KoreGP, QMD from the pipeline's template | 41 – 53 | 240 |

Instructions are the stable figure (the host runs the `schedutil` governor, so
times move a few percent between runs). They are marginal: the difference
between 2000×600 and 2000×100 dispatches, divided by the million extra
dispatches, which cancels start-up and per-buffer costs.

Where it comes from (`perf record` on both):

- In NVK most of a dispatch is NAK packing the QMD field by field through a
  generic bit-field writer (`set_bit_range_u64`, `to_field_bits`), plus
  copying the root descriptor table.
- KoreGP's first version called the same `nak_fill_qmd` and was about 20 %
  cheaper: a 64-byte root record instead of the root table, and no descriptor
  state to flush.
- The rest is the QMD template. A KoreGP pipeline's constant-buffer bindings
  are fixed by convention, not by descriptor state, so everything in the QMD
  except the group counts and the root record's address is known at pipeline
  creation. `kgpDispatch` copies 256 bytes and patches five words, using the
  layout queries NAK already exports for indirect dispatch.

That last point is the proposal's argument in miniature: the saving is not
cleverness in the frontend, it is state the API no longer has.

### Caveats

- **Shim, not hardware.** Upload memory is ordinary RAM here; on the card
  both APIs cost about twice as much and the ratio drops from about seven to
  about five (table above).
- **Submit figures are meaningless offline** (the ioctl is a no-op, and the
  shim's emulation runs in userspace), so they are reported only for
  hardware.
- NVK could cache QMDs too; it does not, and with descriptor-dependent cbuf
  bindings it would need to key the cache on more than the pipeline. The
  comparison is against the driver as it is.
- One shader, one shape. No statement yet about anything but compute dispatch.

## Draw cost, offline

Measured 2026-11-24, on the shim, release build. On hardware: "Twelfth
window" below, 65 - 73 ns against 103 - 106 ns at 100 draws per buffer.

`kgp_triangle bench` and `vk_baseline triangle bench` (`tests/tri_scene.h`):
N command buffers, each one render pass with a clear, the pipeline set
once and M draws of the triangle test's quad, every draw with its own pair
of root pointers. A Vulkan draw is `vkCmdPushConstants` (the two pointers)
plus `vkCmdDrawIndexed`, with the index buffer bound once. Timed like the
dispatch benchmark.

| | Record, ns per draw (500 x 1000) | User instructions per draw (marginal) |
|---|---|---|
| NVK through Vulkan | 58.9 - 59.4 | 660 |
| KoreGP | 18.9 - 19.0 | 218 |

Instructions are marginal as above: 2000 command buffers of 600 draws
against 2000 of 100. At 100 draws per command buffer the per-buffer work
(render pass, state, shaders) is a larger share and the times are 29 - 44
ns against 80 - 112 ns.

The ratio is about three, less than for dispatches, and for a different
reason. Per draw the two send the GPU almost the same thing: the draw
macro with its five parameters, and the root pointers into a hardware root
table (KoreGP one load of four words, NVK two loads of six). Nothing like
the QMD is rebuilt per draw on either side. What NVK executes and KoreGP
does not is bookkeeping: the push constants go into a shadow of the root
descriptor table that is flushed at the draw, and every draw asks whether
any dynamic state, descriptor set or vertex buffer is dirty. KoreGP's draw
has one flag to look at.

One change came out of the measurement. KoreGP rebound the index buffer on
every draw (five more words and a macro call); it now binds it when the
address changes or a draw reads further than the last bind allowed, as an
application calling `vkCmdBindIndexBuffer` once would. That took a draw
from about 30 ns to about 19.

The figures are of the tree as it stood at the end of 2026-11-24, with the
validation mode's branch in the draw path (three instructions; the first
version of the heap bounds checks cost four nanoseconds more by making
the root write variable-sized, found by this benchmark and undone). A
dispatch on the same build is 48 ns and 263 instructions, up from the 240
of the table above by what has been added to it since: the validation
branch and the two heap counts in the root record.

## Root convention v1 (experimental, removed)

The code was deleted on 2026-11-24 at the owner's decision, with its tests
and the suite steps that ran it; commit `23992ca` is the last tree that has
it. This section and the two measurements above are the record of what it
was and why it was not adopted.

Plan task N1.H, implemented behind `KGP_DEBUG=rootv1` so both conventions
could be measured in one hardware window. v0 gives the shader the root *pointer*
(in constant buffer 0) and lets it chase it; v1 additionally binds the
caller's root struct itself as constant buffer 2, and reads of root fields
at constant offsets become constant-buffer loads
(`kgp_nir_lower_root.c:kgp_nir_lower_root_v1`).

On NVIDIA a constant-buffer value can be an instruction operand, so the root
fields stop being loads at all. Static effect, same shaders, from NAK:

| Shader | Instructions, v0 | Instructions, v1 |
|---|---|---|
| saxpy | 25 | 16 |
| prefix_sum | 38 | 26 |
| scratch | 44 | 35 |

saxpy under v1 has no load of the root struct left: the bounds test is
`isetp.lt.u32 r0 c[0x2][0x14]`, the multiply-add takes `a` as
`c[0x2][0x10]`. This is the NVIDIA counterpart of the register-preload
optimization the proposal describes for AMD.

Costs, which are why it is not the default yet:

- the root pointer must be 64-byte aligned (the tests' root structs are
  padded to 64 bytes for this), and only the first 64 KiB of the struct and
  only `readonly` root blocks benefit;
- rewriting a root struct between dispatches now races the hardware's
  constant-buffer cache. The v1 QMD asks for the constant cache to be
  invalidated on every dispatch, which is the safe setting and may cost what
  the saved loads gained.

Whether it is a net win is a GPU-time question the shim cannot answer, and
the first hardware measurement did not answer it either: `tests/bench_gpu.c`
shows the two conventions equal on a shader that turned out to be limited by
memory (see "GPU time of the two root conventions" above). All four GPU
tests pass under v1 on hardware. v0 stays the default; settling v1 needs a
shader whose time is in arithmetic or in many small dispatches, not in
memory traffic. `bench_gpu loop` and its dispatch-count argument were added
for that and ran in the second window ("Root conventions, second
measurement"): v1 is a few percent faster on saxpy and markedly slower on
the loop shader. v0 stays the default.

## Fourth window (2026-11-22 16:25)

Logs in `~/koregp/logs/20261122-162557`. Every step passed. New in it:
`kgp_triangle depth` and its Vulkan twin (R7), correct over all 64×64
pixels: depth clear, depth test and write, additive blending, and the
depth-stencil and blend state objects changed between draws of one
pipeline.

That answers delta D4 for depth as it did for colour: a `D32_FLOAT` depth
target works at an ordinary `kgpMalloc` address (PTE kind 0), where NVK maps
depth through a range tagged with the format's own kind. The fallback
(`KGP_DEBUG=ptekind`) was not needed and did not run. Not covered: depth
formats with stencil, sampling a depth texture, multisampling.

Dispatch cost 95 – 110 ns against 442 – 788 ns; `strace` unchanged;
`bench_gpu` as in the third window (the first-run effect showed again, the
repeated run is at 0.60 ms under both conventions).

## Fifth window (2026-11-23 09:13)

Logs in `~/koregp/logs/20261123-091347`. Every step passed except
`stencil`. New in it:

- `kgp_triangle multi` (R9b): correct over all 64×64 pixels. KoreGP's own
  macro, the 32nd in the macro memory, read the draw count and three
  argument records from GPU memory and stepped both root pointers per draw.
  It had only run on Mesa's simulator before.
- `kgp_triangle stencil` (R7b) with the `D24_UNORM_S8_UINT` texture at an
  ordinary address: **failed**. The clear and both draws produced nothing
  (the image read back all zero), the kernel logged `Xid:13 Graphics
  Exception ... 3D-Z KIND Violation` on six GPCs and killed the channel.
- The same test with the texture mapped a second time through a range
  tagged with NIL's PTE kind for the format (`KGP_DEBUG=ptekind`, as it
  then was): correct over all pixels, stencil clear, write and test.
- The Vulkan twin of the stencil scene passed.

So delta D4 has a split answer on this card: colour and `D32_FLOAT` work at
kind-0 addresses, `D24_UNORM_S8_UINT` does not and needs its kind. The alias
mapping is now what `kgpCreateTexture` does for that format, the debug flag
is gone, and the format requires `KGP_MEMORY_GPU` memory starting on a page
of its allocation (refused otherwise). That change was made after the
window and has run offline only; the code path is the one that passed.

Dispatch cost 95 – 121 ns against 443 – 790 ns; `strace` unchanged. No
first-run outlier in `bench_gpu` this time (saxpy 16M in one dispatch 0.61 ms
first and last), so that effect does not show in every window.

## Sixth window (2026-11-23 09:50)

Logs in `~/koregp/logs/20261123-095013`. Every step passed, `stencil` now
as a plain step: `kgpCreateTexture` maps `D24_UNORM_S8_UINT` under its PTE
kind by itself. Dispatch cost 111 – 135 ns against 789 – 815 ns.

## Seventh window (2026-11-23 11:46)

Logs in `~/koregp/logs/20261123-114623`. Every step passed, two of them
for the first time on the GPU, and the kernel log has no GPU error from
the window:

- `kgp_spec`: specialization constants. The shader computes 10.5 with its
  defaults and -42.25 with three constants replaced.
- `present`: `kgp_present` on a headless swapchain. The frame KoreGP drew
  into the acquired swapchain texture reads back correct, so memory
  allocated by the hidden NVK device, exported as a dma-buf and imported
  into KoreGP's device is the memory KoreGP rendered to, and a render
  target laid out by NVK's NIL and mapped under its PTE kind works as a
  KoreGP colour target. Nothing was shown anywhere: the window had no
  modesetting, so `present_display` was skipped by the suite.

It was also the first run of the edited `to-nvidia.sh` (console unbind
before unloading nouveau): no framebuffer console was bound to nouveau, so
it did nothing, and the host came back as before.

Dispatch cost 83 – 136 ns against 434 – 790 ns. The Vulkan figure moved
more than in earlier windows (790 ns in the first run, 434 – 492 ns in the
other three); nothing in KoreGP explains a change in NVK's number, so read
it as noise of this run until a window repeats it.

## Eighth window (2026-11-23 15:00): display shakedown

`w0.sh display`: nouveau loaded with `modeset=1 config=NvGspRm=1`, nothing
of KoreGP run, output kept in the session only (no log directory). It
passed, and the host came back exactly as before:

- nouveau saw the monitor and lit it: `card0-HDMI-A-2: connected enabled
  3840x2160`, and the kernel console moved to its framebuffer (`fbcon:
  nouveaudrmfb (fb0) is primary device`).
- With the console bound there, nouveau's use count was 0: the console
  holds the DRM framebuffer helper, not the driver (the same reason
  `nvidia_drm` unloads with the console on it). So the unload this window
  was opened to test is not blocked by the console. `to-nvidia.sh` unbinds
  it first all the same.
- `snd_hda_intel` bound the card's HDMI audio to nouveau (`ops
  nv50_audio_component_bind_ops`), which ordinary windows never see; it
  did not hold the module either.
- nouveau unloaded, nvidia loaded, gdm restarted; the console is back on
  `nvidia-drmdrmfb`; no warning, oops or Xid in the kernel log. The card
  was on nouveau for about three seconds.

## Ninth window (2026-11-23 16:54): the suite in a display window

`w0.sh present`, logs `~/koregp/logs/20261123-165430`. Every step passed
except the one the window was opened for; the host came back as before
(about 23 seconds on nouveau, no warning, oops or Xid).

- `kgp_validate` ran on the GPU for the first time and passed.
- The whole suite is unchanged by modesetting being on: compute, copies,
  heaps, triangles, depth, stencil, multi-draw, headless present.
- `kgp_present display` did not reach the monitor. `kgpCreateSwapchain`
  was refused because NVK listed no display, although the connector was
  connected. NVK's WSI lists displays only if the DRM primary node opened
  and made it DRM master, and does not say which failed. Nothing else
  held the node (gdm was stopped, the journal shows no other client), so
  the suspect is the open: `to-nouveau.sh` chowns `card0` to the test
  user, and udev resets a node's owner on its next change event, which a
  connector hotplug is. Not proven; the node was not looked at in the
  window.
- Changed for the next attempt: `kgpCreateSwapchain` now repeats the open
  and names the reason (cannot open, another DRM master, or no display
  connected), and `hardware-suite.sh` logs the node's ownership as found
  and re-chowns it before the display test.
- Dispatch recording, release flavor: KoreGP 91-113 ns, NVK 443-813 ns.

## Tenth window (2026-11-23 18:39): the same, and the suspicion was wrong

`w0.sh present` again, logs `~/koregp/logs/20261123-183927`. The same
result: everything passed but `kgp_present display`, nothing on the
monitor, host restored (no warning, oops or Xid).

- The ownership theory of the ninth window is disproved. `card0` was the
  test user's when the display test started, and the repeated open in
  `kgpCreateSwapchain` succeeded and became DRM master. Its message, "no
  display is connected", was wrong too: the connector was connected.
- The cause, read from the source and matching every observation: Mesa's
  display WSI drives the screen through atomic modesetting only
  (`wsi_common_display.c`: `drmModeAtomicCommit`, no `drmModeSetCrtc`), and
  it drops a connector whose atomic properties (`CRTC_ID`) it cannot find.
  nouveau does not expose the atomic ioctl to user space unless loaded
  with `atomic=1` ("Expose atomic ioctl (default: disabled)"), so the
  properties were hidden, the connector was dropped, and no display was
  listed. Not yet confirmed by a run.
- Changed: `w0.sh present` loads nouveau with `modeset=1 atomic=1`;
  `kgpCreateSwapchain` tests for atomic modesetting and says so; the
  ownership repair is removed from `hardware-suite.sh`.
- `atomic=1` is new for the host. nouveau uses the atomic helpers inside
  the kernel either way (the console did in the eighth to tenth windows);
  the parameter only lets user space issue atomic commits, which is what
  Mesa will do. The unload path is the one already exercised.
- Dispatch recording, release flavor: KoreGP 61-132 ns, NVK 442-792 ns.

## Eleventh window (2026-11-23 19:17): presentation on the display

`w0.sh present` with nouveau loaded `modeset=1 atomic=1`, logs
`~/koregp/logs/20261123-191726`. Every step passed, `kgp_present display`
included, which confirms the cause found after the tenth window.

- `kgp_present display` created a swapchain on the connected display at
  its own mode, rendered and presented 120 frames through NVK's
  `VK_KHR_display` path, and the first frame, read back from the
  swapchain texture, has the scene in it. It ran 4.4 seconds. This is the
  first time a KoreGP program held the display as DRM master.
- The monitor showed it: the owner saw the two red rectangles appear.
  The test itself cannot check that.
- The restore went through with a display client having held the card
  node: nouveau unloaded, nvidia loaded, gdm active, console on
  `nvidia-drmdrmfb`, "host state restored"; no warning, oops or Xid in
  the kernel log.
- Dispatch recording, release flavor: KoreGP 108-133 ns, NVK 790-794 ns.

## Twelfth window (2026-11-24 18:06): everything added since the eleventh

Commit `26f0c8d`. Every step passed, and the host was restored cleanly.
First time on the GPU, all correct:

- `kgp_triangle state` (depth bias, blend constants and constant factors,
  colour write masks), `mrt` (two colour targets), `srgb`, `rgba16f`,
  `rgba32f`, `r32f`, each with its Vulkan twin;
- `kgp_triangle multigpu`: a multi-draw whose arguments, count and per-draw
  roots a compute shader wrote earlier in the same command buffer;
- `kgp_copy mip`, `layer` and `3d`, which is the first check of the mip and
  layer start offsets (the offline comparison masks addresses);
- the shader-side heap bounds check in `kgp_validate`: the fetch inside the
  declared count returns the texel, the one past it returns zero;
- the suite without root convention v1.

Draw cost, release build, 2000 command buffers of 100 draws, three runs
alternating the two APIs:

| | Record, ns per draw | Submit, us per command buffer |
|---|---|---|
| KoreGP | 64.5 - 73.3 | 7.8 - 8.8 |
| NVK through Vulkan | 103.3 - 105.7 | 8.4 - 8.6 |

About 1.6 times less CPU per draw. Offline the same shape gives 29 - 44 ns
against 80 - 112 ns: on the card both sides pay about 35 to 40 ns more per
draw, which is writing the pushbuf into write-combined memory instead of
RAM, so the difference between the two is about what it is offline and the
ratio is smaller. At 100 draws per command buffer the figure also includes
the per-buffer work (render pass, state, shaders). The 500 x 1000 shape of
the offline table was not run in the window.

Dispatch cost, same method as before: KoreGP 90.4, 95.2, 108.9 and 140.0 ns
against NVK's 444.4, 560.9, 811.0 and 838.2 ns, between 4.7 and 7.4 times.

`strace`: 12 `VM_BIND` with 8 command buffers and 12 with 800 (27 and 819
`EXEC`). The count was 8 in the first window; the device makes more
allocations at start-up now. It still does not depend on the number of
submits, which is the claim.

`bench_gpu`: the first run after the swap is again the slow one (saxpy
over 16M elements 0.790 ms first, 0.599 ms when repeated last).

## Memory safety

The `asan` flavor (AddressSanitizer + UBSan + leak check) runs every test
program on the shim, in both root conventions. Nothing is reported in KoreGP
or by AddressSanitizer or the leak check. UBSan prints one non-fatal report
per run from upstream Mesa: a signed left shift in
`winsys/nouveau_context.c:233` (`(0xbeef + channel) << 16`), on the
video-decode probe path. `meson test` cannot drive that flavor (the sanitizer
runtime must be preloaded ahead of the shim); `kgp_offline` in `env.sh` does
it for a single command.

## Reproducing

```
scripts/push.sh
ssh argon-isotopia ~/koregp/KoreGP/scripts/host/build-mesa.sh
ssh argon-isotopia 'cd ~/koregp/build && meson test --suite koregp-offline'

# benchmark (release flavor)
ssh argon-isotopia 'KGP_FLAVOR=release ~/koregp/KoreGP/scripts/host/build-mesa.sh'
ssh argon-isotopia 'export KGP_FLAVOR=release KGP_TEST_OFFLINE=1
   . ~/koregp/KoreGP/scripts/host/env.sh
   kgp_offline $KGP_PREFIX/bin/bench_dispatch kgp 5000 100
   kgp_offline $KGP_PREFIX/bin/bench_dispatch vk  5000 100'
```

`perf` on this host needs root (`perf_event_paranoid` is 4); the instruction
counts were taken with `sudo env … perf stat -e instructions:u`.

On hardware everything above runs from one command, which opens a window,
runs the suite and restores the proprietary driver: `scripts/host/w0.sh
swap`. It needs the owner's go-ahead each time (`test-host.md`).
