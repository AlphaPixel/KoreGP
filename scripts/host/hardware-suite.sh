#!/bin/bash
# Everything that needs the real GPU, in one go, inside an open window:
#
#    with-nouveau.sh -w 20 ~/koregp/KoreGP/scripts/host/hardware-suite.sh
#
# Never opens or closes a window itself. Keeps going after a failure so one
# window yields as much as possible, and logs every step to
# ~/koregp/logs/<time>/. Exit status is 0 only if every step passed.
#
# Order is deliberate: what is known to work first (NVK through Vulkan),
# then KoreGP from the least to the most demanding shader, then measurements.
set -uo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
. "$HERE/env.sh"

[ "$(kgp_gpu_driver)" = nouveau ] || {
   echo "GPU is on '$(kgp_gpu_driver)', not nouveau; run this through with-nouveau.sh" >&2
   exit 1
}

# Without this every step fails the same uninformative way (the first
# window did): the userspace driver cannot open the node and reports no GPU.
node=$(ls /dev/dri/renderD* 2>/dev/null | head -1)
if [ -z "$node" ] || [ ! -r "$node" ] || [ ! -w "$node" ]; then
   echo "cannot open the render node (${node:-none}): $(ls -l /dev/dri 2>&1 | tail -n +2 | tr '\n' ';')" >&2
   echo "to-nouveau.sh hands it to the invoking user; was the window opened some other way?" >&2
   exit 1
fi

LOG="$KGP_ROOT/logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG"
echo "logging to $LOG"
failed=""

step() {
   local name="$1"; shift
   echo "== $name"
   "$@" > "$LOG/$name.log" 2>&1
   local rc=$?
   tail -n "${STEP_TAIL:-6}" "$LOG/$name.log" | sed 's/^/   /'
   [ $rc = 0 ] || { echo "   FAILED (exit $rc)"; failed="$failed $name"; }
}

mtest() { meson test -C "$MESA_BUILD" --no-rebuild --num-processes 1 --print-errorlogs "$@"; }

# What nouveau made of the card: answers the W0 unknowns (GSP, VRAM, BAR).
step dmesg sh -c 'sudo dmesg | grep -i -E "nouveau|gsp" | tail -60'
STEP_TAIL=12 step vulkaninfo kgp_run vulkaninfo --summary
step kgp_info kgp_run "$KGP_PREFIX/bin/kgp_info"

# Baseline: the same shaders through Vulkan on NVK.
step vk_tests mtest --suite koregp-hardware \
   vk_saxpy vk_prefix_sum vk_scratch vk_indirect

# KoreGP. One at a time so a killed channel is attributable.
for t in kgp_reject kgp_saxpy kgp_prefix_sum kgp_scratch kgp_indirect kgp_spec kgp_validate; do
   step "$t" mtest --suite koregp-hardware "$t"
done

# N2, first contact: textures through the heap. Not part of N1 either. The
# pair is the probe for delta D4: "linear" is the layout that needs nothing
# from its memory; the tiled one sits in ordinary memory, which NVK never
# does. Linear passing and tiled failing means textures need the PTE-kind
# mapping (docs/plan.md, N2).
step heap_linear mtest --suite koregp-hardware kgp_heap_linear
step heap_tiled mtest --suite koregp-hardware kgp_heap

# N2, GPU-side copies on the copy engine (T7). The Vulkan twin first; it
# records on NVK's transfer queue, the one place NVK uses that engine.
step vk_copy mtest --suite koregp-hardware vk_copy vk_copy_linear \
   vk_copy_mip vk_copy_layer vk_copy_3d
step copy_linear mtest --suite koregp-hardware kgp_copy_linear
step copy_tiled mtest --suite koregp-hardware kgp_copy
# One mip level, one array layer, a 3D texture (copy_pattern.h).
step copy_sub mtest --suite koregp-hardware kgp_copy_mip kgp_copy_layer \
   kgp_copy_3d

# N3: graphics pipelines, then the first triangles, Vulkan twin first.
step gfx_pipeline mtest --suite koregp-hardware kgp_gfx_pipeline
step vk_triangle mtest --suite koregp-hardware \
   vk_triangle vk_triangle_indirect vk_triangle_depth vk_triangle_stencil \
   vk_triangle_state vk_triangle_mrt vk_triangle_srgb vk_triangle_rgba16f \
   vk_triangle_rgba32f vk_triangle_r32f
step triangle mtest --suite koregp-hardware kgp_triangle
step triangle_indirect mtest --suite koregp-hardware kgp_triangle_indirect
step textured mtest --suite koregp-hardware kgp_textured
# R9b: one GPU-driven multi-draw with per-draw root data, through KoreGP's
# own macro. No Vulkan twin.
step multi mtest --suite koregp-hardware kgp_triangle_multi
# The same with the draw list written by a compute shader (the proposal's
# M2 exit test: GPU-driven multi-draw with per-draw root data).
step multigpu mtest --suite koregp-hardware kgp_triangle_multigpu

# R10: presentation. A headless swapchain in any window; on the monitor
# only when nouveau drives one (a window opened by w0.sh present).
step present mtest --suite koregp-hardware kgp_present
if grep -qx connected /sys/class/drm/card*-*/status 2>/dev/null; then
   step present_display mtest --suite koregp-hardware kgp_present_display
fi

# R7: a depth target, depth-stencil and blend state; R7b: the same with a
# depth-stencil format. D32_FLOAT sits in ordinary memory (PTE kind 0);
# D24S8 is mapped with its own kind, having failed without (2026-11-23).
step depth mtest --suite koregp-hardware kgp_triangle_depth
step stencil mtest --suite koregp-hardware kgp_triangle_stencil
# Depth bias, colour write mask and blend constant (tri_scene.h, "state").
step state mtest --suite koregp-hardware kgp_triangle_state
# Two colour targets, the second through a per-target write mask.
step mrt mtest --suite koregp-hardware kgp_triangle_mrt
# The plain scene into render targets of other formats.
step formats mtest --suite koregp-hardware kgp_triangle_srgb \
   kgp_triangle_rgba16f kgp_triangle_rgba32f kgp_triangle_r32f

# If anything KoreGP failed, a rerun of saxpy that narrows it down while the
# window is open: waiting after every submit, with the pushbuf dumped at the
# failing one.
case "$failed" in *kgp_*|*copy_*|*heap_*)
   kgp_dbg() { KGP_DEBUG="$1" kgp_run "$KGP_PREFIX/bin/kgp_saxpy"; }
   STEP_TAIL=40 step diag_sync kgp_dbg sync
   ;;
esac
# The same for raster, on the triangle; its own output shows the image.
case "$failed" in *triangle*|*textured*|*depth*|*stencil*|*state*|*mrt*|*formats*|*multi*|*present*)
   tri_dbg() { KGP_DEBUG=sync kgp_run "$KGP_PREFIX/bin/kgp_triangle"; }
   STEP_TAIL=60 step diag_triangle tri_dbg
   ;;
esac
# What the kernel said during the run: a killed channel shows here.
[ -z "$failed" ] ||
   STEP_TAIL=30 step dmesg_after sh -c 'sudo dmesg | grep -i -E "nouveau|gsp|fault" | tail -40'

# N1 exit criterion, kernel's view: submits are DRM_NOUVEAU_EXEC (nr 0x52)
# only; VM_BIND (nr 0x51) count does not grow with the number of submits.
trace_submits() {   # <command buffers>: strace one benchmark run
   strace -f -e trace=ioctl -o "$LOG/strace-$1.txt" \
      env LD_LIBRARY_PATH="$KGP_PREFIX/lib/x86_64-linux-gnu" \
      "$KGP_PREFIX/bin/bench_dispatch" kgp "$1" 10
}
ioctl_count() {     # <nr> <name> <command buffers>
   grep -c -E "0x64, 0x$1,|DRM_IOCTL_NOUVEAU_$2" "$LOG/strace-$3.txt"
}
check_submit_ioctls() {
   trace_submits 8 && trace_submits 800 || return 1
   local b8 e8 b800 e800
   b8=$(ioctl_count 51 VM_BIND 8);     e8=$(ioctl_count 52 EXEC 8)
   b800=$(ioctl_count 51 VM_BIND 800); e800=$(ioctl_count 52 EXEC 800)
   echo "8 command buffers:   VM_BIND $b8, EXEC $e8"
   echo "800 command buffers: VM_BIND $b800, EXEC $e800"
   [ "$b8" = "$b800" ] && [ "$e800" -gt "$e8" ]
}
step submit_ioctls check_submit_ioctls

# N1.G: dispatch cost. Only the release build's numbers mean anything.
REL="$KGP_ROOT/prefix-release"
if [ -x "$REL/bin/bench_dispatch" ]; then
   bench() {
      VK_DRIVER_FILES="$REL/share/vulkan/icd.d/nouveau_icd.x86_64.json" \
      LD_LIBRARY_PATH="$REL/lib/x86_64-linux-gnu" \
      "$REL/bin/bench_dispatch" "$@"
   }
   run_bench() {
      for _ in 1 2 3; do
         bench kgp 2000 100 || return 1
         bench vk 2000 100 2>/dev/null || return 1
      done
      bench kgp 200 1000 && bench vk 200 1000 2>/dev/null
   }
   STEP_TAIL=8 step bench run_bench

   # The same for draws (tests/tri_scene.h, "bench"): one render pass per
   # command buffer, every draw with its own pair of root pointers.
   rel() {
      VK_DRIVER_FILES="$REL/share/vulkan/icd.d/nouveau_icd.x86_64.json"       LD_LIBRARY_PATH="$REL/lib/x86_64-linux-gnu"       "$@"
   }
   run_bench_draw() {
      for _ in 1 2 3; do
         rel "$REL/bin/kgp_triangle" bench 2000 100 || return 1
         rel "$REL/bin/vk_baseline" triangle bench 2000 100 2>/dev/null || return 1
      done
   }
   STEP_TAIL=6 step bench_draw run_bench_draw

   # GPU time (tests/bench_gpu.c):
   #  - saxpy at full and quarter size: memory-bound in the first window,
   #    and if so the time scales with the elements;
   #  - the same work cut into 1024 dispatches: per-dispatch cost;
   #  - "loop": arithmetic on root fields.
   run_gpu_bench() {
      gpu() { LD_LIBRARY_PATH="$REL/lib/x86_64-linux-gnu" "$REL/bin/bench_gpu" "$@"; }
      local args
      # The first workload runs again last: the first GPU benchmark after a
      # driver swap is slow (third window).
      for args in "saxpy" "saxpy 4194304" "saxpy 16777216 1024" \
                  "loop" "loop 1048576 1024" "saxpy"; do
         # shellcheck disable=SC2086
         gpu $args || return 1
      done
   }
   STEP_TAIL=10 step bench_gpu run_gpu_bench
else
   echo "== bench: skipped (no release build; KGP_FLAVOR=release build-mesa.sh)"
fi

echo
if [ -n "$failed" ]; then
   echo "FAILED:$failed    (logs in $LOG)"
   exit 1
fi
echo "all hardware steps passed (logs in $LOG)"
