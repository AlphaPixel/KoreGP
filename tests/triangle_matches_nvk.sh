#!/bin/sh
# Differential check of raster command emission (plan N3, task R6): from
# the start of the render pass to the draw, KoreGP's 3D methods for
# tests/kgp_triangle.c must be NVK's for its Vulkan twin, method for method
# and value for value. What is masked and dropped, and why, is in
# triangle_matches_nvk.awk.
#
# Runs on the DRM shim: it compares what would be sent, not what the GPU
# does with it. Macro calls are compared as calls; what a macro emits
# depends on state on the GPU.
#
# usage: triangle_matches_nvk.sh <awk script> <kgp_triangle exe> <vk_baseline exe> [indirect]
set -eu

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

ind=0
[ "${4:-}" = indirect ] && ind=1

NVK_DEBUG=push_dump "$3" triangle ${4:+"$4"} 2>&1 >/dev/null |
   awk -v indirect=$ind -f "$1" > "$tmp/nvk"
KGP_DEBUG=push "$2" ${4:+"$4"} 2>&1 >/dev/null |
   awk -v indirect=$ind -f "$1" > "$tmp/koregp"

diff -u "$tmp/nvk" "$tmp/koregp"
echo "triangle${4:+ $4}: $(wc -l < "$tmp/koregp") 3D methods from render pass to draw, identical to NVK"
