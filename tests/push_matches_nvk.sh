#!/bin/sh
# Differential check of command emission: run the same test through KoreGP
# and through Vulkan on NVK, dump what each pushes, and require the method
# streams of one subchannel to be identical: the compute one, or for the
# "copy" test the copy engine's.
#
# Runs on the DRM shim (the environment comes from tests/meson.build), so
# it compares what would be sent to the GPU, not what the GPU does with it.
#
# Masked, because they legitimately differ:
#   - QMD and shader-local-memory addresses (their sizes are compared);
#   - where an indirect dispatch's GPU-side copies land (OFFSET_OUT), and
#     the words the dumper prints as that copy's inline data: the data comes
#     from the caller's memory, so what it prints there is the next methods;
#   - texture / sampler pool location and size: NVK points them at its
#     descriptor tables, the KoreGP tests compared here bind no heap;
#   - on the copy subchannel, source and destination addresses.
# Dropped from both: NV906F_SET_REFERENCE. It is a channel-level method that
# either side may send on any subchannel; NVK uses the 3D one, which a
# KoreGP channel does not have.
#
# The copy comparison takes NVK's side from its transfer-only queue, the one
# place it drives the copy engine; KoreGP's side also dispatches a shader,
# on the compute subchannel, which is not part of that comparison.
#
# usage: push_matches_nvk.sh <test> <kgp test exe> <vk_baseline exe> [arg]
#        [arg] is passed to both programs (e.g. "linear").
set -eu

case "$1" in
copy) subch=4; what=copy;    marker=LAUNCH_DMA ;;
*)    subch=1; what=compute; marker=SEND_PCAS_A ;;
esac

stream() {
   awk -v subch="subch $subch " '
      / HDR / {
         on = (index($0, subch) > 0)
         sub(/^\[0x[0-9a-f]+\] HDR /, "HDR ")
         held = $0
         next
      }
      /mthd/ {
         if (held != "") {
            if (/NV906F_SET_REFERENCE/) on = 0
            if (on) print held
            held = ""
         }
         addr = /_OFFSET_OUT|_LOAD_INLINE_DATA|_OFFSET_IN/
      }
      {
         if (!on || !NF) next
         if (addr && !/mthd/) sub(/\(0x[0-9a-f]+\)/, "(masked)")
         print
      }
   ' | sed -E 's/[.](QMD_ADDRESS_SHIFTED8|OFFSET_UPPER|OFFSET_LOWER|MAXIMUM_INDEX|ADDRESS_UPPER|ADDRESS_LOWER) = \(0x[0-9a-f]+\)/.\1 = (masked)/'
}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

NVK_DEBUG=push_dump "$3" "$1" ${4:+"$4"} 2>&1 >/dev/null | stream > "$tmp/nvk"
KGP_DEBUG=push "$2" ${4:+"$4"} 2>&1 >/dev/null | stream > "$tmp/koregp"

for f in nvk koregp; do
   if ! grep -q "$marker" "$tmp/$f"; then
      echo "no $marker in the $f pushbuf dump" >&2
      exit 1
   fi
done

diff -u "$tmp/nvk" "$tmp/koregp"
echo "$1${4:+ $4}: $(grep -c mthd "$tmp/koregp") $what methods, identical to NVK"
