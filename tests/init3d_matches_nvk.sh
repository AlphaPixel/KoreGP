#!/bin/sh
# The 3D engine's channel init (plan N3): everything KoreGP sends on the 3D
# subchannel when a device is created must be exactly how NVK's 3D stream
# begins. That covers the object bind, all 31 macros and their slots, the
# default state, the root tables and constant buffer 0.
#
# KoreGP's side is kgp_info (create a device, nothing else); NVK's is any
# Vulkan twin, whose 3D stream continues after the init with per-submit and
# per-command-buffer state, so only its head is compared.
#
# Compared: the decoded methods and their values, in order. Not the packet
# headers: nv_push picks the one-word immediate form when the compiler can
# prove the value is a small constant, which depends on inlining, so the
# same method legitimately comes out in either form.
# Masked: the addresses of constant buffer 0 and of the zero page (macro
# scratch registers 26 to 29 at this Mesa pin, and the vertex stream
# substitute); and the location and size of the texture and sampler pools,
# which follow the init on both sides (NVK's own tables; NULL for KoreGP
# until a heap is set).
# Cut off: KoreGP's own macro (kgp_mme.c), which is loaded after the pools
# for exactly this reason; NVK has nothing in that slot.
#
# usage: init3d_matches_nvk.sh <kgp_info exe> <vk_baseline exe>
set -eu

stream() {
   awk '
      / HDR / {
         on = (index($0, "subch 0 ") > 0)
         next
      }
      /mthd/ {
         addr = /SCRATCH\((26|27|28|29)\)|_SET_VERTEX_STREAM_SUBSTITUTE_|_SET_TEX_(HEADER|SAMPLER)_POOL_/
      }
      {
         if (!on || !NF) next
         if (addr && !/mthd/) sub(/\(0x[0-9a-f]+\)/, "(masked)")
         print
      }
   '
}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

NVK_DEBUG=push_dump "$2" saxpy 2>&1 >/dev/null | stream > "$tmp/nvk"
KGP_DEBUG=push "$1" 2>&1 >/dev/null | stream |
   awk '/_SET_TEX_HEADER_POOL_A/ { pools = 1 }
        pools && /LOAD_MME_START_ADDRESS_RAM_POINTER/ { exit }
        { print }' > "$tmp/koregp"

lines=$(wc -l < "$tmp/koregp")
if [ "$lines" -lt 1000 ]; then
   echo "only $lines lines of 3D init in the koregp pushbuf dump" >&2
   exit 1
fi

head -n "$lines" "$tmp/nvk" | diff -u - "$tmp/koregp"
echo "3D init: $(grep -c mthd "$tmp/koregp") methods, identical to the head of NVK's"
