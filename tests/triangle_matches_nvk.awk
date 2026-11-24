# Used by triangle_matches_nvk.sh. Reads a pushbuf dump and prints the 3D
# subchannel's methods, one per line with their decoded fields, from the
# start of the render pass (the per-command-buffer prologue before it
# included) to the draw macro and its parameters.
#
# Not compared: packet headers (see init3d_matches_nvk.sh for why).
# Masked: every address. Those are the colour targets', the depth
# target's, the two root
# pointers and the eight sample-location bytes NVK leaves partly
# uninitialized (all LOAD_ROOT_TABLE data), the index buffer's (the bind
# macro's first two parameters) and the two program addresses.
# Dropped: the root-table write of NVK's printf buffer address, which only
# assert-enabled NVK builds emit (root table 2).

function flush_line() {
   if (line != "") lines[++n] = line
   line = ""
}

/ HDR / { on = (index($0, "subch 0 ") > 0); next }
!on || !NF { next }
/mthd/ {
   flush_line()
   sub(/^[ \t]+mthd [0-9a-f]+ /, "")
   line = $0
   next
}
{
   v = $0
   gsub(/^[ \t]+/, "", v)
   line = line " | " v
}

END {
   flush_line()

   first = 0
   last = 0
   for (i = 1; i <= n; i++) {
      if (!first && lines[i] ~ /SET_SURFACE_CLIP_HORIZONTAL/)
         first = i - 4      # the 3 prologue methods and the view mask
      # The draw macro and its parameters: direct (slot 12, six of them) or,
      # with -v indirect=1, indexed-indirect (slot 14, five).
      if (!indirect && lines[i] ~ /CALL_MME_MACRO\(12\)/) {
         last = i + 5
         break
      }
      if (indirect && lines[i] ~ /CALL_MME_MACRO\(14\)/) {
         last = i + 4
         break
      }
   }
   if (first < 1 || last == 0) {
      print "no render pass followed by a draw in this dump" > "/dev/stderr"
      exit 1
   }

   skip = 0
   ib = 0
   for (i = first; i <= last; i++) {
      l = lines[i]
      if (l ~ /SET_ROOT_TABLE_SELECTOR \| \.ROOT_TABLE = \(0x2\)/) {
         skip = 2
         continue
      }
      if (skip > 0 && l ~ /LOAD_ROOT_TABLE/) {
         skip--
         continue
      }
      skip = 0

      # ZCULL, the coarse depth test: NVK sets up a region when the render
      # pass clears a depth target (and clears the region, with a second
      # copy of the depth clear value in front); KoreGP leaves ZCULL off
      # (kgpBeginRenderPass says why). Dropped on both sides, the region
      # select included, which is the one method where the two differ.
      if (l ~ /ZCULL_(REGION|STORAGE|SUBREGION|CRITERION|DIR_FORMAT)|ASSIGN_ZCULL/)
         continue
      if (l ~ /SET_Z_CLEAR_VALUE/ && lines[i + 1] ~ /CLEAR_ZCULL_REGION/)
         continue

      # Index buffer bind: address, and for an indirect draw also the size
      # (NVK knows its buffer's; KoreGP sets the limit as high as it goes).
      # Indirect draw: the address of the arguments.
      if (l ~ /CALL_MME_MACRO\(4\)/)
         ib = indirect ? 3 : 2
      if (l ~ /CALL_MME_MACRO\(14\)/)
         ib = 2
      if (ib > 0 || l ~ /LOAD_ROOT_TABLE|SET_COLOR_TARGET_[AB]\([0-7]\)|SET_ZT_[AB] |PROGRAM_ADDRESS_[AB]/) {
         gsub(/\(0x[0-9a-f]+\)/, "(masked)", l)
         if (ib > 0) ib--
      }
      print l
   }
}
