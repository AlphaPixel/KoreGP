#!/bin/sh
# KoreGP's multi-draw macro on Mesa's simulator of the macro processor: the
# methods it emits for three draws must be the reviewed ones in
# mme_multi_sim.expected. That file is the simulator's own output; after an
# intended change to the macro, read the new output and replace it.
#
# usage: mme_multi_sim.sh <kgp_mme_sim exe> <expected>
set -eu
"$1" | diff -u "$2" -
echo "multi-draw macro: output identical to $(basename "$2")"
