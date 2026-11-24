#!/bin/sh
# The kernel-interface fact the pointer-only design hangs on (proposal 5.1):
# a submit names no buffers, and makes none resident. On nouveau that means
# DRM_NOUVEAU_EXEC carries only pushbuf address ranges and no VM_BIND happens
# per submit. Checked here from userspace: the number of VA binds must not
# depend on how many command buffers are submitted.
#
# usage: no_bind_per_submit.sh <bench_dispatch exe>
set -eu

binds() {
   KGP_DEBUG=vm "$1" kgp "$2" 10 2>&1 >/dev/null | grep -c 'bind vma' || true
}

few=$(binds "$1" 8)
many=$(binds "$1" 800)

echo "VA binds: $few with 8 command buffers, $many with 800"
[ "$few" -gt 0 ] && [ "$few" = "$many" ]
