# Sourced by the other host scripts. Everything lives under $KGP_ROOT in the
# user's home; nothing here may point at a system path.
# HOME is unset when to-nvidia.sh runs as the dead-man (a root systemd unit);
# to-nouveau.sh passes KGP_ROOT to it explicitly.
KGP_ROOT="${KGP_ROOT:-${HOME:-/nonexistent}/koregp}"
KGP_REPO="$KGP_ROOT/KoreGP"          # copy of this repo (scripts/push.sh)
MESA_TAG="mesa-26.3.1"               # the pin; bump deliberately, see docs/plan.md
MESA_SRC="$KGP_ROOT/mesa"
# Build flavors, each with its own build dir and private prefix:
#   debug    debugoptimized with asserts; what the tests run on
#   release  asserts off; only for measuring (bench_dispatch)
#   asan     debug + AddressSanitizer/UBSan; offline suite only
KGP_FLAVOR="${KGP_FLAVOR:-debug}"
case "$KGP_FLAVOR" in
   debug)   MESA_BUILD="$KGP_ROOT/build";         KGP_PREFIX="$KGP_ROOT/prefix"
            KGP_FLAVOR_OPTS="-Dbuildtype=debugoptimized" ;;
   release) MESA_BUILD="$KGP_ROOT/build-release"; KGP_PREFIX="$KGP_ROOT/prefix-release"
            KGP_FLAVOR_OPTS="-Dbuildtype=release -Db_ndebug=true" ;;
   asan)    MESA_BUILD="$KGP_ROOT/build-asan";    KGP_PREFIX="$KGP_ROOT/prefix-asan"
            KGP_FLAVOR_OPTS="-Dbuildtype=debugoptimized -Db_sanitize=address,undefined -Db_lundef=false" ;;
   *) echo "KGP_FLAVOR must be debug, release or asan" >&2; return 1 2>/dev/null || exit 1 ;;
esac
KGP_VENV="$KGP_ROOT/venv"
KGP_GPU="0000:09:00.0"               # TU104 RTX 2070 SUPER

export PATH="$KGP_VENV/bin:${HOME:+$HOME/.cargo/bin:}$PATH"

# Point a single process at the private Mesa build. Never exported globally,
# never written to /etc or /usr/share/vulkan.
kgp_run() {
   VK_DRIVER_FILES="$KGP_PREFIX/share/vulkan/icd.d/nouveau_icd.x86_64.json" \
   LD_LIBRARY_PATH="$KGP_PREFIX/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
   "$@"
}

# Size of the GPU's BAR1 (the CPU window onto VRAM) as log2 of MiB, which is
# the unit /sys/.../resource1_resize takes: 8 = 256 MiB, 13 = 8 GiB.
# "none" if the BAR is not assigned.
kgp_bar1_bits() {
   local start end _ mib bits=0
   read -r start end _ < <(sed -n 2p "/sys/bus/pci/devices/$KGP_GPU/resource")
   [ $((end)) -gt 0 ] || { echo none; return; }
   mib=$(( (end - start + 1) >> 20 ))
   while [ $((1 << bits)) -lt "$mib" ]; do bits=$((bits + 1)); done
   echo "$bits"
}

kgp_gpu_driver() {
   local d
   d=$(readlink "/sys/bus/pci/devices/$KGP_GPU/driver" 2>/dev/null) || { echo none; return; }
   basename "$d"
}

# Run a command on Mesa's nouveau DRM shim: the userspace driver sees a
# TU104, nothing reaches a kernel driver, and nvidia stays bound.
kgp_offline() {
   # The sanitizer runtime has to come before any other preload. (For the
   # same reason `meson test` cannot run the asan flavor: run the installed
   # programs through this function instead.)
   local first=""
   [ "$KGP_FLAVOR" = asan ] && first="$(cc -print-file-name=libasan.so):"
   LD_PRELOAD="$first$KGP_PREFIX/lib/x86_64-linux-gnu/libnouveau_noop_drm_shim.so" \
   NOUVEAU_CHIPSET=164 \
   kgp_run "$@"
}
