#!/bin/bash
# Configure (first run) and build Mesa NVK + KoreGP into the private prefix.
# Headless: no X11/Wayland/GL, so the build cannot collide with the desktop
# stack even by accident.
set -euo pipefail
. "$(dirname "$0")/env.sh"

case "$KGP_PREFIX" in
   "$HOME"/*) ;;
   *) echo "refusing: prefix $KGP_PREFIX is outside \$HOME" >&2; exit 1 ;;
esac

# Upstream Mesa is changed only through mesa-patches/. A patch that is
# already in the tree is skipped; one that neither applies nor is applied
# stops the build.
for p in "$KGP_REPO"/mesa-patches/*.patch; do
   [ -e "$p" ] || continue
   if git -C "$MESA_SRC" apply --reverse --check "$p" 2>/dev/null; then
      continue
   fi
   git -C "$MESA_SRC" apply "$p"
   echo "applied $(basename "$p")"
done

# Overlay the frontend: the repo root becomes mesa/src/koregp (its
# meson.build is the subdir file the build-hook patch pulls in).
ln -sfn "$KGP_REPO" "$MESA_SRC/src/koregp"

KGP_MESON_OPTS=(
   --prefix="$KGP_PREFIX"
   $KGP_FLAVOR_OPTS
   -Dvulkan-drivers=nouveau
   -Dgallium-drivers=
   -Dplatforms=
   -Dglx=disabled -Degl=disabled -Dgbm=disabled
   -Dopengl=false -Dgles1=disabled -Dgles2=disabled
   -Dllvm=enabled -Dshared-llvm=enabled
   -Dvideo-codecs=
   -Dtools=drm-shim
   -Dvalgrind=disabled -Dlibunwind=disabled
   -Dkoregp=true
)

if [ ! -f "$MESA_BUILD/build.ninja" ]; then
   meson setup "$MESA_BUILD" "$MESA_SRC" "${KGP_MESON_OPTS[@]}"
elif ! meson introspect "$MESA_BUILD" --buildoptions | grep -q '"name": "koregp", "value": true'; then
   meson configure "$MESA_BUILD" -Dkoregp=true
fi

ninja -C "$MESA_BUILD"
ninja -C "$MESA_BUILD" install >/dev/null
echo "installed to $KGP_PREFIX"
