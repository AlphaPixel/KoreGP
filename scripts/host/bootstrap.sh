#!/bin/bash
# Idempotent user-local toolchain + source setup on the test host.
# Touches only $KGP_ROOT and ~/.cargo. System packages are a separate, gated
# step (docs/test-host.md, "System packages").
set -euo pipefail
. "$(dirname "$0")/env.sh"

mkdir -p "$KGP_ROOT"

if [ ! -x "$KGP_VENV/bin/meson" ]; then
   python3 -m venv "$KGP_VENV"
   "$KGP_VENV/bin/pip" install --quiet meson mako pyyaml packaging
fi

# Mesa rejects bindgen 0.72.0 and 0.73.0/0.73.1 and needs >= 0.71.1.
command -v bindgen  >/dev/null || cargo install --locked --quiet bindgen-cli
command -v cbindgen >/dev/null || cargo install --locked --quiet cbindgen

if [ ! -d "$MESA_SRC/.git" ]; then
   git clone --quiet --depth 1 --branch "$MESA_TAG" \
      https://gitlab.freedesktop.org/mesa/mesa.git "$MESA_SRC"
fi

echo "meson    $(meson --version)"
echo "bindgen  $(bindgen --version)"
echo "cbindgen $(cbindgen --version)"
echo "rustc    $(rustc --version)"
echo "mesa     $(git -C "$MESA_SRC" describe --tags --always)"
