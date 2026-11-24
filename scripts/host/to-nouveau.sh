#!/bin/bash
# Runtime swap: proprietary nvidia -> nouveau, for one test window.
#
# No reboot and no persistent change: nothing is written under /etc, /boot or
# /lib. The nouveau blacklist stays in place; we bypass it for this one
# modprobe with an empty config dir. A reboot at any point returns the host
# to stock nvidia.
#
# Usage: to-nouveau.sh [window-minutes]   (default 120)
# A dead-man timer runs to-nvidia.sh when the window expires, so a dropped
# SSH session cannot leave the host on nouveau.
#
# DO NOT run without the owner's go-ahead for this window: it stops gdm.
set -euo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
# sudo drops the environment: hand the knobs through explicitly.
[ "$(id -u)" = 0 ] || exec sudo KGP_ROOT="$KGP_ROOT" \
   KGP_NOUVEAU_ARGS="${KGP_NOUVEAU_ARGS:-}" KGP_PCI_RESET="${KGP_PCI_RESET:-}" \
   KGP_BAR1_SIZE="${KGP_BAR1_SIZE:-}" "$(readlink -f "$0")" "$@"

WINDOW_MIN="${1:-120}"
HERE="$(dirname "$(readlink -f "$0")")"
STATE=/run/koregp
# Hardware knobs for the first windows; see docs/test-host.md "Unknowns".
NOUVEAU_ARGS="${KGP_NOUVEAU_ARGS:-modeset=2 config=NvGspRm=1}"   # headless, GSP firmware
PCI_RESET="${KGP_PCI_RESET:-0}"                                  # 1: reset GPU between drivers
# Resize BAR1 while no driver is bound, e.g. 13 for 8 GiB (kgp_bar1_bits in
# env.sh). Empty: leave it alone. Not persistent, and to-nvidia.sh puts the
# old size back, but it adds a way for the restore to fail: only use it in a
# window where the owner has agreed to that. See docs/test-host.md.
BAR1_SIZE="${KGP_BAR1_SIZE:-}"

[ "$(kgp_gpu_driver)" = nvidia ] || { echo "GPU driver is '$(kgp_gpu_driver)', expected nvidia" >&2; exit 1; }

# Refuse if a person is at the desk or anything real is using the GPU.
# (Captured, not piped into `grep -q`: under pipefail that can read false.)
desk_sessions=$(loginctl list-sessions --no-legend | awk '$4 == "seat0" && $3 != "gdm"')
if [ -n "$desk_sessions" ]; then
   echo "refusing: a user session is active on seat0" >&2; exit 1
fi
compute_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [ -n "$compute_apps" ]; then
   echo "refusing: compute processes are using the GPU" >&2; exit 1
fi
# The udev queue must be quiet: a backlog of nvidia driver events is how the
# load/unload loop of 2026-11-20 started (docs/test-host.md).
if ! udevadm settle -t 20; then
   echo "refusing: udev is still busy; try again when 'udevadm settle' returns" >&2; exit 1
fi

# to-nvidia.sh pins nvidia in use for two minutes after a restore.
if kill -0 "$(cat "$STATE/pin" 2>/dev/null)" 2>/dev/null; then
   echo "refusing: the last restore is still settling; try again in two minutes" >&2; exit 1
fi

mkdir -p "$STATE"
for u in gdm nvidia-persistenced; do
   systemctl is-active --quiet "$u" && echo "$u"
done > "$STATE/units" || true

# Arm the dead-man before touching anything. A unit left over from an
# earlier window (stopped, or failed) would make systemd-run refuse the name.
systemctl stop koregp-restore.timer 2>/dev/null || true
systemctl reset-failed koregp-restore.service koregp-restore.timer 2>/dev/null || true
systemd-run --quiet --unit=koregp-restore --on-active="${WINDOW_MIN}min"    --setenv=KGP_ROOT="$KGP_ROOT" "$HERE/to-nvidia.sh"

# From here on, any failure restores nvidia.
trap '"$HERE/to-nvidia.sh"' ERR

while read -r u; do systemctl stop "$u"; done < "$STATE/units"
sleep 2
modprobe -r nvidia_uvm nvidia_drm nvidia_modeset nvidia
# Ubuntu's udev rules react to the driver going away; let them finish.
udevadm settle -t 30

[ "$PCI_RESET" = 1 ] && echo 1 > "/sys/bus/pci/devices/$KGP_GPU/reset"

if [ -n "$BAR1_SIZE" ]; then
   kgp_bar1_bits > "$STATE/bar1_orig"
   # A refusal (no room in the bridge window) is an answer, not a failure;
   # the kernel keeps the old size.
   echo "$BAR1_SIZE" > "/sys/bus/pci/devices/$KGP_GPU/resource1_resize" ||
      echo "BAR1 resize to 2^$BAR1_SIZE MiB refused by the kernel"
   echo "BAR1 is now 2^$(kgp_bar1_bits) MiB (was 2^$(cat "$STATE/bar1_orig"))"
   # An unassigned BAR1 means neither driver will work: restore now.
   [ "$(kgp_bar1_bits)" != none ]
fi

empty=$(mktemp -d)
# shellcheck disable=SC2086
modprobe -C "$empty" nouveau $NOUVEAU_ARGS
rmdir "$empty"

for _ in $(seq 20); do
   [ "$(kgp_gpu_driver)" = nouveau ] && break
   sleep 0.5
done
[ "$(kgp_gpu_driver)" = nouveau ]

# The render node is root:render 0660 and the test user is in neither group
# (the proprietary driver never needed it). Hand this window's node to the
# invoking user; it disappears with nouveau, so nothing persists.
udevadm settle -t 20 || true
if [ -n "${SUDO_UID:-}" ]; then
   # (card*: the node a display window presents through.)
   for node in /dev/dri/renderD* /dev/dri/card*; do
      [ -e "$node" ] && chown "$SUDO_UID" "$node"
   done
fi

trap - ERR
echo "GPU is on nouveau; auto-restore in ${WINDOW_MIN} min (to-nvidia.sh to restore now)"
dmesg | grep -i -E "nouveau.*(gsp|chipset|bios version|DRM: (VRAM|GART))" | tail -8 || true
