#!/bin/bash
# Restore the proprietary nvidia driver after a nouveau test window.
# Safe to run at any time and from any state; also run by the dead-man timer.
#
# If this script cannot restore nvidia it says so and stops. It never reboots:
# this host's boot path is fragile (docs/test-host.md) and a reboot is the
# owner's call.
#
# Three things here were learned the hard way on 2026-11-20 (docs/test-host.md,
# "The first window"):
#  - never pipe into `grep -q` under pipefail to test for a module;
#  - `modprobe -r nouveau` is a silent no-op on this host (alias nouveau off);
#  - never load nvidia while nouveau still holds the card, and never unload
#    and reload it in quick succession: Ubuntu's udev rules turn the resulting
#    driver add/remove events into an endless load/unload loop.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
[ "$(id -u)" = 0 ] || exec sudo KGP_ROOT="$KGP_ROOT" "$(readlink -f "$0")" "$@"

STATE=/run/koregp

# Disarm the dead-man (harmless if we *are* the dead-man's service).
systemctl stop koregp-restore.timer 2>/dev/null

if grep -q '^nouveau ' /proc/modules; then
   # Anything still holding the nouveau node is one of our tests.
   fuser -s -k -TERM /dev/dri/renderD* /dev/dri/card* 2>/dev/null && sleep 2
   fuser -s -k -KILL /dev/dri/renderD* /dev/dri/card* 2>/dev/null && sleep 1
   # In a display window (nouveau loaded with modeset=1) the kernel console
   # sits on nouveau's framebuffer: take it off before the unload. In an
   # ordinary window nothing is bound and this does nothing.
   for vt in /sys/class/vtconsole/vtcon*; do
      if grep -q "frame buffer" "$vt/name" 2>/dev/null &&
         [ "$(cat "$vt/bind" 2>/dev/null)" = 1 ]; then
         echo 0 > "$vt/bind"
      fi
   done
   # Bypass the modprobe config, as to-nouveau.sh does to load it.
   empty=$(mktemp -d)
   modprobe -C "$empty" -r nouveau
   rmdir "$empty"
fi

if grep -q '^nouveau ' /proc/modules; then
   cat >&2 <<EOF
RESTORE FAILED: nouveau would not unload, so nvidia was not loaded
(loading it now would only make things worse). Something still holds the
device: see 'sudo fuser -v /dev/dri/*'. Do NOT reboot remotely -- tell the
owner. dmesg tail follows.
EOF
   dmesg | tail -15 >&2
   exit 2
fi

# Put BAR1 back to the size nvidia was running with, if a window changed it.
if [ -f "$STATE/bar1_orig" ] && [ "$(kgp_gpu_driver)" = none ]; then
   cat "$STATE/bar1_orig" > "/sys/bus/pci/devices/$KGP_GPU/resource1_resize" ||
      echo "WARNING: could not restore BAR1 size" >&2
   rm -f "$STATE/bar1_orig"
fi

# Let udev finish with whatever the unload produced before giving it more.
udevadm settle -t 30

# Loading is idempotent, so this is also right when nvidia never left.
modprobe nvidia_drm
modprobe nvidia_uvm

# Pin nvidia in use while udev reacts to the driver appearing. With the
# module unable to unload, a stale "driver removed" event cannot start the
# load/unload loop (scripts/host/break-nvidia-udev-loop.sh explains it).
if [ -e /dev/nvidiactl ]; then
   setsid sh -c 'sleep 120 < /dev/nvidiactl' > /dev/null 2>&1 &
   # to-nouveau.sh refuses while this is alive: it could not unload nvidia.
   mkdir -p "$STATE"; echo $! > "$STATE/pin"
fi
udevadm settle -t 60
modprobe nvidia_drm
modprobe nvidia_uvm

if [ "$(kgp_gpu_driver)" = nvidia ] && nvidia-smi >/dev/null 2>&1; then
   if [ -f "$STATE/units" ]; then
      # nvidia-persistenced first, gdm last.
      sort -r "$STATE/units" | while read -r u; do systemctl start "$u"; done
      rm -f "$STATE/units"
   fi
   echo "GPU is on nvidia: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
   exit 0
fi

# gdm and nvidia-persistenced are left stopped (they need the driver); the
# list stays in $STATE/units so a later run of this script restarts them.
cat >&2 <<EOF
RESTORE FAILED: GPU driver is '$(kgp_gpu_driver)' and nvidia-smi does not
answer. Do NOT reboot remotely and do not retry in a loop -- tell the owner.
dmesg tail follows.
EOF
dmesg | tail -30 >&2
exit 2
