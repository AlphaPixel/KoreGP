#!/bin/bash
# The first nouveau windows on this host, scripted (docs/plan.md, W0).
#
#    w0.sh swap       swap to nouveau, run hardware-suite.sh, swap back
#    w0.sh deadman    swap to nouveau and do nothing: the 3-minute dead-man
#                     timer has to bring nvidia back on its own
#    w0.sh bar1 <n>   like swap, with BAR1 resized to 2^n MiB (13 = 8 GiB)
#                     while no driver is bound. Extra risk, see
#                     docs/test-host.md; only after swap and deadman passed.
#    w0.sh display    display shakedown: swap to nouveau with modesetting
#                     on, so that it drives the monitor and the console,
#                     report what it sees, swap back. Nothing of KoreGP
#                     runs. Extra risk, see docs/test-host.md.
#    w0.sh present    like swap, in a display window: the suite then also
#                     presents on the monitor. Only after display passed.
#
# RING 3. Needs the owner's go-ahead for each run, with the owner able to
# reach the machine: if nvidia cannot be restored the only fix is a reboot,
# and this host's reboot is fragile. Nothing here reboots.
set -uo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
. "$HERE/env.sh"

status() { "$HERE/gpu-status.sh"; }

case "${1:-}" in
swap|bar1|display|present)
   cmd="$HERE/hardware-suite.sh"; window=20
   if [ "$1" = bar1 ]; then
      [ -n "${2:-}" ] || { echo "usage: $0 bar1 <log2 MiB>" >&2; exit 64; }
      export KGP_BAR1_SIZE="$2"
   fi
   if [ "$1" = present ]; then
      # atomic=1: Mesa's display WSI is atomic-only, and nouveau does not
      # expose the atomic ioctl unless asked (results-n1.md, tenth window).
      export KGP_NOUVEAU_ARGS="modeset=1 atomic=1 config=NvGspRm=1"
   fi
   if [ "$1" = display ]; then
      export KGP_NOUVEAU_ARGS="modeset=1 config=NvGspRm=1"
      cmd="$HERE/display-report.sh"; window=5
      echo "== console before"; cat /sys/class/vtconsole/vtcon*/bind | tr '\n' ' '; echo
   fi
   # Otherwise "before" and "after" are both nouveau and compare as restored.
   [ "$(kgp_gpu_driver)" = nvidia ] || {
      echo "GPU is on '$(kgp_gpu_driver)', not nvidia: a window is already open; wait for it to close" >&2
      exit 1
   }
   echo "== before"; status | tee /tmp/kgp-w0-before
   "$HERE/with-nouveau.sh" -w "$window" "$cmd"
   rc=$?
   if [ "$1" = display ]; then
      sleep 5   # the console comes back with nvidia's framebuffer
      echo "== console after"; cat /sys/class/vtconsole/vtcon*/bind | tr '\n' ' '; echo
   fi
   echo "== after"; status | tee /tmp/kgp-w0-after
   if diff <(grep -v -E '^(NVIDIA|modules)' /tmp/kgp-w0-before) \
           <(grep -v -E '^(NVIDIA|modules)' /tmp/kgp-w0-after) >/dev/null; then
      echo "host state restored (driver, units, BAR1, timer match)"
   else
      echo "HOST STATE DIFFERS from before the window -- look at the two blocks above"
      rc=2
   fi
   exit $rc
   ;;
deadman)
   echo "== before"; status
   "$HERE/to-nouveau.sh" 3 || exit 1
   echo "on nouveau; waiting for the dead-man (3 min) without touching anything"
   for _ in $(seq 60); do
      sleep 5
      [ "$(kgp_gpu_driver)" = nvidia ] && break
   done
   sleep 10   # let gdm come back up
   echo "== after"; status
   if [ "$(kgp_gpu_driver)" = nvidia ] && systemctl is-active --quiet gdm; then
      echo "dead-man restored nvidia on its own"
   else
      echo "DEAD-MAN DID NOT RESTORE nvidia. Run to-nvidia.sh; do not reboot remotely."
      exit 2
   fi
   ;;
*)
   sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
   exit 64
   ;;
esac
