#!/bin/bash
# Recovery for one specific failure: the nvidia modules loading and
# unloading in an endless loop after a botched restore.
#
# Ubuntu's /lib/udev/rules.d/71-nvidia.rules answers "nvidia driver added"
# with `modprobe nvidia-modeset/-drm/-uvm` and "nvidia driver removed" with
# `modprobe -r` of the same. A full unload emits "removed" and a full load
# emits "added", so once one of each is in udev's queue while nothing holds
# the driver, each keeps regenerating the other, every few seconds, forever.
#
# The loop ends when the unload half fails: with nvidia pinned in use, a
# queued "removed" can no longer unload it (so emits nothing) and a queued
# "added" finds it loaded (so emits nothing). This script pauses udev's
# queue, makes sure nvidia is loaded, pins it by holding its control device
# open, lets the queue drain, then brings the rest up once, in order.
#
# Changes nothing persistent. Does not reboot.
set -uo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
[ "$(id -u)" = 0 ] || exec sudo "$(readlink -f "$0")" "$@"

if grep -q '^nouveau ' /proc/modules; then
   echo "nouveau is loaded; this script is only for the nvidia load/unload loop" >&2
   exit 1
fi

udevadm control --stop-exec-queue
trap 'udevadm control --start-exec-queue' EXIT
sleep 12                      # let the command udev is running finish

grep -q '^nvidia ' /proc/modules || modprobe nvidia
sleep 1

# The pin: an open file on the control device keeps the module's use count
# above zero for ten minutes, far longer than the queue needs.
setsid sh -c 'sleep 600 < /dev/nvidiactl' > /dev/null 2>&1 &
sleep 1
if ! pgrep -f 'sleep 600' > /dev/null; then
   echo "could not open /dev/nvidiactl to pin the driver; nothing else done" >&2
   exit 1
fi

udevadm control --start-exec-queue
trap - EXIT
if ! udevadm settle -t 150; then
   echo "udev's queue did not drain; the loop may still be running" >&2
   exit 1
fi
echo "udev queue drained"

modprobe nvidia_modeset
modprobe nvidia_drm
modprobe nvidia_uvm
systemctl start nvidia-persistenced
systemctl start gdm
sleep 10
"$HERE/gpu-status.sh"
