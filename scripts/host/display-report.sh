#!/bin/bash
# Read-only: what nouveau made of the display, for a window opened with
# modesetting on (w0.sh display). Changes nothing.
for c in /sys/class/drm/card*-*; do
   echo "$(basename "$c"): $(cat "$c/status") $(cat "$c/enabled") $(head -n 1 "$c/modes" 2>/dev/null)"
done
for v in /sys/class/vtconsole/vtcon*; do
   echo "$(basename "$v"): $(cat "$v/name") bind=$(cat "$v/bind")"
done
echo "fb: $(cat /proc/fb)"
echo "nouveau use count: $(awk '$1 == "nouveau" {print $3}' /proc/modules)"
ls -l /dev/dri/
sudo dmesg | grep -i -E "nouveau|fbcon|gsp" | tail -n 30
exit 0
