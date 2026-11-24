#!/bin/bash
# Read-only: which driver has the GPU, and is a restore timer armed.
. "$(dirname "$(readlink -f "$0")")/env.sh"
echo "driver:  $(kgp_gpu_driver)"
echo "modules: $(lsmod | awk '$1 ~ /^(nvidia|nouveau)/ {printf "%s(%s) ", $1, $3}')"
echo "units:   gdm=$(systemctl is-active gdm) nvidia-persistenced=$(systemctl is-active nvidia-persistenced)"
echo "bar1:    2^$(kgp_bar1_bits) MiB"
echo "timer:   $(systemctl is-active koregp-restore.timer 2>/dev/null)"
[ "$(kgp_gpu_driver)" = nvidia ] && nvidia-smi --query-gpu=name,driver_version,memory.used --format=csv,noheader
exit 0
