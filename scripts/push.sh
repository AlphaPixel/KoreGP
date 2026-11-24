#!/bin/bash
# Copy the tracked + untracked-unignored files of this repo to the test host.
# Run from Git Bash on the workstation: scripts/push.sh
set -euo pipefail
HOST="${KGP_HOST:-argon-isotopia}"
cd "$(git rev-parse --show-toplevel)"
git ls-files -co --exclude-standard -z |
   tar --null -T - -cz |
   ssh -o BatchMode=yes "$HOST" 'mkdir -p ~/koregp/KoreGP && tar xz -C ~/koregp/KoreGP && chmod +x ~/koregp/KoreGP/scripts/host/*.sh'
