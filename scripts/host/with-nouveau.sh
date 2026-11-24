#!/bin/bash
# Run one command inside a nouveau window and always restore nvidia after.
# Usage: with-nouveau.sh [-w minutes] <command...>
# The command runs as the invoking user with the private Mesa build selected.
set -uo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
. "$HERE/env.sh"

WINDOW=30
[ "${1:-}" = -w ] && { WINDOW="$2"; shift 2; }
[ $# -gt 0 ] || { echo "usage: $0 [-w minutes] <command...>" >&2; exit 64; }

"$HERE/to-nouveau.sh" "$WINDOW" || exit 1
kgp_run "$@"
rc=$?
"$HERE/to-nvidia.sh" || exit 2
exit $rc
