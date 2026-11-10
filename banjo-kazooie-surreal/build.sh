#!/bin/bash
# Build the Surreal mod from a Banjo-Kazooie decomp checkout.
#
# usage: build.sh /path/to/banjo-kazooie
#
# The checkout must be at commit e8d31fa5 (the commit surreal.patch was made against),
# contain your own baserom.us.v10.z64, and have been built once with plain `make` so that
# assets/ exists. cargo must be installed, or make silently ignores asset changes.
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
DECOMP="${1:?usage: build.sh /path/to/banjo-kazooie}"
cd "$DECOMP"

[ -f baserom.us.v10.z64 ] || { echo "baserom.us.v10.z64 not found in $DECOMP"; exit 1; }
[ -f assets/assets.yaml ] || { echo "assets/ not extracted yet: run 'make' once first"; exit 1; }
command -v cargo >/dev/null || { echo "cargo is required (asset edits are ignored without it)"; exit 1; }

# 1. engine changes
if [ ! -f src/core2/gc/surreal.inc ]; then
    git apply --check "$HERE/surreal.patch"
    git apply "$HERE/surreal.patch"
fi

# 2. asset changes, always generated from a pristine copy
[ -d assets.vanilla ] || cp -r assets assets.vanilla
python3 "$HERE/tools/recolor.py" assets.vanilla assets
python3 "$HERE/tools/dreamtalk.py" assets.vanilla assets

# 3. rebuild (the patch turns the anti-tamper flags off; make doesn't notice flag changes)
rm -rf build/us.v10/src
make -j"$(nproc)"

echo
echo "Surreal ROM: $DECOMP/build/us.v10/banjo.us.v10.z64"
echo "('baserom.us.v10.z64 differs' above is expected: it is no longer the original game)"
