#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <base_shards_dir>"
  exit 1
fi

BASE_DIR="$1"
echo "▶︎ Flattening nested dirs under $BASE_DIR …"

for i in $(seq 1 35); do
  SHARD="$BASE_DIR/m$i"
  if [ ! -d "$SHARD" ]; then
    echo "  – Shard $i not found, skipping"
    continue
  fi
  echo "  ▶︎ Processing shard $i …"

  # 1) Move every file two‐or‐more levels deep into the shard root
  find "$SHARD" -mindepth 2 -type f -print0 \
    | xargs -0 -I{} mv -n {} "$SHARD/"

  # 2) Remove any directories left under the shard
  find "$SHARD" -mindepth 1 -type d -print0 \
    | xargs -0 rm -rf

  echo "    ✓ Flattened $SHARD"
done

echo "✔️  All done!"