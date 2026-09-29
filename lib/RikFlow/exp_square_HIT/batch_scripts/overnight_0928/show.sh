#!/bin/bash
# show.sh <start-regex> <nlines> <log> [<log> ...]: print each log (colour codes stripped) from the first match of <start-regex>.
pat=$1; n=$2; shift 2
for f in "$@"; do
  echo "== $(basename -- "$f")"
  sed 's/\x1b\[[0-9;]*m//g' -- "$f" | grep -v '^\s*$' | awk -v p="$pat" 'found || $0 ~ p {found=1; print}' | head -n "$n"
done
