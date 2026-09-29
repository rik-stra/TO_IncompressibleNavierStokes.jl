#!/bin/bash
# waitfor.sh <D6mini name> [n = 230]: return once analysis/output/D6mini_<name> holds n member files
# and no run_d6.jl is still writing into it (the file count reaches n just before the last write closes).
D=/export/scratch2/rik/time_series_TO/lib/RikFlow/analysis/output/D6mini_$1
n=${2:-230}
until [ -d "$D" ] && [ "$(ls -- "$D" | wc -l)" -ge "$n" ]; do sleep 60; done
sleep 90
echo "$(date +%T) $1 complete: $(ls -- "$D" | wc -l) files"
