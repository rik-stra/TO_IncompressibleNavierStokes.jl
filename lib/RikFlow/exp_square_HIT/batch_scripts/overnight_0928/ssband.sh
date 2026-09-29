#!/bin/bash
# ssband.sh <score log> ...: from score_d6.jl's "LEVEL q -- primary" block, the spread-skill ratio per
# (QoI, lead), cells inside S7's [0.8, 1.25], and the median ratio; overall and for leads <= 400.
for f in "$@"; do
  sed 's/\x1b\[[0-9;]*m//g' -- "$f" | awk -v name="$(basename -- "$f")" '
    /^LEVEL q/ {on=1; next}
    on && /^[A-Z]/ && !/^LEVEL/ {on=0}
    on && /saturation lead/ {q=$1}
    on && $1 ~ /^[0-9]+$/ && NF >= 5 {r=$4; n++; rs[n]=r; ib += (r>=0.8 && r<=1.25);
       if ($1 <= 400) {m++; ib4 += (r>=0.8 && r<=1.25); row4[q]=row4[q] sprintf(" %.2f", r)} }
    END {asort(rs); med = (n%2) ? rs[(n+1)/2] : (rs[n/2]+rs[n/2+1])/2;
         printf "%s: %d/%d cells in [0.8,1.25] (%d/%d at leads <= 400), median %.3f\n", name, ib, n, ib4, m, med;
         for (k in row4) printf "   %-9s leads 25..400: %s\n", k, row4[k] }'
done
