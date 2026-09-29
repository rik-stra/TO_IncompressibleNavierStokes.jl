# Overnight 2026-09-28/29 — the scripts behind `analysis/results_LSTMS.md` §13

Desktop-only shell glue, kept for the record and for reproduction. ⚠️ They hard-code the desktop
paths of that night (the `.claude/worktrees/overnight` checkout, the job-scratch log directory);
edit `R`/`L` before reuse.

| script | what it does |
|---|---|
| `gpu_queue_night8.sh` | the last of eight queue orderings: one mini-D6 at a time on the 3090 (`run_d6.jl --all`, selection block, M = 5 or 10, nlead 400), skipping complete runs |
| `run_ar2_3tu.sh` | 3 TU run on the full D6's 19 selection ICs (odd ordinals 89–125) × M = 10, `V=<model>` |
| `eval_mini.sh <run> <baseline>...` | paired primary score against each baseline, unpaired score, tail census |
| `eval_sel3.sh <model>` | the same for a 3 TU run, paired with `D6_LinReg1` / `D6_LinReg7` on the same 19 ICs |
| `eval_sub19.sh <A> <B>` | a mini-D6 pair restricted to those 19 ICs (IC-subset control, §13g) |
| `ssband.sh <score log>...` | spread–skill cells inside S7's [0.8, 1.25] from `score_d6.jl`'s LEVEL-q table (⚠️ it counts one spurious "0.00" cell from the summary line; subtract it: n/31 → n/30) |
| `show.sh`, `waitfor.sh` | log printing, wait-for-run |
| `fit_m3f_mem.sh` | the §13d recurrent-memory M3ᶠ fit |
