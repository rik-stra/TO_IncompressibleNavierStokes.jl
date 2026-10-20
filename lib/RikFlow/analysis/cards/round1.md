# Card: round 1 — step 1, a coloured residual (paper Sec. 6.1), opened 2026-10-06, closed —

WORKFLOW.md Stage 0. One card for the round's nine closures: they share the protocol, the
criterion and the code path, and differ in the rows of the table below. Methods text:
`closures.tex` §Step 1 (`sec:lever-colour`) and `to_method.tex` §LinReg + MVG; implementation check:
`paper/round1_gate01.md`.

🔒 **Once any full-hindcast number of a round-1 closure exists, nothing above "Post hoc" changes.**
Late observations go under "Post hoc", dated.

- [ ] Rik has seen this card (date: ____)

## Closures

All in the LinReg path (`D6_CLOSURE=lrs`), fitted on 1–10 TU, `ridge_solver = :exact`, h = 5,
`penalize_intercept = false`, deployed with the turbulence gate. Every LinReg-path closure uses the
same member seeds (`member_seed(k, m)`), so paired differences within the round share their random
draws (Sec. 3.4).

| run | closure | the one change | matched partner (secondary) | artefact (`TO_LRS/`) | expected, and why (Sec. 6.1) |
|---|---|---|---|---|---|
| S1.1 | LinReg7 + AR(2) | AR(2) residual on LinReg7 (λ = 1 on every QoI) | LinReg7 | `LinReg7_ar2n` | vs LinReg7: more calibrated cells (ridge coloured the residual; the white draw lost its low-frequency power, hence LinReg7's 5/36) and a lower CRPS_0.5 at short leads (AR state starts from the realized residuals). vs LinReg1: **no improvement** — LinReg7 is 18 % behind on CRPS_0.5, and ridge on every QoI removed the dynamics the AR(2) restores only in part |
| S1.2a | LinReg^E, white, λ = 0.1 | ridge on E[0,6] only (T_int < h rule) | LinReg1 | `Splice1_E0x11` (row from LinReg11) | ≈ LinReg1: the E[0,6] correction decorrelates in ~4 steps, so its lags carry little; its residual becomes slightly coloured and is drawn white. **No improvement** expected (partner for S1.3a) |
| S1.2b | LinReg^E, white, λ = 1 | as S1.2a | LinReg1 | `Splice1_E0x7` | as S1.2a; E[0,6]'s residual lag-1 0.44 (colour table) is lost by the white draw → its cells under-dispersed |
| S1.2c | LinReg^E, white, λ = 10 | as S1.2a | LinReg1 | `Splice1_E0x8` | as S1.2b, more so |
| S1.3a | LinReg^E + AR(2), λ = 0.1 | AR(2) on S1.2a | S1.2a | `Splice1_E0x11_ar2n` | vs S1.2a: little to gain (little colour at λ = 0.1). vs LinReg1: level |
| S1.3b | LinReg^E + AR(2), λ = 1 | AR(2) on S1.2b | S1.2b | `Splice1_E0x7_ar2n` | vs S1.2b: more calibrated cells, lower CRPS_0.5 (colour restored in E[0,6]). vs LinReg1: **the round's candidate** — CRPS_0.5 level with LinReg1, calibrated count higher → clause C |
| S1.3c | LinReg^E + AR(2), λ = 10 | AR(2) on S1.2c | S1.2c | `Splice1_E0x8_ar2n` | vs S1.2c: as S1.3b. vs LinReg1: calibration up, but more skill lost in E[0,6]'s mean than at λ = 1 → C possible, S not |
| S1.4 | paper 3's rule + AR(2), base λ = 0.03 | ridge also on the other five QoIs (λ_i = λ(σ_i/σ_1)², most on the small scales) | S1.3a (λ = 0.1; E[0,6] gets 0.06) | `LinReg14_ar2n` | vs S1.3a: CRPS_0.5 higher (ridge on the persistent small-scale corrections, where the lags carry the skill); calibration up in the bands it colours. vs LinReg1: **no improvement** on S; C uncertain |
| S1.5 | paper 3's rule + AR(2), base λ = 0.3 | as S1.4 | S1.3b (λ = 1; E[0,6] gets 0.62) | `LinReg16_ar2n` | as S1.4, more skill lost |

Artefacts checked on the desktop 2026-10-06 (`meta_files/handoff_desktop_round1_2026-10-06.md`, Task A): all `:exact` or λ = 0, (400, 4000), h = 5; every AR(2) re-fitted by `lrs_ar_variant.jl --report` and equal to the file. 2026-10-07: the `*_ar2n` artefacts (constrained fit) replace the `*_ar2` ones; all six built and `--report`-checked on the desktop 2026-10-07 @ `09099363` (results_LSTMS.md §16). Copy the desktop's six to Snellius before the runs.

## Clauses (fixed; paper Sec. 3.7, Rik 2026-10-06)

Hindcast K = 90 ICs, M = 10, 100 + 1200 steps, leads 25/50/100/200/400/1000; policy A in the
verdict, policy B reported beside; long runs 5 × 100 TU. Against **LinReg1** (the reused `D6_LinReg1`
and its R2 long runs):

- **G** KS guard: FAIL iff all 5 single-run summed KS lie above all 5 of LinReg1's (p = 1/252).
- **S** CRPS_0.5 paired difference: 90 % CI upper end < 0.
- **C** calibrated cells (0.8 ≤ r ≤ 1.25, 36 cells): paired CI of the count difference > 0, AND the
  CRPS_0.5 CI upper end < +10 % of LinReg1's CRPS_0.5.
- **Verdict:** IMPROVES iff G passes and (S or C).

Against the **matched partner** (mechanism, secondary, no verdict): paired CRPS_0.5, paired
calibrated-count difference, and for every AR-vs-white pair the paired change in normalized error
(read as the mean) and in calibrated count (read as the spread; Sec. 6.1, review A4).

Reported for every closure, no clause: CRPS_all, normalized error, flat count with its reference,
calibrated count's reference, mean bias by lead, long-run bias and spread–skill, divergences
(members / ICs), gate firings (hindcast: % of steps, members, ICs; long runs: % of steps).

Tool: `julia --project=analysis analysis/score_d6.jl --compare <closure dir> <partner dir>`
(`D6_EXCLUDE_ICS` = the union of both closures' diverged ICs for policy A).

### Amendment 2026-10-08 (Rik), before any round-1 closure was scored

Only the reused baselines had been scored when these were changed.

1. **Divergences: lead-wise rule** replaces policy A/B. A diverged member is scored at every step
   before its stop and left out of both closures from there (`score_d6.jl` `LEADWISE`, the
   default). No `D6_EXCLUDE_ICS`. Reason: Rik, diverged runs "should be included as much as
   possible"; policy A dropped whole ICs, conditioning the closure that diverged in its own favour.
2. **CRPS_0.5 and CRPS_all are window averages over every forecast step**: steps 1–200 and 1–1200
   (step 1 = the first closure step after the 100-step replay), over the ICs and the six QoIs. The
   normalized error likewise: δ₀.₅ (1–200) and δ_all (1–1200); the mean/spread reading of an
   AR-vs-white pair uses δ₀.₅. Reason: the fair CRPS is ≥ 0 and proper per step, and the squared
   error cannot cancel, so a window average hides nothing; the grid weighted each doubling of lead
   equally and made CRPS_all mostly a short-lead score.
3. **Unchanged, per lead (36 cells):** r and the calibrated count (clause C), the rank histograms
   and flat count, the bias by lead. Reason: r is two-sided, so too narrow early and too wide late
   would cancel in a window, and pooled sums of squares are dominated by the long leads.
4. **Margin stays 10 %** (the half-width moves 7.0 → 7.3 % of LinReg1's CRPS_0.5 on the baseline
   pair, so a closure as good as LinReg1 passes the margin about 0.73 of the time, was 0.76).

Clause S and the CRPS part of clause C now read the window CRPS_0.5. Plan and measurements:
`meta_files/handoff_window_scores_2026-10-08.md`; paper Sec. 3.4.

5. **Later the same day (Rik):** no δ_all (past 0.5 TU the mean error approaches climatology for
   every closure; it does not resolve LinReg1 vs LinReg7 and has no attribution role); no flat count
   as a reported number (it overlaps the calibrated count; the rank histograms stay as a figure, for
   their slope, which r does not show). So "Reported for every closure" above now reads: CRPS_all,
   δ₀.₅, calibrated count's reference, rank-histogram figure (slopes), mean bias by lead, long-run
   bias and spread–skill (per run), divergences, gate firings.

### Amendment 2026-10-13 (Rik), before any round-1 closure was run

6. **Criterion, two baselines** (paper Sec. 3.7): the KS guard G compares with **LinReg7** (the
   better baseline in the long runs): FAIL iff all 5 single-run summed KS lie above all 5 of
   LinReg7's. Clauses S and C still compare with LinReg1. Against a matched partner, the partner
   replaces both. Reason: against LinReg7 alone an improvement is cheap (LinReg6, indistinguishable
   from LinReg1, passes S and C against LinReg7); LinReg1's long runs depend on the guard
   (results_LSTMS.md §22–23).
7. **Training hindcast, a diagnostic** (Rik: "just evaluate on the training hindcast", no tuning):
   5 ICs inside the fit window (k = 5, 11, 16, 22, 28; t = 1.00–6.75 TU; warm-up and the 3 TU
   forecast inside steps 400–4000; `build_d6_ics.jl training`, `select_train_ics`, V87), M = 10, the
   same member seeds, for LinReg1, LinReg6, LinReg7 and every round-1 closure; output
   `D6train_<closure>`. Reported for every closure: divergences, gate firings, spread–skill pooled
   over leads ≤ 0.5 TU and over 1–3 TU, pooled rank histograms, ensemble-mean bias. **Not scored
   against the held-out set, no clause, no tuning.** A closure skips its full hindcast only for a
   defect: a diverged member, or a pooled spread–skill ratio (leads ≤ 0.5 TU) more than a factor 2
   from its matched partner's in any QoI. Everything else runs.
   *Corrected the same day (Rik: "go with b"), after the baselines' training hindcasts and before
   any round-1 closure's:* the first version ("a ratio outside [0.5, 2]") flagged LinReg7 itself
   (0.39–0.44 in-sample; held-out median 0.56), so it would have skipped closures for the real
   under-dispersion step 1 tests rather than for a broken artefact. Partners as in the table above
   (S1.1 → LinReg7; S1.2a–c → LinReg1; S1.3a–c → S1.2a–c; S1.4 → S1.3a; S1.5 → S1.3b).

## Settings and where they come from

- No offset and no noise scale fitted against coupled runs (D-10: none in round 1).
- Fixed by rule: E[0,6] alone (T_int < h on 1–10 TU, `analysis/tint_rule.jl`); AR(2) fitted to the
  residual ACF at lags 1–20 on 1–10 TU, **constrained to φ1 ≥ 0, φ2 ≤ 0** (poles with non-negative
  real part), Σ_ξ = DRD; the λ ladder 0.1 / 1 / 10, all reported.
- 🔒 **2026-10-07, before any round-1 hindcast (Rik: "make it the rule"):** the AR(2) constraint
  replaces the unconstrained fit in every AR(2) closure, on offline evidence only (training-window
  residual, tracked dQ, the replay response kernel; `results_LSTMS.md` §16). It binds materially in
  Z[0,6] and Z[16,32] of S1.1, S1.4, S1.5 (negative real pole removed); in S1.3a–c only the white
  LinReg1 rows move (|φ| ≤ 0.017 → 0). Artefacts renamed `*_ar2n`; the `*_ar2` ones are not run.
  ⚠️ Cost found after the decision: one-step under-dispersion where it binds (data innovation sd /
  σ_ξ 1.17 and 1.11 in LinReg7's Z[0,6] and Z[16,32], vs 1.01 and 1.04 unconstrained); disclosed in
  closures.tex Step 1. [x] Rik confirmed the rule 2026-10-07, after the offline comparison with AR(1) + white
  and two AR(1)s (results_LSTMS.md §18): "keep the constrained AR(2)".
- Pilot-chosen, disclosed (Sec. 3.8): LinReg7's λ = 1; paper 3's base 0.03 / 0.3; the AR order 2;
  the E[0,6] rule itself was formulated after a pilot scan.

## Post hoc

- **2026-10-13, training hindcast (amendment 7), after the round-1 closures' training runs.** No
  member diverged in any closure. The defect rule (ratio ≤ 0.5 TU more than a factor 2 from the
  partner's) flagged **S1.1** only: LinReg7 + AR(2) / LinReg7 = 1.82 1.42 2.12 2.14 2.06 2.06
  (S1.1's own ratios 0.73 0.62 0.83 0.84 0.87 0.86, LinReg7's 0.40–0.44). That is the widening step 1
  predicts for S1.1 ("vs LinReg7: more calibrated cells"), not a broken artefact (its fans and
  ratios are ordinary, close to LinReg1's). **Rik: "go with i, run S1.1 anyway".** S1.1 gets its
  full hindcast; the rule's intent (catch a broken artefact) is unchanged, and this exception is the
  only one. The other eight: ratio to partner 0.56–1.79, no defect. Numbers: results_LSTMS.md §25.
- **2026-10-15, LinReg7 + AR(1) added after S1.1 was scored (Rik: "add a closure to run online based
  on linreg7+AR1 fit").** `LinReg7_ar1ls`: the AR(1) fitted as the AR(2) is, by LS on the residual ACF
  at lags 1–20 (`ar1_ls`, the AR(1) of paper Fig. colour-resacf-ar1), Σ_ξ = DRD as in S1.1. Not a
  registered round-1 closure: an ablation of the AR order, compared with S1.1 (same mean, same window)
  and LinReg7. Not the old `LinReg7_ar1` (φ = ρ̂(1), §12e′). Built before any of its runs; its offline
  check predicts one-step over-dispersion in the middle band (data innovation sd / σ_ξ 0.61, 0.53).
