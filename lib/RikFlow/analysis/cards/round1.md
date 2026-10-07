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
| S1.1 | LinReg7 + AR(2) | AR(2) residual on LinReg7 (λ = 1 on every QoI) | LinReg7 | `LinReg7_ar2` | vs LinReg7: more calibrated cells (ridge coloured the residual; the white draw lost its low-frequency power, hence LinReg7's 5/36) and a lower CRPS_0.5 at short leads (AR state starts from the realized residuals). vs LinReg1: **no improvement** — LinReg7 is 18 % behind on CRPS_0.5, and ridge on every QoI removed the dynamics the AR(2) restores only in part |
| S1.2a | LinReg^E, white, λ = 0.1 | ridge on E[0,6] only (T_int < h rule) | LinReg1 | `Splice1_E0x11` (row from LinReg11) | ≈ LinReg1: the E[0,6] correction decorrelates in ~4 steps, so its lags carry little; its residual becomes slightly coloured and is drawn white. **No improvement** expected (partner for S1.3a) |
| S1.2b | LinReg^E, white, λ = 1 | as S1.2a | LinReg1 | `Splice1_E0x7` | as S1.2a; E[0,6]'s residual lag-1 0.44 (colour table) is lost by the white draw → its cells under-dispersed |
| S1.2c | LinReg^E, white, λ = 10 | as S1.2a | LinReg1 | `Splice1_E0x8` | as S1.2b, more so |
| S1.3a | LinReg^E + AR(2), λ = 0.1 | AR(2) on S1.2a | S1.2a | `Splice1_E0x11_ar2` | vs S1.2a: little to gain (little colour at λ = 0.1). vs LinReg1: level |
| S1.3b | LinReg^E + AR(2), λ = 1 | AR(2) on S1.2b | S1.2b | `Splice1_E0x7_ar2` | vs S1.2b: more calibrated cells, lower CRPS_0.5 (colour restored in E[0,6]). vs LinReg1: **the round's candidate** — CRPS_0.5 level with LinReg1, calibrated count higher → clause C |
| S1.3c | LinReg^E + AR(2), λ = 10 | AR(2) on S1.2c | S1.2c | `Splice1_E0x8_ar2` | vs S1.2c: as S1.3b. vs LinReg1: calibration up, but more skill lost in E[0,6]'s mean than at λ = 1 → C possible, S not |
| S1.4 | paper 3's rule + AR(2), base λ = 0.03 | ridge also on the other five QoIs (λ_i = λ(σ_i/σ_1)², most on the small scales) | S1.3a (λ = 0.1; E[0,6] gets 0.06) | `LinReg14_ar2` | vs S1.3a: CRPS_0.5 higher (ridge on the persistent small-scale corrections, where the lags carry the skill); calibration up in the bands it colours. vs LinReg1: **no improvement** on S; C uncertain |
| S1.5 | paper 3's rule + AR(2), base λ = 0.3 | as S1.4 | S1.3b (λ = 1; E[0,6] gets 0.62) | `LinReg16_ar2` | as S1.4, more skill lost |

Artefacts checked on the desktop 2026-10-06 (`meta_files/handoff_desktop_round1_2026-10-06.md`, Task A): all `:exact` or λ = 0, (400, 4000), h = 5; every AR(2) re-fitted by `lrs_ar_variant.jl --report` and equal to the file. They live on the desktop; copy them to Snellius before the runs.

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

## Settings and where they come from

- No offset and no noise scale fitted against coupled runs (D-10: none in round 1).
- Fixed by rule: E[0,6] alone (T_int < h on 1–10 TU, `analysis/tint_rule.jl`); AR(2) fitted to the
  residual ACF at lags 1–20 on 1–10 TU, Σ_ξ = DRD; the λ ladder 0.1 / 1 / 10, all reported.
- Pilot-chosen, disclosed (Sec. 3.8): LinReg7's λ = 1; paper 3's base 0.03 / 0.3; the AR order 2;
  the E[0,6] rule itself was formulated after a pilot scan.

## Post hoc
(none yet)
