# Results — M0 and DDN scored on HIT

**What this is.** Measurements for paper 4: TO+LRS (**M0**) and paper 1's data-driven noise model
(**DDN**) scored on one axis. Companion to `meta_files/plan.md` (design) and
`meta_files/metrics.md` (definitions); where a number here disagrees with either of those, this
file is the measurement and they are the prediction.

🔴 **REBASED ON THE NEW DATA, 2026-09-15. Read this before any number below.**
🆕 **D6 ran 2026-09-16 and §4c is the first paired measurement in this file.** It is the only
section whose comparisons are not confounded by realisation variance, and it **reverses §3's
ranking of the two LRS cells**. Where §3 and §4c disagree, §4c is the higher-powered experiment
(#61) — but see §7 for the validation that has not been run.


Every section is now computed on **P2r's rebaselined pipeline** — R1's tracking record, the
regenerated HF reference and R2's online ensembles — unless the section says otherwise. The
scoring driver takes the dataset as a switch and the two runs write different files:

```bash
julia --startup-file=no --project=analysis analysis/score_m0_ddn.jl                  # archive
RIKFLOW_DATASET=new julia --startup-file=no --project=analysis analysis/score_m0_ddn.jl   # primary
```

**What could not be rebased, and why — each is a structural limit, not an omission.**

| stays on paper 2's archive | why |
|---|---|
| §3's five-configuration sweep (LinReg1/63/64/73/74) | Only `LinReg1` has a rebaselined counterpart. The other four are h ∈ {10, 40} and λ = 0.01 cells that have never been run on the new system; each needs its own 5 × 100 TU ensemble. The rebaselined side sweeps **λ at fixed h = 5** instead (LinReg1/5/6/7/8/9/10), so the two sweeps vary different axes and their orderings are not comparable. |
| §3's **G1 online acceptance** | G1 *reproduces paper 2's published KS table*. There is nothing to reproduce on a different dynamical system, so it is not run on the new data rather than run and reported as a failure. |
| the 10 TU clamp census | No 10 TU rebaselined record exists and none is coming: R1 tracked for 100 TU precisely so one record carries both the fit window and D6's IC pool (#58). |

🔴 **The two datasets must never be pooled or compared number-for-number.** The rebaselined runs
are post-`09954be1`, which changed the Nyquist convention and therefore `∂`, `tau` and `dQ`. They
are a **different dynamical system**, not a better measurement of the same one. Where a comparison
is unavoidable it is made against each dataset's own **noise floor**, which is the only quantity
that transfers. ⚠️ That floor is itself a single unstable draw — it swings by a factor 2.7 with
where the record is cut (§3) — so it bounds a comparison rather than calibrating one.

**Reproduce it.**

```bash
cd lib/RikFlow      # from the fork root
julia --startup-file=no --project=analysis analysis/extract_archive.jl   # cache D3 and D5
julia --startup-file=no --project=analysis analysis/score_m0_ddn.jl      # -> output/paper4_scores.jld2
julia --startup-file=no --project=analysis analysis/plot_paper4.jl       # -> figures/*.png
julia --startup-file=no --project=test     test/runtests.jl              # 1461 tests
```

§4b, the rebaselined runs, is a separate pipeline on separate data and has its own two commands:

```bash
julia --startup-file=no --project=analysis analysis/extract_rebaseline.jl  # cache the R2 ensembles
julia --startup-file=no --project=analysis analysis/plot_rebaseline.jl     # -> figures/fig8_*.png
```

`plot_rebaseline.jl` prints §4b's three tables in the form §4b quotes them, so the section can be
diffed against a re-run rather than retyped.

**Data.** HIT, on the rebaselined pipeline: **R1** the 100 TU tracking record
(`data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3`, Float64, `freeze = 1`, OU seed 333),
**D3′** the regenerated 100 TU HF reference (40 001 points, 20.72 h, #54), **R2** the six online
closures, and **D8** a split of D3′ for the KS noise floor. The archive's counterparts keep their
old names (`D1`, `D3`, `D5`) and are used only where the table above says so; they are gitignored
and live outside the repository, reached through `RIKFLOW_ARCHIVE` and `RIKFLOW_DEV_ARCHIVE`.

🔴 **Which tracked record, and why it matters.** M0 here is fitted on **`D1`, the 100 TU record**
(`data_track2_dns512_les64_Re2000.0_tsim100.0`) — it has to be, because only that record has a
held-out 10–100 TU window. Paper 2 fitted `LinReg1` on a **10 TU** tracked record
(`6_online_TO_LRS.jl:19`), and the two records are two *realisations* of the same tracking
configuration: same initial field, same reference, so the level `q` agrees throughout to 1e-5–1e-4
of a standard deviation — but the **correction `dQ` decorrelates**, from 1e-5 of a `dQ` sd over the
first ~100 steps to **0.7–1.1 sd** beyond step 1000, i.e. across most of the (400, 4000) training
window. Refitting `LinReg1`'s configuration in Float64 over that window reproduces the archived
coefficients to **7.6e-5** from the 10 TU record and only to **0.282** from the 100 TU one
(predictions 8.3e-7 against 1.5e-4).

**So the M0 scored here is a different fit of the same configuration, not paper 2's model.** Every
regime-A number below is self-consistent (one fit, its own held-out window) and every regime-C
number uses the *archived* trajectories rather than this fit, so nothing is mixed — but a
coefficient-level quantity from §4 may not be compared with one derived from the archive, and
`test/test_g1.jl` reproduces the archive from the 10 TU record for exactly this reason.

**Partitions.** `plan.md` §7 — no time unit serves two roles. Training is paper 2's own
`train_range = (400, 4000)`, i.e. **1.00–10.00 TU**; the held-out window is **10.00–100.00 TU**,
35 994 verification instances. `Δt = 2.5×10⁻³`.

**What each metric is computed on — the regime decides, not consistency with the other regimes.**

| regime | scored on | why |
|---|---|---|
| **A** one-step, teacher-forced | **identical on `q` and `dQ`** | Given the history, `q*` is known and fixed at that step, so the two predictive laws differ by a shift common to the forecast and the truth. NLL, CRPS and the ranks are therefore the same number either way — verified to 4e-16 on CRPS and bit-identically on ranks (V26). Reported in `dQ` units because that is the conditional scale; the values are the level's too. |
| **B** offline unrolled, `q*` replayed | `dQ` only | The replayed `q*` pins the level, so `q` cannot separate models: 0.027–0.289 across nine models on `q` against 0.18–2.82 on `dQ` (O7). **Nothing in this report is regime B.** |
| **C** online, coupled | **`q`**, with `dQ` secondary | Nothing is replayed; `q*` comes from the solver. The level is free, and the level is what the physical claims are about — the long-term QoI distribution, the QoI decorrelation time, the QoI spread. Whether the correction's own marginal is right is not a claim anyone makes. It is also what paper 2 scores (`compute_ks.jl:48`), which is what makes G1's online acceptance checkable. |
| **0** fit-time | scaled q-space | Unavoidable: the design matrix is the history of `q`/`q*` and M0's regression target is the level. |

For M0 the transformation between the two is exact —

$$q \sim \mathcal N(\mu_s, \Sigma_s) \;\Longrightarrow\; dQ \sim \mathcal N\!\big(\mu_s\odot\sigma + \mu - q^{n*},\; D_\sigma \Sigma_s D_\sigma\big)$$

— same covariance, shifted mean, so nothing is lost in either direction.

⚠️ **A tension this exposes, measured rather than argued.** The level is the right *target* and a
poor *diagnostic* for temporal structure: across the five configurations Δρ₁ on the level spans
0.000–0.003 and Δρ₄₁ only 0.050–0.162, while on the correction Δρ₁ spans 0.022–5.29 and separates
λ = 0 from λ = 0.01 by two orders of magnitude — because ρ₁(q) ≈ 1, the level being far smoother
than the correction. So both are reported: the level because it carries the claim, the correction
because that is where a change in the model is visible.

---

## 1. The records

![HF reference and tracked LF QoIs](figures/fig1_trajectories_new.png)

The left column is the 100 TU overview with the training window in blue and the held-out window in
pink. The right column is the same data at full resolution across the boundary, and it exists
because the overview shows **one line, not two**: nudging holds the tracked low-fidelity QoIs on
the regenerated reference to

| `rms(track − ref) / sd(ref)` | Z[0,6] | E[0,6] | Z[7,15] | E[7,15] | Z[16,32] | E[16,32] |
|---|---|---|---|---|---|---|
| **R1, new** | 8.49e-5 | 7.57e-5 | 4.11e-5 | 3.29e-5 | 2.32e-3 | 1.46e-3 |
| D1, archive | 7.75e-5 | 7.73e-5 | 4.21e-5 | 3.24e-5 | 2.27e-3 | 1.44e-3 |

✅ **Tracking is as tight on the new system as on the old**, band for band, to within a few
percent of itself. Whatever the merge changed, it did not change how well the nudged LF run can be
held on its reference. So the tracked record *is* the reference for practical purposes, and the
interesting quantity is the correction that holds it there. In the right column the orange dashed
predictor `q*` separates visibly from the corrected `q` only in the two smallest-scale bands —
the statement that `dQ` is a 0.2–1.5 % correction on the level.

![The SGS correction dQ](figures/fig2_dQ_new.png)

### Timescales — both definitions, on **both** the correction and the level

🔴 Every archived document said `t_int = 0.04 TU`. Two separate conflations were hiding in that
number, and the second one still bites.

**First: two different statistics.** An exponential fit to the lag-1 autocorrelation,
`T_exp = -Δt / ln ρ₁`, and the integral `T_int = Δt (½ + Σₖ ρₖ)` taken to the first non-positive
`ρₖ` (Sokal's window). They are not interchangeable.

**Second, and load-bearing: two different series.** The **correction** `dQ` and the **level** `q`
have timescales that differ by a factor 3–60, and the project needs both — D6's forecast grid is
sized by the *level's* `T_int` (#58), while Δρ's lag below comes from the *correction's*. Quoting
one for the other mis-sizes either the grid or the temporal metric. Measured on the regenerated
reference:

| QoI | ρ₁(dQ) | T_exp(dQ) | T_int(dQ) | ρ₁(q) | T_exp(q) | T_int(q) | sd(dQ) |
|---|---|---|---|---|---|---|---|
| Z[0,6] | 0.9566 | 0.0564 | 0.1162 | 0.9998 | 15.91 | 0.2489 | 2.92 |
| E[0,6] | 0.7430 | 0.0084 | 0.0081 | 0.9992 | 3.00 | 0.4732 | 8.85e-3 |
| Z[7,15] | 0.9778 | 0.1113 | 0.0589 | 0.9999 | 30.20 | 0.4893 | 3.48 |
| E[7,15] | 0.9894 | 0.2336 | 0.0666 | 0.9999 | 26.73 | 0.4742 | 5.41e-4 |
| Z[16,32] | 0.9972 | 0.8816 | 0.3800 ⚠️ | 0.9999 | 34.37 | 0.5431 | 27.9 |
| E[16,32] | 0.9956 | 0.5614 | 0.3745 ⚠️ | 0.9999 | 32.89 | 0.5395 | 6.09e-4 |

All times in TU. ⚠️ marks a **truncated** integral: the ACF had not reached zero inside the
500-lag window, so that `T_int` is a **lower bound**, not a measurement. The level uses a 4000-lag
(10 TU) window and is not truncated. `correlation_time` now returns this flag instead of leaving
it to be assumed.

🔴 **`T_exp` is meaningless on the level and must never be quoted.** `ρ₁(q) ≈ 0.9999`, so
`-Δt / ln ρ₁` reads **3–34 TU** — longer than a third of the record. It measures the smoothness of
a nearly-integrated series, not a decorrelation time. On the level use `T_int`; on the correction
either, with the truncation caveat.

**Numbers to carry forward.**

- **Correction:** median `T_int` = **0.0914 TU**, so Δρ's lag below is **37 steps** (archive:
  0.102 TU, 41 steps). Still nothing like 0.04 TU, and the two definitions still disagree by up to
  a factor 4 within one QoI — E[7,15], 0.234 against 0.067.
- **Level:** `T_int` spans **0.249–0.543 TU**, median **0.474 TU**.

🔑 **The level decorrelates ~1.6× more slowly on the new reference than on the archive's** —
0.249–0.543 against 0.255–0.315 TU, medians 0.474 against 0.297. Not a detail: it is what raises
the KS noise floor in §3 — by about 1.7× on medians, which matches this 1.6× — and that floor
is what every regime-C number has to be read against.

⚠️ **This is a third estimate of the level's `T_int`, and it does not match memory #58.** That
entry quotes 0.94–1.08 TU on the new record from `report_marginals`' estimator, then argues down to
"0.5–0.63 TU stands for both" from the quarter-by-quarter spread. The Sokal-window estimator used
here gives 0.249–0.543 TU on the reference and 0.249, 0.473, 0.489, 0.474, 0.543, 0.539 on the
tracked record. **The two estimators disagree by about a factor 2**, and the ACF below says why:
the level's autocorrelation is not an exponential, so an integral-based timescale is not a
well-defined property of this series. D6's sizing table (#58) is built on the larger figure and is
therefore conservative — the safe direction — but see the lead-grid finding below, where being
conservative in that direction turns out to be expensive.

### The curves themselves — and what they say about D6's lead grid

![Autocorrelation of the level and the correction](figures/fig1b_acf.png)

`analysis/plot_acf.jl`. Left: the level `q` out to 10 TU, with D6's lead grid for that band drawn
in orange (the solid line is the 10× point that fixes `N_LEAD`). Right: the correction `dQ` out to
1 TU. The grey band is ±2 Bartlett standard errors under the null that the series is uncorrelated
past its 0.1 crossing — inside it an autocorrelation is not distinguishable from zero.

🔑 **The factor-2 estimator disagreement is explained, though not adjudicated.** The level's ACF is
**not a decaying exponential.** It falls steeply to ~0.1 within half a TU and then *rings*,
oscillating between roughly −0.2 and +0.2 with a period near 1 TU, all the way to the end of the
10 TU window. "The" decorrelation time is therefore not a well-defined property of this series, and
any estimator that integrates the ACF is integrating that ringing — which is precisely why a Sokal
window and `report_marginals`' estimator can differ by 2× without either being wrong. The robust
statistics are the crossings, and they are shorter than either `T_int`:

| reference, level `q` | Z[0,6] | E[0,6] | Z[7,15] | E[7,15] | Z[16,32] | E[16,32] |
|---|---|---|---|---|---|---|
| lag at ρ = 1/e | 0.290 | 0.302 | 0.333 | 0.325 | 0.355 | 0.352 |
| lag at ρ = 0.1 | 0.430 | 0.590 | 0.517 | 0.503 | 0.600 | 0.590 |
| `T_int`, Sokal | 0.249 | 0.473 | 0.489 | 0.474 | 0.543 | 0.539 |
| ±2 Bartlett se | 0.119 | 0.114 | 0.128 | 0.126 | 0.132 | 0.132 |

All in TU. The 1/e time is **0.29–0.36 TU on every band** — a factor 1.2–2.2 below `T_int` and a
factor 3 below memory #58's 0.94–1.08 TU. Note how tight the six are: on the level the 1/e times
span only **1.22×**, against the 2.18× that `T_int` spans and the 36.8× the correction spans. The
per-QoI lead grid is buying almost nothing on this series.

🔴 **D6's lead grid runs far past the point where the reference remembers its own state.** The grid
is `(0.25, 0.5, 1, 2, 5, 10) × T_int`; the reference's residual autocorrelation at each of those
leads is

| ρ(q) at lead | 0.25×T | 0.5×T | 1×T | 2×T | 5×T | 10×T |
|---|---|---|---|---|---|---|
| Z[0,6] | 0.941 | 0.811 | 0.470 | 0.031 | 0.147 | 0.054 |
| E[0,6] | 0.742 | 0.484 | 0.166 | 0.149 | 0.055 | 0.084 |
| Z[7,15] | 0.859 | 0.569 | 0.123 | 0.185 | 0.065 | 0.077 |
| E[7,15] | 0.862 | 0.574 | 0.124 | 0.170 | 0.062 | 0.074 |
| Z[16,32] | 0.852 | 0.542 | 0.124 | 0.212 | 0.061 | −0.038 |
| E[16,32] | 0.851 | 0.545 | 0.123 | 0.212 | 0.063 | −0.036 |

Read that against the ±2 se row above (0.114–0.132): **every lead from 2×T onward is inside the
band, on every band.** At 1×T only `Z[0,6]` (0.470) is clearly outside; the other five sit at
0.123–0.166, which is marginal. Past ~1×T_int ≈ 0.5 TU the truth at the lead is statistically
independent of the truth the forecast was initialised on, so *no* information carried from the
initial condition can help there. Whatever skill a model shows at 5× or 10× is its **climatology**
matching the reference's — which is exactly what regime C's free-running marginal KS already
measures, and what §3/#61 show is the underpowered statistic D6 was built to escape.

⚠️ **The long leads are not worthless, but they are over-sampled.** Spread at saturation is a real
diagnostic — does the ensemble variance converge to the climatological variance? — and the pilot
does answer it (ratio 0.55–1.3 at long leads). But that needs *one or two* anchor points, not half
the grid, and the run length is set by the largest lead.

✅ **The pilot's own numbers land where this predicts.** LinReg7's worst spread–skill departures
are at leads 380–434 = **0.95–1.09 TU ≈ 2×T_int** (ratio 2.1–2.5), and the fans in
`fig9b_d6_fans_by_ic.png` are fully open by about 2 TU past the warm-up. The informative region is
the short and medium leads; the tail is confirming climatology at full price.

✅ **Decision (Rik, 2026-09-16): `N_LEAD` cut to 1200 steps = 3.00 TU, and it is now set in
physical time from the ACF rather than as `10 × T_INT_MAX`.** `score_d6.jl` grew its own
`MULTIPLIERS = (0.25, 0.5, 1, 2, 5)`; the `10×` entry no longer fits and was dropped, and `5×` is
kept as the single saturation anchor so the spread–skill ratio still has a point past decorrelation.
The longest lead is now `5 × 0.5430 = 1086` of 1200 available steps, 26 distinct leads in the union
(was 32, longest 2172 of 2172).

| | was | is |
|---|---|---|
| `N_LEAD` | 2172 steps, 5.43 TU | **1200 steps, 3.00 TU** |
| `N_WARM` | 220 steps, 0.55 TU | **100 steps, 0.25 TU** |
| multipliers | `0.25, 0.5, 1, 2, 5, 10` | **`0.25, 0.5, 1, 2, 5`** |
| longest lead | 2172 (`Z[16,32]`) | **1086** |
| steps per member | 2392 | **1300** (1.84× shorter) |
| IC pool | `k ∈ [42, 377]`, 336 fields | **`k ∈ [42, 388]`, 347 fields** |
| spacing at K = 180 | 0.4679 TU (0.86 × `T_INT_MAX`) | **0.4832 TU (0.89 ×)** |

### The warm-up went with it: 220 → 100

🔑 **The warm-up replays recorded `dQ`, so while it lasts the run *is* the tracked record — and its
length therefore changes nothing but where the forecast starts.** Measured on the 50 LinReg7 pilot
members, mean absolute deviation from the reference in units of `sd(reference)`:

| column `c` | 1 | 51 | 101 | 221 |
|---|---|---|---|---|
| `Z[16,32]` | 0.0022 | 0.0022 | 0.0023 | 0.0020 |
| `E[16,32]` | 0.0014 | 0.0014 | 0.0015 | 0.0013 |
| the other four | 0.0000 | 0.0001 | 0.0000 | 0.0000 |

Flat across the whole warm-up, and equal to §1's tracking error (2.32e-3, 1.46e-3 on those two
bands). 🔴 **So the recorded rationale for 220 was wrong as written** — *"the slowest bands entered
every forecast still carrying the record's state rather than the model's"*. At the end of a warm-up
of any length the state is the record's, because that is what a replay is; no model output is used
during it. The only thing a warm-up must do is fill `q_hist`, and `hist_len = 5`.

✅ 100 is what the long online runs deploy (`6_online_TO_LRS.jl:108`, `dQ_data = data_track.dQ[:,
1:100]`), so D6 now scores the deployed configuration; ordinal 0's validation exercises the deployed
length instead of only the mechanism (#66's caveat is retired); and the IC pool gains two fields.
`N_WARM_DRIVER` stays a separate constant even though the two now coincide — if `N_WARM` moves
again, the validation must not move with it.

🔴 **The ordinal → `k` map moved.** A shorter forecast and a shorter warm-up free 11 more fields at
the end of the record, so `select_ics` re-spaced: ordinals 1–4 are still `k = 42, 44, 46, 48`,
ordinal 5 is now `k = 50` (was 49), and later ordinals shift. All 180 packages were rebuilt; those no
longer selected were moved to `analysis/output/d6_ics_stale/` rather than deleted.

⚠️ **The LinReg7 pilot on disk stays readable but is no longer paired.** A 2172-lead run contains
every lead of a 1200-lead grid, so `score_d6.jl` scores those 60 files unchanged. Their ICs are the
old selection, so they cannot be compared member-for-member against anything produced after the
change.


⚠️ **`sd(dQ)` spans four orders of magnitude.** Anything pooling QoIs in raw units is an enstrophy
statistic with the energy bands contributing nothing. Every pooled number below is normalised.

---

## 2. Density and calibration — regime A (one-step, teacher-forced)

Every number in this section is **simultaneously a `q` score and a `dQ` score**: given the history
the two predictive laws differ by a shift common to the forecast and the truth, so the NLL, the
CRPS and the ranks are invariant. Verified — CRPS to 4.4e-16, ranks bit-identical, NLL to
summation order. Units are `dQ`'s because that is the conditional scale of the thing being
predicted; the *conclusions* are statements about the level.

### #1 Held-out negative log-likelihood

$$\text{NLL} = \frac{1}{2N}\sum_n\Big[\log\det\Sigma_n + (y_n-\mu_n)^\top\Sigma_n^{-1}(y_n-\mu_n) + N_Q\log 2\pi\Big]$$

| | in-sample | held-out |
|---|---|---|
| **M0** | −21.898 | **−21.119** |
| **DDN** | — | **−8.282** |

Per sample, so lengths compare. M0 beats DDN by **12.8 nats**. That is the expected direction and
it is worth almost nothing on its own — NLL is unbounded, and the gap is dominated by the energy
bands whose `log det Σ` is large and negative.

### #2 CRPS

$$\text{CRPS}(F,y) = \int\big(F(x) - \mathbf 1\{x\ge y\}\big)^2dx, \qquad
\text{fair ensemble form: } \frac1M\sum_i|x_i-y| - \frac{1}{2M(M-1)}\sum_i\sum_j|x_i-x_j|$$

The fair denominator $2M(M-1)$ is used throughout. The common $2M^2$ is biased at small `M` and
rewards under-dispersion — the exact defect under investigation.

Reported as **CRPS / sd(dQ_ref)**, dimensionless. ⚠️ This denominator is the one genuine choice
the `q`/`dQ` equivalence leaves open: dividing instead by `sd(q_ref)` would give a skill score
against the QoI *climatology*, which is a different and much less demanding comparison — the
conditional spread of `q` given the history is `sd(dQ)`, not `sd(q)`, so the conditional scale is
the informative one.

| QoI | M0 | DDN |
|---|---|---|
| Z[0,6] | 0.142 | 0.561 |
| E[0,6] | 0.362 | 0.560 |
| Z[7,15] | 0.050 | 0.555 |
| E[7,15] | 0.029 | 0.562 |
| Z[16,32] | 0.029 | 0.577 |
| E[16,32] | 0.032 | 0.579 |
| **mean** | **0.107** | **0.566** |

🔑 **DDN lands on $1/\sqrt\pi = 0.5642$ in every band.** That is the CRPS of a climatological
Gaussian forecast, analytically. DDN is therefore not merely *like* the climatological forecast for
`dQ` — it *is* it, to within 3 % in all six bands. The negative control is confirmed from the score
rather than from the argument, and E[0,6] is where M0 comes closest to being merely climatological
(0.362 against 0.560).

### #4 Rank histogram — RH-1, the paper's problem statement

$$r_n = 1 + \#\{i: x_i^{(n)} < y_n\}, \qquad
Z_{\text{slope}},\,Z_{\text{cvx}} = \sqrt{\tfrac{N_{\text{eff}}}{N}}\cdot\frac{\sum_k a_k n_k}{\sqrt{e\sum_k a_k^2}}$$

with linear weights $a_k \propto k - \tfrac{K+1}{2}$ for the slope (bias) and quadratic weights
$a_k \propto (k-\tfrac{K+1}{2})^2 - \tfrac{K^2-1}{12}$ for the convexity (dispersion). Positive
convexity = **U** = under-dispersed; negative = ∩ = over-dispersed. `M = 100`, ties broken at
random with a recorded RNG, 95 % intervals from a moving-block bootstrap.

![RH-1](figures/fig3_rank_histograms_new.png)

🔑 **The in-sample histogram is the null, and flatness is not.** On the training window M0's
residual mean is 1e-4 standard deviations and its residual standard deviation matches the fitted
Σ to **1.0001** — the fit is exact there by construction. The histogram is nonetheless ∩-shaped,
convexity −3.7 to −8.7, because the residual is **leptokurtic**: excess kurtosis +0.94 to +1.95,
so it is more sharply peaked than the Gaussian fitted to it and the truth lands in the middle bins
more often than a Gaussian truth would. A simulated control reproduces this with the mean and the
variance both exact by construction. **Reading a held-out convexity against zero would therefore
attribute a pure shape effect to dispersion** — flattering the model in the high bands and
condemning it in the low ones.

| QoI | ex. kurt | sd(res)/sd(fit) in | out | cvx in-sample | cvx held-out | **Δ vs null** |
|---|---|---|---|---|---|---|
| Z[0,6] | +0.94 | 1.0001 | 0.988 | −3.70 | −8.40 [−10.2, −6.2] | **−4.70** |
| E[0,6] | +1.00 | 1.0001 | 0.993 | −4.23 | −7.76 [−9.9, −5.9] | **−3.53** |
| Z[7,15] | +1.95 | 1.0001 | **1.179** | −8.66 | +6.01 [+3.5, +8.2] | **+14.67** |
| E[7,15] | +1.80 | 1.0001 | **1.143** | −7.97 | +2.42 [+0.2, +4.6] | **+10.39** |
| Z[16,32] | +1.39 | 1.0001 | **1.172** | −5.52 | +10.20 [+8.0, +12.2] | **+15.72** |
| E[16,32] | +1.44 | 1.0001 | **1.185** | −6.27 | +12.38 [+10.3, +14.6] | **+18.65** |

**The answer is band-dependent, and it is confirmed twice over.** In the four smaller-scale bands
the held-out residual is **14–18 % wider than the fitted Σ**, the convexity moves by +10 to +19
against the null, and every interval excludes zero: **M0 is under-dispersed there**. In the two
largest-scale bands the variance is right to 1 % and the convexity moves the other way, while the
slope reaches +9.1 and +7.5: **M0 is biased, not mis-dispersed, in the low bands**.

DDN's convexity is +0.8 to +2.3 in five bands — nearly flat, as a climatological ensemble must be —
with a slope of −2.6 to −2.8 in the smallest scales where its constant mean does not match the
held-out mean.

⚠️ **The $N_{\text{eff}}$ correction turns out to be a no-op here, and that is worth recording.**
`plan.md` and `metrics.md` both insist on it, correctly in principle. Measured: the *rank* series
has $N_{\text{eff}}$ of 34 300–36 000 against `N = 35 994`, and a bootstrap block length of 1–2
steps. The QoIs are strongly serially correlated ($\rho_1 \approx 0.94$–0.998) but the ranks are
nearly white, because the rank depends on where truth falls relative to the model's own spread and
the model tracks the QoI. So the raw and corrected contrasts agree to about 1 % on this data. The
machinery is still needed — on a synthetic AR(1) null the uncorrected contrast has a standard
deviation of 4.9 instead of 1 — but it does not change any number in this table.

### Flat is not skilful

![Dynamics against calibration](figures/fig4_dynamics_vs_calibration_new.png)

The lag-1 autocorrelation of the predicted mean `dQ`: M0 gives 0.735–0.999, tracking the realised
values; **DDN gives exactly 0.0000 in every band**, because its predictive mean is a constant. Set
beside its near-flat histograms and its climatological CRPS, that is the trap `metrics.md` states
abstractly — **flat histogram, good marginal score, no dynamics at all** — demonstrated without
running a single large-λ cell.

---

## 3. Marginal and temporal accuracy — regime C (online, coupled)

Scored on the **QoI level** against the regenerated HF reference, with the correction alongside.
Ten closures, all launched from the same initial field as the tracking record (rel diff ≤ 1.2e-16).

![Regime C](figures/fig6_online_new.png)

| closure | h | λ | replicas | stable | summed KS `q` | ens KS | Δρ₁(q) | Δρ₃₇(q) | summed KS `dQ` | Δρ₁(dQ) | clamp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **LinReg1** | 5 | 0 | 5 | 5/5 | 0.586–1.010 | 0.836 | 0.000 | 0.146 | 0.352–0.545 | 0.028 | 1.84% |
| **LinReg5** | 5 | 1e-5 | 5 | **4/5** | 0.692–0.988 | 0.858 | 0.000 | 0.169 | 0.354–0.497 | 0.030 | 1.25% |
| **LinReg6** | 5 | 1e-4 | 5 | 5/5 | 0.554–0.945 | 0.810 | 0.000 | 0.161 | 0.322–0.504 | 0.037 | 1.18% |
| **LinReg7** | 5 | 1 | 5 | 5/5 | 0.537–0.913 | **0.648** | 0.000 | **0.049** | 0.350–0.514 | 2.740 | **0.00%** |
| **LinReg8** | 5 | 10 | 5 | 5/5 | 0.716–0.823 | 0.740 | 0.002 | 0.055 | 0.622–0.711 | **4.922** | **0.00%** |
| **LinReg9** | 5 | 100 | 5 | 5/5 | 1.221–1.375 | 1.297 | 0.018 | 0.126 | 1.154–1.212 | **6.666** | **0.00%** |
| **LinReg10** | 5 | 1e4 | 5 | 5/5 | 2.245–2.295 | 2.270 | **3.119** | **3.972** | 2.242–2.251 | **8.602** | **0.00%** |
| **DDN** | — | — | 5 | 5/5 | 1.285–1.382 | 1.332 | 0.001 | 0.201 | 0.291–0.306 | **5.658** | n/a |
| no model | — | — | 1 | 1/1 | — | 2.487 | 0.000 | 0.116 | none | none | n/a |
| Smagorinsky `c_s=0.07` | — | — | 1 | 1/1 | — | 3.710 | 0.001 | 0.322 | none | none | n/a |

⚠️ **`LinReg2` (λ = 1e-2) is fitted but not yet run** — the one remaining hole in the ladder, and it
sits in the two-decade gap between 1e-4 and 1. `plot_rebaseline.jl` and `score_m0_ddn.jl` skip it by
name until its ensemble lands (`rebase_available`); nothing below interpolates across the gap.

🔴 **THE NOISE FLOOR, AND WHY IT IS NOT A THRESHOLD.**

`ks_noise_floor` (`src/ts_score.jl:671`) splits the HF reference in half at 50 TU, takes the
two-sample KS distance between the halves per QoI, and sums the six. Both halves are the same
system, so whatever they show is the record's own sampling noise. Contiguous halves, not resampled,
because the record is serially correlated.

| | archive | rebaselined |
|---|---|---|
| floor at the 50 TU cut | **0.1848** | **0.8179** |
| floor across cut points, 25–75 TU | 0.166–0.429 | 0.328–0.889 |
| **median over cut points** | **0.281** | **0.469** |

🔴 **It is ONE DRAW and it is very unstable — do not quote the single number as a threshold.**
Moving the cut in 5 TU steps swings the archive's floor over 0.166–0.429 and the new record's over
0.328–0.889. The canonical 50 TU cut happens to land near the **bottom** of the archive's range and
near the **top** of the new record's, which inflates any archive-vs-new ratio built from the two
canonical values (4.43×) against the ratio of medians (**1.67×**).

✅ **The direction is real and the magnitude was not.** 1.67× on medians agrees with the
independently measured ~1.6× slowdown in the level's decorrelation time (§1), which is the
mechanism: slower decorrelation ⇒ fewer independent samples in 100 TU ⇒ the halves differ more.

⚠️ **It is also not sample-size matched to what it is being compared against.** KS grows as samples
shrink — median over all disjoint pairs:

| split | points per part | archive | rebaselined |
|---|---|---|---|
| 2-way | 20 000 | 0.185 | 0.818 |
| 4-way | 10 000 | 0.526 | 0.862 |
| 8-way | 5 000 | 0.640 | 1.262 |

The floor compares 20 000 points against 20 000. A per-replica summed KS compares 40 001 against
40 001, and the ensemble form pools 5 × 40 001 against 40 001. **So the floor is a yardstick with
the wrong units, not a calibrated threshold**, and a "ratio to floor" carries both the cut-point
lottery and a sample-size mismatch.

🔴 **What this means for LinReg1's 0.836 against the archive's 0.198 — the question is STILL OPEN.**
An earlier version of this section claimed the two were each at their own floor (ratios 1.07 and
1.02) and that the ~4× gap was therefore explained by the floor moving. **That claim rested on the
single 50 TU cut and does not survive the cut-point sweep.** Against the *median* floor the archive
run sits at **0.70×** — below its own floor — and the rebaselined run at **1.78×** — above its own.
The floor moved in the right direction but nowhere near far enough to absorb the gap.

🔑 **The fix is a proper null, and one already exists in this repository.**
`analysis/plot_hf_new_vs_archive.jl` uses a **block permutation** null for exactly this comparison
(#54: per-band 95th percentile 0.094–0.108), which returns a distribution rather than one number and
respects the serial correlation. Regime C should use it, sampled at the sample sizes actually being
compared. Until then, quote the floor as a **range with its construction stated**, never as a single
number, and treat every "ratio to floor" in this report as indicative.

⚠️ **What still stands regardless of the floor.** The DDN (1.332), no model (2.487) and Smagorinsky
(3.710) are above even the largest floor draw on this record (0.889), so the ordering
**LRS < DDN < no model < Smagorinsky** is safe. So is the statement that the five cells at
λ ≤ 10 (0.648–0.858) cannot be separated from each other **on this statistic**: their spread is far
smaller than the floor's own swing, and their per-replica ranges overlap heavily (λ = 1's
0.537–0.913 and λ = 10's 0.716–0.823 both sit inside λ = 0's 0.586–1.010). The two strong cells are
a different matter: `LinReg9` at 1.297 and `LinReg10` at 2.270 clear the largest floor draw, so
**λ ≥ 100 is measurably worse** and that ordering does not depend on the floor's construction.

🔑 **So the λ = 1 and λ = 10 improvements must be argued from statistics that do not overlap**, and
three are available, none of them a distributional distance with a floor problem: the clamp fires
**0 times in 5 × 40 000 steps** against 160–736 at λ = 0; the minimum of `Z[16,32]` rises from
**181** to 380 (λ = 1) and 587 (λ = 10) against a reference minimum of 721; and the climatological
spread–skill ratio moves from 1.507, outside S7's band, to **0.982**, inside it. See the ladder
below.

### The weak λ probe — *small* regularization does not fix the excursions 🆕 2026-09-15

⚠️ **Scope, added after the strong ladder ran.** Everything in this subsection is measured at
λ ≤ 1e-4 and its negative conclusion holds only there. At λ = 1 the excursions *do* go away — see
the ladder below, which is the same experiment continued four decades further and reaches the
opposite answer. The two are kept apart because the weak probe is what motivated the strong one,
and because "regularization does not help" is exactly the conclusion this report would have shipped
had the sweep stopped at 1e-4.

`LinReg5` (λ = 1e-5) and `LinReg6` (λ = 1e-4) were fitted and run to test whether the deployed
λ = 0 model's downward excursions come from its unresolved coefficient vector (§4). They do not.

| | λ = 0 | λ = 1e-5 | λ = 1e-4 |
|---|---|---|---|
| ρ(C̃) | 2.6888 | 1.0031 | 1.0033 |
| starred gain | 269.80 | 8.01 | 4.10 |
| ensemble KS | 0.836 | 0.858 | 0.810 |
| mean ratio, `E[16,32]` | 0.913 | 0.920 | 0.925 |
| clamp rate, worst replica | 1.84% | 1.25% | 1.18% |
| stable replicas | 5/5 | **4/5** | 5/5 |

🔴 **The fit-time diagnostics move by a factor 30–65 and the deployed behaviour barely moves.**
λ = 1e-5 collapses ρ(C̃) from 2.69 to 1.003 and the starred gain from 270 to 8 — reproducing paper
2's archived operator norms almost exactly — yet the summed KS is unchanged within the floor, the
low bias improves by less than 1.5 points, and the clamp still fires on over 1% of steps.

🔴 **λ = 1e-5 lost a replica.** `LinReg5` replica 5 diverged at **t = 37.81 TU** with `Z[0,6]`
reaching **3.0e7** against a reference median near 2000. One replica of five is not evidence that
regularization *hurts* stability, but it is decisive against the hypothesis that it *cures* the
excursions. Excluded from every number above; see the guard in `extract_rebaseline.jl`.

🔑 **What this rules out.** ρ(C̃) > 1 on the standalone recursion is **not** what produces the
excursions — the λ cells have ρ ≈ 1.003 and excurse just as much. Combined with the step-level
attribution (§4b), which finds `q*` already below the reference minimum on 96.6–99.5% of excursion
steps, the excursions are a property of the **coupled** LF system rather than of the model's
open-loop spectrum. The remaining candidates are the changed `tau` (#46) and the reference
realisation itself; the clamp is ruled out too, since it fires at a similar rate in all three.

### The strong λ ladder — λ = 1 is the best closure this project has run 🆕 2026-09-15

The weak probe moved ρ(C̃) by a factor 2.7 and changed nothing deployed, which said the interesting
range was much higher. `LinReg7/8/9/10` (λ = 1, 10, 100, 1e4) continue the same sweep across the
point where the standalone operator becomes **contractive** (ρ crosses 1 between λ = 1 and λ = 10;
`4_setup_search.jl` carries the fit-time table).

| | λ = 0 | λ = 1e-4 | **λ = 1** | λ = 10 | λ = 100 | λ = 1e4 |
|---|---|---|---|---|---|---|
| ρ(C̃) | 2.6888 | 1.0033 | **1.0057** | 0.9992 | 0.9930 | 0.9793 |
| train RMSE | 0.00669 | 0.00671 | **0.00943** | 0.01659 | 0.02946 | 0.14909 |
| ensemble KS, `q` | 0.836 | 0.810 | **0.648** | 0.740 | 1.297 | 2.270 |
| Δρ₃₇(q) | 0.146 | 0.161 | **0.049** | 0.055 | 0.126 | 3.972 |
| mean ratio, `E[16,32]` | 0.913 | 0.925 | **0.940** | 0.954 | 0.973 | 1.018 |
| min `Z[16,32]` (ref: 721) | 181 | 204 | **380** | 587 | 1244 | 1777 |
| max `Z[16,32]` (ref: 3791) | 4652 | 4074 | **3700** | 3451 | 2783 | 2417 |
| clamp rate, worst replica | 1.84% | 1.18% | **0.00%** | 0.00% | 0.00% | 0.00% |
| spread–skill on `q` | 1.507 | 1.439 | **0.982** | 0.759 | 0.486 | 0.166 |
| ensemble KS, `dQ` | 0.435 | 0.426 | **0.402** | 0.665 | 1.188 | 2.243 |
| Δρ₁(dQ) | **0.028** | **0.037** | 2.740 | 4.922 | 6.666 | 8.602 |

🔑 **λ = 1 is the best cell this project has run, and it is best on the correction too.** Summed KS
**0.648** against 0.836 at λ = 0 — the lowest in the report — on per-band KS of 0.087–0.118, the
lowest of any closure in five of the six bands. ⚠️ The exception is `E[0,6]`, where it ties the DDN
at 0.0869 and **no model scores 0.0279**, a third of it: the largest scales are the one place an
uncorrected LF run is already close to the reference, so no closure earns credit there. The lag-37
autocorrelation error falls 3×; the 6–9% low
bias narrows to 4–6%; the stabilizer **never fires once in 5 × 40 000 steps**; and the
climatological spread–skill ratio lands at **0.982**, the only LRS cell inside S7's [0.8, 1.25]
band. `dQ`'s ensemble KS **also improves**, 0.435 → 0.402.

🔴 **The level-versus-correction tension is real but it begins at λ = 10, not at λ = 1.** From
λ = 10 upward the two move in opposite directions: the level keeps improving to 0.740 while `dQ`'s
KS degrades to 0.665 and its lag-1 autocorrelation error reaches **4.922**, a 176× degradation of
the quantity the model is actually fitted to emit. λ = 1 is the cell where nothing has to be traded,
which is why it, and not λ = 10, is the one to deploy. The O7 warning still stands for the rest of
the ladder: past λ = 1, a λ chosen by how well `dQ` is reproduced picks the wrong cell.

⚠️ **Δρ₁(dQ) is already 2.740 at λ = 1**, against 0.028 at λ = 0 — so the correction's *temporal*
structure is degraded even in the best cell, while its marginal is not. The ridge shrinks toward
"predict the climatological mean level", which makes the correction smoother and more persistent
and injects less variance into the LF system. That is a plausible mechanism for the excursions
disappearing — less forcing, less chance of driving a band to zero — and it is **not** the same
claim as "contractivity fixed it".

🔴 **ρ(C̃) < 1 is not what is doing the work, and the ladder now says so twice.** The best cell,
λ = 1, has ρ = **1.0057**, i.e. *not* contractive — so crossing 1 is not even necessary for the
optimum. And `LinReg10`, the most contractive cell fitted (ρ = 0.9793), is the second-worst closure
in the whole table (2.270, beaten only by Smagorinsky): its level autocorrelation is broken outright
(Δρ₁(q) = 3.119 against ≤ 0.018 everywhere else) and its range has collapsed to 1777..2417 on
`Z[16,32]` inside a reference spanning 721..3791. It is under-dispersed, not accurate. ρ is neither
necessary nor sufficient here, and the earlier statement that it does not predict deployed
behaviour survives this ladder intact.

⚠️ **The optimum is bracketed but still not located.** λ = 1e-2 (`LinReg2`) is fitted and unrun, so
the measured curve jumps 1e-4 → 1 and the minimum could sit anywhere in those two decades. Nothing
here claims λ = 1 is *the* optimum, only that it is the best of what has been run. Running
`LinReg2` (`./batch_scripts/submit_lrs.sh 2`) closes the last gap.

### #10 / #11 Summed and ensemble KS, and G1's online acceptance — **ON THE ARCHIVE**

🔴 **This subsection alone stays on paper 2's data, and must.** G1's acceptance is defined as
*reproducing paper 2's published KS table*; there is nothing to reproduce on a different dynamical
system. It is also the only place a configuration **sweep** exists — h ∈ {5, 10, 40} and
λ ∈ {0, 0.01} — because the other four cells have never been run on the rebaselined system. Numbers
here are archive-vs-archive and **must not be compared with the table above**; each dataset has
its own floor and both floors are unstable draws (see the rebase table in the header).

The archive figure is `figures/fig6_online.png` (no `_new` suffix).

$$\text{KS}_r = \sum_{i=1}^{N_Q}\sup_x\big|F_i^{(r)}(x) - F_i^{\text{ref}}(x)\big|$$

summed over QoIs, **never over replicas**; the ensemble form pools replicas into one EDF first.
Both are reported and never averaged together.

🔑 **G1's online acceptance passes, exactly.** `plan.md` §12 asks for the summed ensemble KS per
configuration to fall within paper 2 Fig. 6's replica min–max range. Read from the archived
`ks_dists_*.jld2` tables and compared against this round's level-based numbers:

| config | h | λ | this round | paper 2's archived table | archived ensemble |
|---|---|---|---|---|---|
| LinReg1 | 5 | 0 | 0.142–0.371 | 0.142–0.371 | 0.198 |
| LinReg63 | 10 | 0.01 | 0.400–0.580 | 0.400–0.580 | 0.466 |
| LinReg64 | 10 | 0 | 0.213–0.392 | 0.213–0.392 | 0.197 |
| LinReg73 | 40 | 0.01 | 0.472–0.694 | 0.472–0.694 | 0.603 |
| LinReg74 | 40 | 0 | 0.281–0.436 | 0.281–0.436 | 0.250 |

Every range reproduces to the digit and the ensemble values match too — an independent confirmation
of the KS implementation, the QoI extraction and the replica bookkeeping in one. It is only
checkable because regime C is scored on the level; a `dQ`-based score has nothing to compare
against.

⚠️ Two caveats belonging to the archive rather than to the comparison. `compute_ks.jl` decides
which replicas are stable by testing `data_online_tsim100.0_replica<i>.jld2` with `isfile` and then
loads `..._rand_initial_dQ.jld2` — a different family — so the published table pairs a stability
check on one set of runs with distances computed on another. And the tables are split by
configuration index across several files, so absence from one file is not absence from the archive.

🔑 **The noise floor changes the reading.** A reference-vs-reference split of D3 (D8) gives a summed
KS of **0.1848 on the level** (0.1328 on the correction). LinReg1's replicas span 0.142–0.371, so
**the best replicas of the best configuration sit at or below the floor**, and LinReg1 against
LinReg64 (0.213–0.392) is not a separation this statistic can support. The λ = 0.01 and h = 40
configurations are clearly above it.

### #12–#15 Autocorrelation error

$$\Delta\rho(\tau) = \sum_i\big|\rho_i^{\text{model}}(\tau) - \rho_i^{\text{ref}}(\tau)\big|,
\qquad \Delta\rho_{\text{int}} = \sum_i\int_0^T\big|\rho_i^{\text{mod}} - \rho_i^{\text{ref}}\big|\,d\tau$$

![Autocorrelation](figures/fig7_autocorr_new.png)

Where KS asks whether the values are right, Δρ asks whether the order is right.

🔴 **On the level, Δρ₁ has no dynamic range at all: 0.000–0.003 across five configurations that
differ by a factor 5 on KS.** ρ₁(q) ≈ 1 because the level inherits the QoI's own smoothness, so a
one-step difference is invisible. Even Δρ₄₁, at the measured integral timescale, spans only
0.050–0.162. This is `plan.md` §8b's argument for preferring the integral-timescale lag, and the
measurement says the effect is stronger than §8b assumed: on the level, lag 1 is not a weak
statistic but a null one.

**On the correction the same lag is the sharpest discriminator available** — 0.022–0.026 at λ = 0
against 3.46–5.29 at λ = 0.01 — because it isolates exactly the loss of short-lag dynamics that
regularization causes. The ordering inverts at the longer lag (0.050–0.162 on the level,
0.288–0.968 on the correction), since by 41 steps both model and reference have decayed.

**So the two answer different questions and both belong in the report.** The level is where "the
decorrelation time must be right" is a claim; the correction is where a change in the model shows
up. Quoting only the level would make all five configurations look equally good on temporal
structure, which the KS column shows they are not.

### #16 Stability fraction, and what ρ(C̃) does not tell you

**On the rebaselined runs: 5/5 replicas complete 100 TU for LinReg1, LinReg6, LinReg7, LinReg8,
LinReg9, LinReg10, the DDN and both deterministic baselines — and 4/5 for LinReg5.** Nine of the ten
closures are at 1.00; the single failure among the 42 launched members (7 LRS cells × 5, the DDN's
5, and one each for no-model and Smagorinsky) is at λ = 1e-5.

🔴 **The one failure is at λ = 1e-5, not at λ = 0.** `LinReg5` replica 5 diverged at **t = 37.81 TU**
with `Z[0,6]` reaching **3.0e7** against a reference median near 2000. It is excluded from every
score; `extract_rebaseline.jl` now refuses any replica shorter than `EXPECTED_STEPS` rather than
truncating it into the ensemble, because `dQ` is *preallocated* and a short run's tail is zeros that
a clamp census would otherwise read as 24 883 firings that never happened.

🔑 **Set that against ρ(C̃) = 2.6888 for the deployed h = 5, λ = 0 cell (§4).** The standalone QoI
process has a fastest-growing mode nearly tripling every step, and the coupled system runs 100 TU
five times out of five without a failure — while the cell with ρ = 1.0031 is the one that lost a
replica. `plan.md` §8a's caveat — *ρ(C̃) < 1 is neither necessary nor sufficient for coupled
stability* — is no longer a caveat but a measurement, and now with the sign of the association
pointing the wrong way for the naive reading.

⚠️ **One failure in five is not evidence that λ hurts stability.** With M = 5 the difference between
5/5 and 4/5 is one draw. What it does rule out is the claim that λ *cures* the instability, which is
what the probe was run to test.

### #18 Climatological spread–skill

$$\text{ratio} = \sqrt{\tfrac{M+1}{M}}\cdot\frac{\text{spread}}{\text{skill}}$$

| config | ratio on `q` | ratio on `dQ` |
|---|---|---|
| LinReg1 | 1.188 | 1.187 |
| LinReg63 | 0.965 | 0.961 |
| LinReg64 | **1.250** | 1.254 |
| LinReg73 | **0.809** | 0.997 |
| LinReg74 | 1.052 | 1.052 |

**The two agree to 0.004 in four of five configurations and disagree sharply in the fifth.**
LinReg73 (h = 40, λ = 0.01) reads 0.997 on the correction and **0.809** on the level: the
correction's spread happens to match the correction's error while the level's does not. This is the
clearest single case for scoring the level — on `dQ` this configuration looks perfectly dispersed,
and on the quantity the criterion is actually about it is the worst of the five.

🔑 **Both edges of S7's [0.8, 1.25] band are touched, by opposite failures**: LinReg73 at 0.809 is
over-confident and LinReg64 at 1.250 over-dispersed. Neither is outside, but neither has margin,
and the band's own width is doing the work.

⚠️ **The finite-M correction matters more than either document says.** Both quote 0.953 for M = 10.
At the archive's **M = 5** an uncorrected perfect ensemble reads $\sqrt{5/6} = \mathbf{0.913}$,
which eats a third of the lower margin — enough to move LinReg73 from inside to outside on its own.
The correction is applied above. Labelled climatological and not a lead: these replicas have fully
decorrelated from their common start, so this tests long-run variance, close to what KS already
measures.

#### On the rebaselined ladder 🆕 2026-09-15

| config | λ | ratio on `q` | ratio on `dQ` | inside S7's [0.8, 1.25]? |
|---|---|---|---|---|
| LinReg1 | 0 | **1.507** | 1.488 | no — over-dispersed |
| LinReg5 | 1e-5 | **1.368** | 1.387 | no — over-dispersed |
| LinReg6 | 1e-4 | **1.439** | 1.451 | no — over-dispersed |
| LinReg7 | 1 | **0.982** | 0.966 | **yes — and nearly perfect** |
| LinReg8 | 10 | **0.759** | 0.886 | no — marginally over-confident |
| LinReg9 | 100 | **0.486** | 0.856 | no — over-confident |
| LinReg10 | 1e4 | **0.166** | 2.067 | no — badly over-confident |
| DDN | — | 0.979 | 1.004 | **yes** |

🔑 **λ carries the ratio monotonically down through the band, and λ = 1 lands in it.** 1.507 at
λ = 0 down to 0.166 at λ = 1e4, crossing 1 between 1e-4 and 1. `LinReg7` reads **0.982** on the
level and 0.966 on the correction — the best-dispersed LRS cell this project has measured, and
within 2% of perfect on both. λ = 0's 1.507 is outside the band on the over-dispersed side and
λ = 10's 0.759 just outside on the over-confident side, so the acceptable window in λ is narrow and
λ = 1 sits in the middle of it.

✅ **Dispersion and KS agree for once, and they agree on λ = 1.** `LinReg7` has the best summed KS
in the report (0.648) *and* a spread–skill ratio inside the band (0.982). That is worth stating
explicitly because these two criteria disagree everywhere else here — the DDN is inside the band at
0.979 while scoring 1.332 on KS, and λ = 0 scores 0.836 on KS while sitting at 1.507 on dispersion.

⚠️ **On dispersion alone `LinReg7` and the DDN are indistinguishable** — 0.982 against 0.979, a
0.003 gap on a statistic whose finite-M correction is itself 0.087 at M = 5. The separation between
those two closures comes from KS (0.648 against 1.332), not from this criterion; do not report
λ = 1 as "better dispersed than the DDN".

⚠️ **`LinReg10` is the one place the `q`/`dQ` agreement breaks, and it breaks hard** — 0.166 on the
level against 2.067 on the correction, a factor 12. Everywhere else the two agree to within 0.13.
Read on the correction alone λ = 1e4 looks over-dispersed; on the level it is the most
over-confident ensemble in the report. Same lesson as LinReg73 above, an order of magnitude louder.

## 4. Mechanism diagnostics — regime 0 (fit-time)

### 🔴 A precision result that gates three of the metrics below — and it got larger on the new data

R1's record is Float64, but the *design* is what decides resolution, and it has
`κ(X) = 1.85e6` (harmonized, `:normal`) to `1.02e7` (faithful, `:standardise`), so
`κ·ε₃₂ = 0.22` to `1.22`: **fitted in single precision the coefficient vector is not resolved at
all.** Paper 2's archive was fitted that way.

The clean way to see it: at λ = 0 the two normalization conventions differ only by subtracting a
constant from every design column and from the target, and a least-squares fit carrying an
intercept is invariant to exactly that, so their slope blocks *must* agree.

| | Float32 | Float64 |
|---|---|---|
| slope-block relative difference between conventions | **2.708** | **1.40e-10** |
| slope-block difference, Float32 vs Float64 | — | **0.972** |
| ρ(C̃) | 1.0031 | **2.6888** |
| H∞ starred-block gain | 14.08 | **269.80** |
| total block sum ‖S − I‖ | 2.805e-2 | 2.910e-2 |

🔴 **On the new record the Float32/Float64 gap is bigger than it was on the archive** — ρ(C̃) 1.0031
against 2.6888 here, 1.012 against 2.174 there; the starred gain 14 against **270** here, 25 against
109 there. Same conclusion, larger margin.

🔑 **This is the single most consequential difference between the deployed R2 model and every LRS
that came before it.** Paper 2's archived `LinReg1`, refitted from its own 10 TU record under its
own Float32 arithmetic, reproduces to **1.6e-4** — so the archive *is* a Float32 fit — and that fit
has ρ(C̃) = **1.0002** and a starred gain of **10.3**. Refitting the *same data* in Float64 gives
ρ = **2.534** and gain **60.2**. **Every LRS deployed in this project before R2 was regularized by
its own round-off**; R2's is the first to run the actual λ = 0 least-squares solution.

Decomposing the deployed-vs-archived coefficient difference (physical affine map, relative):

| comparison | isolates | value |
|---|---|---|
| new record vs archive record, both Float64 `:normal` | the **record / system** | **1.23** |
| archive record, Float64 vs Float32 `:standardise` | **precision** | **12.48** |
| deployed-new vs archive-as-stored | both | 9.29 |

**Precision dominates the record by an order of magnitude.** The two fits differ mostly because the
archive's coefficients were never resolved, not because the dynamical system moved.

✅ **The normalization switch is exonerated, decisively.** `:normal` versus `:standardise` at λ = 0
in Float64 changes the physical slope by **3.0e-10**, the intercept by 4.7e-12 and Σ by 8.0e-14, and
leaves ρ(C̃) and the starred gain identical to four decimals. It is a mathematical no-op at λ = 0 —
and it *improves* conditioning by a factor 5.5 (`κ` 1.85e6 against 1.02e7). TODO-0's first half was
a free win.

⚠️ **And regularizing back to the archive's operator norms does not recover its behaviour.** λ = 1e-5
restores ρ ≈ 1.003 and gain ≈ 8 at +0.1 % training RMSE, but §3's λ probe shows the deployed summed
KS, bias and clamp rate barely move. So the precision finding explains why the *coefficients* differ;
it does not explain the excursions.

**Consequences.** Everything reading individual coefficients — #21 ρ(C̃), #22 gain — must be computed
in Float64 and **cannot be taken from an archived Float32 fit**; ⚠️ nor is Float64 alone enough,
because those two also move with *which record* the fit came from (2.534 archive-record against
2.689 new-record, both Float64). Aggregate diagnostics survive, because they average the errors away
(‖S − I‖ agrees to 4 %). This also explains why G1's coefficient reproduction looks good: it compares
two Float32 fits computed the same way, which agree with each other without either being close to the
true minimiser. All regime-0 numbers below are Float64.

### #23 Gram spectrum — which of three stories λ tells

$$\alpha_j = \frac{\sigma_j^2}{\sigma_j^2+\lambda}$$

![Gram spectrum](figures/fig5_gram_spectrum_new.png)

Harmonized (`:normal`) design, N = 3595, 67 features: $\sigma^2_{\max} = 2.03\times10^5$, median
$2.06\times10^{-2}$, $\sigma^2_{\min} = 7.06\times10^{-8}$, κ = 1.70e6, **full rank 67/67**.

| λ | directions erased (α < 0.5) | effective dof | branch |
|---|---|---|---|
| 0.01 | 31 / 67 | 37.1 | **rank-deficiency fix** |
| 0.1 | 40 / 67 | 26.8 | **shrinks determined directions** |
| 1 | 51 / 67 | 16.8 | shrinks determined directions |
| 10 | 60 / 67 | 9.3 | shrinks determined directions |

🔑 **Paper 2's two λ values straddle the boundary.** λ = 0.01 sits inside the small end of the
spectrum — story **(i)**, a conditioning fix whose accuracy cost is the suppression of exactly the
collinear directions a long history exists to supply, which is the direct argument for the L0
lever. λ = 0.1 already reaches directions above the median — story **(ii)**, shrinkage toward the
model's own marginal. They are not two points on one mechanism, and results at the two values
should not be read as a single trend. Story **(iii)** is excluded: λ = 0.01 erases 31 of 67
directions, so ridge is doing a great deal.

**And the convention moves the spectrum, as §8a predicted.** Switching to the uncentred
`:standardise` design multiplies $\sigma^2_{\max}$ by **31.5** and κ by **5.6**, leaving
$\sigma^2_{\min}$ and the median unchanged — the uncentred mean enters as one large leading
eigenvalue. That is the measured justification for TODO-0, and it is independent of where the
signal level ends up.

### #24 Total block sum and bias

$$S = A_* + \sum_k A_k + \sum_k B_k$$

‖S − I‖/√N_Q = **3.15e-2** under both conventions, agreeing to four digits; max |intercept| 1.1e-2
(harmonized) and 8.2e-3 (faithful).

🔴 **Correction to `plan.md` §8a and to the handoff.** Both state that under `:standardise` the
intercept has to carry the signal level. It does not. The scaled target mean reaches 2.85e4 while
the fitted intercept is at most 65 — the **coefficient block** carries the level, because at
leading order every lag in the window is ≈ `q^{n-1}` and S ≈ I whether or not the data were
centred. TODO-0's conclusion stands on #23's spectrum, not on the intercept.

### #21 / #22 ρ(C̃) and the starred-block gain

Under closure A (`q^{n*} ≈ q^{n-1}`) and closure B (**same-index pairing**, settled below):

**ρ(C̃) = 2.6888** and **H∞ starred gain 269.80** at ω = 0, on R1's record in Float64 — the
coefficients actually deployed in R2.

🔴 **Both move with the record AND with the precision, and the two compound.** Four values of ρ
from one configuration:

| coefficients | ρ(C̃) | H∞ starred gain |
|---|---|---|
| archived `LinReg1`, as stored (Float32) | 1.0002 | 10.3 |
| refit, archive 10 TU record, Float32 | 1.0002 | — (reproduces the archive to 1.6e-4) |
| refit, archive 10 TU record, Float64 | **2.534** | 60.2 |
| refit, archive 100 TU record, Float64 | **2.174** | 108.6 |
| **refit, R1 record, Float64 — DEPLOYED** | **2.6888** | **269.80** |

The design has κ ≈ 1e6–1e7 and the training target differs by ~1 `dQ` sd between records, so the
coefficient vector is neither well determined nor record-independent. **The aggregate conclusion is
robust and the digits are not** — quote ρ with its record and its precision, or quote it as a range.

🔴 **ρ(C̃) > 1 does NOT predict the deployed behaviour, and §3's λ probe is the proof.** `LinReg5`
and `LinReg6` have ρ = 1.0031 and 1.0033 and starred gains of 8.01 and 4.10 — a factor 30–65 below
the deployed cell on both — and their online summed KS, bias and clamp rate are indistinguishable
from it. Whatever produces the LRS's downward excursions, it is not the standalone recursion's
spectral radius. This retires the reading that ρ near 2.7 is a stability warning about the
deployment; it is a statement about an open-loop operator that deployment never runs, because the
solver supplies `q*` rather than the model's own previous output (#13).

🔴 **The strong ladder confirms this from the other side, and it is the stronger test.** ρ now runs
*below* 1 for three fitted-and-run cells — 0.9992, 0.9930, 0.9793 at λ = 10, 100, 1e4 — and their
deployed quality is 0.740, 1.297, 2.270: **not monotone in ρ, and the most contractive cell is the
worst of the three.** Meanwhile the *best* cell in the report, λ = 1 at 0.648, has ρ = **1.0057** —
above 1. So contractivity is neither necessary (the optimum is not contractive) nor sufficient (the
most contractive cell is nearly the worst). ρ moves over a range of 2.7 across this sweep while
deployed KS traces a U whose minimum sits on the non-contractive side of the crossing; a diagnostic
non-monotone in the thing it is meant to predict is not a predictor.

⚠️ **No fit-time scalar in this report orders these cells correctly.** The training residual is
monotone in λ (RMSE 0.0067, 0.0067, 0.0094, 0.0166, 0.0295, 0.1491) while deployed KS falls then
rises, so it cannot pick the minimum either; ρ is non-monotone; the starred gain falls monotonically.
The best deployed cell is neither the best-fitting nor the most contractive — it is the one that
trades a little fit for a correction small enough not to drive bands to zero, and *how much* is the
right amount is visible only in the coupled system. Any future criterion for choosing λ has to be
measured there.

### #25 Rank / pinv check — the metric does not work as specified

`plan.md` §0 item 11 specifies `‖X\Y − pinv(X)·Y‖`, reasoning that a nonzero gap would reveal that
the backslash had silently truncated and λ = 0 was never unregularized. **The premise is right and
the inference does not follow.** Because the backslash returns *the same* minimum-norm solution
`pinv` does, the gap is ~1e-15 **exactly when** truncation happens. Verified four ways on Julia
1.12.7 — an appended duplicate column, an inserted duplicate, a zero column, an exact linear
combination: rank drops in every case and the gap stays at machine precision.

**Use `rank_deficit = ncol − rank` instead.** On this design: rank 67/67, deficit **0**, so λ = 0
really is an unregularized fit here and the falsification criterion is safe. The *relative* gap
does retain power on a **near**-deficient design (a column duplicated to within 1e-12 gives a
relative gap of 1e-4 at nominal full rank), so it is still worth reporting; the absolute gap is
not.

### #26 Clamp census

$$\text{rate} = \#\{n : \exists i,\ |q_i^{n*}| < 10^{-2}\}\,/\,n_{\text{steps}}$$

**Zero. Everywhere.**

| record | firing rate | global min |q*| |
|---|---|---|
| tracked 10 TU | 0.000 over 4 000 steps | 3.99e-2 |
| tracked 100 TU | 0.000 over 40 000 steps | 2.95e-2 |
| channel tracked | 0.000 over 2 000 steps | 3.67e-1 |
| online LinReg1 … 74 | 0.000 over 40 000 steps × 5 configs | 2.19e-2 (LinReg1) |

🔑 **The inherited stabiliser never fires anywhere in the available HIT or channel data**, on
tracked *or* free-running records, and the closest approach is a factor 2.2 above the threshold.
`plan.md` §0 item 12 — *"until its firing rate is known, no stability or accuracy number in this
lineage is cleanly attributable to the model"* — **closes for these records**: every number in this
report is attributable to the model. The item stays open only for configurations not archived here,
and for Taylor-Green, whose training path additionally drops rows at a **different** threshold
(0.5e-2 against deployment's 1e-2, a factor 2 apart, with rows in between trained on and then
clamped).

---

## 4b. R1's tracking run, and what the online trajectories look like

### R1 — the tracking run

`data_track_dns512_les64_Re2000.0_tsim100.0_f64_lmwray3.jld2`: `q` (6, 40001) Float64, 401 stored
fields at 0.25 TU, `Re = 2000.0`, `Δt = 2.5e-3`, `savefreq = 100`, `freeze = 1`, OU seed 333.

Tracking error `|q_ref − q| / |q_ref|` over all 40 001 columns:

| band | max | mean | p99 |
|---|---|---|---|
| Z[0,6] | 1.545e-4 | 8.46e-6 | 5.12e-5 |
| E[0,6] | 2.314e-4 | 8.21e-6 | 5.94e-5 |
| Z[7,15] | 4.707e-5 | 5.47e-6 | 2.20e-5 |
| E[7,15] | 3.692e-5 | 4.14e-6 | 1.79e-5 |
| Z[16,32] | **9.338e-4** | 4.99e-4 | 7.85e-4 |
| E[16,32] | 5.502e-4 | 3.06e-4 | 4.70e-4 |

🔑 **The two smallest-scale bands track ~60× worse than the other four** — means 3–5e-4 against
4–8e-6. Expected in kind (smallest scales, least LF skill) but it is the margin that matters: the
driver's gate ships at a deliberately loose 1e-1 and the real bound should come from this table.

### The trajectories, one figure per closure

`analysis/plot_rebaseline.jl` writes one figure per closure that has runs on disk, currently ten,
`fig8_online_<model>.png`, each showing that
closure's online QoI trajectories against the regenerated reference across all six bands, with the
marginal the KS statistic scores drawn beside each band.

⚠️ **Band naming.** "The two top bands" elsewhere in this file means the two *highest-wavenumber*
ones, `Z[16,32]` and `E[16,32]`. Below they are called **the `[16,32]` pair** and the other four
**the larger-scale bands**, so nothing turns on which end of the spectrum "top" points at.

⚠️ **How to read them.** Regime C is free-running — nothing is replayed, `q*` comes from the
solver, and two runs launched from the same field decorrelate within an eddy turnover (~0.3 TU). So
*pointwise* agreement past the opening is not expected and its absence is not a defect. The
trajectory panel is read for the **envelope**: does the closure hold the right band of amplitudes,
does it drift, does it collapse. The marginal beside it carries the claim.

🔑 **The direction of each failure, which KS discards.** KS is a distance and has no sign, so the
ratio of means is reported beside it:

| model, mean(q) / mean(q_ref) | Z[0,6] | E[0,6] | Z[7,15] | E[7,15] | Z[16,32] | E[16,32] |
|---|---|---|---|---|---|---|
| LinReg1 (λ=0) | 0.941 | 0.943 | 0.926 | 0.928 | 0.911 | 0.913 |
| LinReg5 (λ=1e-5) | 0.947 | 0.950 | 0.932 | 0.934 | 0.919 | 0.920 |
| LinReg6 (λ=1e-4) | 0.949 | 0.952 | 0.936 | 0.938 | 0.924 | 0.925 |
| LinReg7 (λ=1) | 0.961 | 0.964 | 0.950 | 0.953 | 0.938 | 0.940 |
| LinReg8 (λ=10) | 0.970 | 0.971 | 0.963 | 0.964 | 0.953 | 0.954 |
| LinReg9 (λ=100) | 0.983 | 0.984 | 0.977 | 0.978 | 0.972 | 0.973 |
| LinReg10 (λ=1e4) | 1.011 | 1.012 | 1.010 | 1.010 | 1.019 | 1.018 |
| DDN | 0.941 | 0.967 | 0.947 | 0.941 | 0.821 | 0.842 |
| no model | 0.939 | 1.012 | 0.918 | 0.870 | **2.391** | **2.015** |
| Smagorinsky | **1.169** | **1.157** | **1.206** | **1.166** | **2.153** | **1.942** |

**Every TO-LRS cell below λ = 1e4 is low in every band and the two deterministic baselines are high
in `[16,32]`.** Under-dissipation at the smallest resolved scales is the baselines' failure; the TO
closures have the opposite sign, and nothing in the summed KS says so.

🔑 **λ is a monotone dial on this bias and it crosses zero.** Over λ ≤ 1e-4 it moves the LRS bias by
about one point per decade — real, monotone, and far too small to matter, which is what the weak
probe concluded. At λ = 1 the same dial has closed a third of the gap (0.913 → 0.940 on
`E[16,32]`), at λ = 10 half, at λ = 100 three quarters, and at λ = 1e4 it has **overshot to 1.018**.
So the low bias is removable by regularization alone; what the weak probe actually measured was
that 1e-4 is four decades short of the range where the dial has travel.

⚠️ **Removing the bias is not the same as fixing the closure**, and λ = 1e4 is the counterexample
sitting in this very table: its mean ratio is the best of any closure at 1.010–1.019 and its
summed KS is the second-worst in the report (2.270). A model can match the reference's mean in
every band while reproducing none of its distribution — which is exactly why the ratio of means is
reported *beside* KS here and never instead of it.

![LinReg1 online](figures/fig8_online_LinReg1.png)

🔑 **The LRS's error is a modest low bias carried by a left tail.** Its marginal sits on the
reference's through the body of the distribution — the modes line up in all six bands — but the
mean runs **6–9% low in every band**, and the trajectories show where that comes from: downward
excursions to values the reference never visits. `Z[16,32]` reaches **181** against a reference
minimum of **721**; `E[16,32]` reaches **6.2e-3** against **0.0243**.

![LinReg6 online](figures/fig8_online_LinReg6.png)

🔴 **λ = 1e-4 does not remove them.** Same left tail, same envelope. Put beside the λ = 0 figure
this is the clearest statement of the weak λ probe's negative result: a fit whose standalone
recursion is 2.7× less explosive produces a visually indistinguishable trajectory.

![LinReg7 online](figures/fig8_online_LinReg7.png)

✅ **λ = 1 all but removes them, and this is the best figure in the set.** `Z[16,32]` bottoms out at
**380** against the reference's 721, where λ = 0 reached 181 — the minimum is still below the
reference's, but by a factor 1.9 rather than 4 — and the top of the envelope lands at 3700 against
the reference's 3791, the closest match of any closure. The excursions are shortened, not abolished;
what is abolished is the class of them deep enough to trip the 1e-2 stabilizer. Worth reading beside
`fig8_online_LinReg1.png`: the change lives in the tails, which is exactly what a summed KS
dominated by the body of the distribution reports only weakly (0.836 → 0.648).

![LinReg8 online](figures/fig8_online_LinReg8.png)

⚠️ **λ = 10 goes one step further and one step too far.** Its `Z[16,32]` minimum, 587, is closer to
the reference's 721 than λ = 1's 380, so on the left tail alone it looks better — but its *maximum*
has fallen to 3451, now undershooting the reference's 3791, and its spread–skill ratio has dropped
to 0.759, outside S7's band. This is the first cell where the envelope is being squeezed from both
ends rather than lifted from below, and it is why the ladder's optimum is at λ = 1 and not here.

![LinReg10 online](figures/fig8_online_LinReg10.png)

🔴 **λ = 1e4 shows what over-damping looks like, and it is not a left tail.** The envelope has
collapsed inward from *both* sides — `Z[16,32]` spans 1777..2417 inside a reference spanning
721..3791 — so the marginal is a narrow spike sitting near the reference's mean. Its mean ratio is
the best in the report (1.019) and its summed KS the second-worst (2.270). Read this figure and the
λ = 1 one together: they are the two failure modes regularization moves between — excursions on one
side, collapse on the other — and λ = 1 is the measured turning point between them.

![DDN online](figures/fig8_online_DDN.png)

🔑 **The DDN fails differently, and only in `[16,32]`.** Its four larger-scale bands are as good as
the LRS's or better (`E[0,6]` KS 0.0869 against 0.1107), but `Z[16,32]` and `E[16,32]` both
overshoot the reference — max **7 967** against **3 791** — *and* repeatedly collapse toward zero:
minimum `Z[16,32]` **0.42** and `E[16,32]` **8.4e-6**, the latter more than three orders below the
reference's own minimum of 0.0243. The marginal is bimodal, with mass piled at the bottom of the
range where the reference has none.

![no model online](figures/fig8_online_nomodel.png)

![Smagorinsky online](figures/fig8_online_smag.png)

🔴 **The 3.710-against-2.487 ordering does not come from `[16,32]`, where KS is saturated.** Both
baselines pile up there — no model reaches `Z[16,32]` = **10 170** against the reference's median of
2 023 — and the KS values (0.84–0.97) sit where the distributions barely overlap either way. What
the trajectories add is that the eddy viscosity *does* cut the pile-up: max `Z[16,32]` **7 135**
against **10 170**, mean ratio 2.15 against 2.39.

**The gap is paid in the four larger-scale bands.** Decomposing 3.710 − 2.487 = 1.223: **1.048 of
it (86%) comes from those four** — 1.795 for Smagorinsky against 0.746 for no model — and only
0.174 from the saturated `[16,32]` pair. No model is close to the reference in the mean there
(0.918–1.012 on three of four, `E[7,15]` the exception at 0.870, and `E[0,6]` KS **0.0279**, the
lowest number anywhere in this report); `c_s = 0.07` is biased **16–21% high on all four**. A
statement about this untuned `c_s` on these kernels, not about Smagorinsky.

⚠️ **Smagorinsky here is an untuned point, not paper 2's baseline.** Paper 2 tuned `c_s` against its
own reference with its own kernels and reported 0.705; this is `c_s = 0.07` on upstream's rewritten
kernels against a different reference. Never quote the two side by side.

### Where the excursions come from — the step-level attribution

At every step where the level falls below the reference's own minimum on that band:

| | `q*` already below ref min | median `dQ` share of the shortfall |
|---|---|---|
| LinReg1, all bands | **96.6–99.5%** | 0.00 to −0.04 |
| DDN, `[16,32]` | 88.9–89.2% | −0.18 to −0.22 |

🔑 **At the excursion step the correction is not the proximate cause** — `q*` is already low and
`dQ` moves it by a few percent of the gap. But cumulatively the correction is doing the work: no
model runs `[16,32]` at 2.39× the reference, the LRS at 0.91×. Together with the λ probe's negative
result this locates the excursions in the **coupled** LF system rather than in the model's
open-loop spectrum.

### The clamp, and the asymmetry that remains

🔴 **The stabilizer fires on every weakly regularized LRS cell, cannot fire on the DDN** (memory
#59), **and stops firing entirely at λ ≥ 10.**

| closure | steps under the 1e-2 threshold, worst replica | global min `\|q*\|` | can the clamp fire? |
|---|---|---|---|
| LinReg1 (λ=0) | 1.84% | 6.2e-3 | yes — fires |
| LinReg5 (λ=1e-5) | 1.25% | 4.7e-3 | yes — fires |
| LinReg6 (λ=1e-4) | 1.18% | 6.9e-3 | yes — fires |
| LinReg7 (λ=1) | **0.00%** | **1.4e-2** | yes — but never does |
| LinReg8 (λ=10) | **0.00%** | **2.1e-2** | yes — but never does |
| LinReg9 (λ=100) | **0.00%** | **4.2e-2** | yes — but never does |
| LinReg10 (λ=1e4) | **0.00%** | **6.0e-2** | yes — but never does |
| **DDN** | **3.45%** | 4.6e-4 | **no — `MVG_sampler` never receives `q*`** |

🔑 **The DDN crosses the threshold nearly twice as often as the λ = 0 LRS and nothing stops it.**
That is a sharper statement of the asymmetry than a firing count alone: it is not that the DDN
stays clear of the condition, it is that the condition is never tested for it. The clamp lives only
in the `LinReg` path (`time_series_methods.jl:162,165,190,193`). So a LRS-vs-DDN difference is
model *plus* stabilizer, and **either clamp both or neither** before D6 runs.

🔑 **λ ≥ 1 gives a fourth option that the weak cells hid: don't need the clamp.** At λ = 1 the
smallest `\|q*\|` reached anywhere in 5 × 40 000 steps is 1.4e-2 and at λ = 10 it is 2.1e-2 — 1.4×
and 2.1× *above* the threshold — so the stabilizer is inert rather than merely quiet, and these
trajectories are what the model produces with no intervention at all. That removes the confound from
any comparison built on them: `LinReg7`-vs-DDN is model against model, where `LinReg1`-vs-DDN is
model-plus-clamp against model. If the "clamp both or neither" fix does not land before D6,
**`LinReg7` is the LRS cell to run it with** — best on KS and on dispersion, and for it the two
options coincide.

⚠️ **λ = 1's margin is 1.4×, not a comfortable one.** `LinReg8` clears the threshold by 2.1× and
`LinReg9` by 4.2×, so if D6's initial conditions push further into the left tail than these 100 TU
free runs did, λ = 1 is the first of the three that could start clamping and stop being a clean
model-versus-model comparison. Worth re-checking the census on the D6 ensemble rather than assuming
it carries over.

✅ **`E[16,32]` accounts for 100% of fired steps in all five λ = 0 replicas** — memory #59 checked
only the worst — and no other band is ever under the threshold on a fired step. The events are
**clustered, not a uniform tax**: 13 / 20 / 31 / 20 / 79 contiguous bursts, so replica 1's 160
steps are 13 bursts inside a single 0.78 TU window (steps 32 728–33 040).

⚠️ **The threshold sits close to the physics.** The reference's own `E[16,32]` minimum is 0.0243,
only 2.4× the 1e-2 clamp. Reconstructed `|q*|` minima: LRS 4.7e-3 – 8.6e-3, DDN down to 4.57e-4.
Memory #59 calls the DDN figure *"three orders below the threshold"*; the value is right and the
characterisation is not — 1e-2 / 4.57e-4 = **22**. Three orders is the right description of a
different quantity, the DDN's minimum on the **level**: `E[16,32]` reaches **8.4e-6** against the
reference minimum of 0.0243.

## 4c. D6 — the paired, multi-IC forecast experiment

**Run 2026-09-16.** Three closures, each forecasting from the **same** initial conditions:
`LinReg1` (h = 5, λ = 0), `LinReg7` (h = 5, λ = 1) and the `DDN`. K = 90 ICs × M = 10 members ×
1200 lead steps, `nwarm = 100`, Float64, on `gpu_h100`.

🔴 **Scored on 87 ICs, not 90, because THREE LinReg1 MEMBERS DIVERGED.** Confirmed from
`slurm-26795092_67.out` (Rik, 2026-09-16), member 7 of `k` = 170:

```
[ Info: t = 1.75   Δt = 0.0025   umax = 2.2
┌ Warning: Unreasonable large QoI at n = 712     (RikFlow.jl:576)
┌ Warning: NaNs detected in the solution. Stopping the simulation.
ERROR: q has 713 columns, expected nt + 1 = 1301  (run_d6.jl:413)
```

Run step 712 is **lead 612 = 1.53 TU past the warm-up — inside the scored grid**.

⚠️ **The driver then aborts the whole task, so one divergence costs the rest of the IC.**
`run_d6.jl:413` raises on the short `q`, and members after the failing one are never attempted:

| IC | completed | diverged | never attempted |
|---|---|---|---|
| 170 | 1–6 | **7** (confirmed) | 8–10 |
| 197 | 1–2 | **3** (inferred, same signature) | 4–10 |
| 313 | 1–5 | **6** (inferred) | 7–10 |

**Stability, metric #16:**

| | LinReg1 | LinReg7 | DDN |
|---|---|---|---|
| diverged members / attempted | **3 / 886 = 0.34%** | 0 / 900 | 0 / 900 |
| **member stability fraction** | **0.9966** | **1.0000** | **1.0000** |
| ICs containing a divergence | **≥ 3 of 90** | 0 | 0 |
| IC stability fraction | **≤ 0.967** (bound) | 1.000 | 1.000 |

The IC figure is a **bound, not a measurement**: 14 members were never tested, so more of those three
ICs might have failed. 🔴 **This is a measurement defect as well as a robustness one** — stability
fraction is metric #16, and the driver currently destroys the data needed to estimate it. It should
record the member as diverged and continue to the next, not abort the task.

🔑 **`umax` was DECREASING into the blow-up — 5.0 → 3.4 → 2.2.** This is not a velocity CFL runaway.
The flow was decaying and the *QoI* exploded, which is the signature of the TO correction
over-draining until `src_Q` gets small and `tau = dQ/src_Q` blows up. 🔴 **And `k` = 170 has the
2nd-highest turbulence-gate rate of all 90 ICs (3.32%)** — the gate was firing there and did not
prevent it. That is a limit of the gate, not a case it missed.

🔴 **So the skill and calibration tables below are conditional on the ICs LinReg1 survived, which
flatters LinReg1.** The three ICs are excluded from **all three** closures (`D6_EXCLUDE_ICS`),
because a paired comparison must run over the intersection of what the closures produced — but for
LinReg1 that intersection is not a random subset, it is "the ICs where the model did not break".
Read the skill table with the stability table above it, never on its own. `load_members` refuses a
ragged ensemble outright — the finite-M correction is a function of M — so nothing was silently
averaged.

| | LinReg1 | LinReg7 | DDN |
|---|---|---|---|
| **mean skill**, all 30 (band, lead) cells | **0.5521** | 0.5666 | 0.9284 |
| short leads (0.25, 0.5 × T_int) | **0.2908** | 0.3135 | 0.7655 |
| long leads (2, 5 × T_int) | 0.7767 | **0.7735** | 1.0611 |
| **spread–skill inside S7's [0.8, 1.25]** | **23 of 30** | 2 of 30 | 3 of 30 |
| median spread–skill ratio | **0.974** | 0.600 | 0.405 |
| clamp firing rate | 0.392% | 0% | 0% |
| bands saturating inside the grid | 0 of 6 | 0 of 6 | 2 of 6 |

Skill is the RMSE of the ensemble mean as a fraction of the climatological level, so **1.0 is
"no better than climatology"** and lower is better.

### 🔴 D6 inverts the free-running ranking, which is the result this experiment was built to get

On free-running 100 TU marginal KS, LinReg7 (λ = 1) was the best cell at 0.648 against LinReg1's
0.836 (§3). **Paired, LinReg1 wins on skill at 9 of the 10 short- and mid-lead cells and is
calibrated where LinReg7 is not** — 23 of 30 cells inside S7's band against 2, and a median ratio
of 0.974 against 0.600. LinReg7 is systematically **under-dispersed**: its ensemble is too narrow
for the error it actually makes.

🔑 That is exactly the failure mode #61 predicted. Free-running marginal KS compares *climatologies*
and carries 90–170 independent samples; it cannot see calibration at all, and it ranked the two
cells backwards. The λ ladder's "null result" (#63) now reads as a null on the *wrong statistic*.

### The DDN is not a weak model here; it is worse than climatology in the small scales

| skill / climatology | lead 0.25×T | 1×T | 5×T |
|---|---|---|---|
| `Z[16,32]` LinReg1 / DDN | 0.105 / **1.432** | 0.608 / **1.770** | 0.793 / **1.647** |
| `E[16,32]` LinReg1 / DDN | 0.110 / **1.129** | 0.610 / **1.445** | 0.800 / **1.407** |

The DDN exceeds 1.0 at **every** lead in both smallest-scale bands, reaching 1.98 — it is the only
configuration that would be improved by replacing its forecast with the climatological mean, and
it is the only one whose bands saturate inside the grid (`Z[16,32]`, `E[16,32]`, both at lead 54).
The LRS ratio `LR1/DDN` there runs **0.073–0.57**: a factor 3–14 better.

![D6 fans, LinReg1](figures/fig9_d6_fans_LinReg1.png)
![D6 fans, DDN](figures/fig9_d6_fans_DDN.png)

`analysis/plot_d6_fans.jl`, 4 initial conditions at the derived non-overlapping stride (fans are
3.25 TU, the minimum IC gap is 0.75 TU, so every 5th). The mechanism is visible: in the DDN figure
**all four fans drive `Z[16,32]` and `E[16,32]` to zero** and hold them there for ~1 TU, then
overshoot to ~6400 against a reference maximum of 3791. LinReg1's fans stay inside the reference's
own envelope in every band. Over all 90 ICs the DDN sits below the reference's own minimum on
**19.5%** of `Z[16,32]` steps and **13.5%** of `E[16,32]` steps, reaching 0.234 and 1.13e-5 against
reference minima of 720.6 and 0.0243 — factors of 3075 and 2147.

⚠️ **Part of that gap is a missing gate, not the noise model.** All four `TURBULENCE_GATE` sites are
inside `LinReg`'s `get_next_item_timeseries`; `MVG_sampler`'s method takes **no `q_star` argument**
(`src/time_series_methods.jl:111`), so the DDN cannot gate even in principle, and it fired on 0 of
276 000 steps. Counterfactually an LRS run in the same states would have zeroed `dQ` on **6436 of
276 000 forecast steps (2.33%), affecting 115 of 230 runs (50%)**, first trip at a median lead of
447 steps (1.12 TU), always triggered by `E[16,32]`. So the small-scale comparison is not
step-for-step apples-to-apples. The 2.33% bounds it; only a gated DDN re-run would separate the two.

### Where the LRS does not win

⚠️ **`E[0,6]` is the DDN's band.** It is better there at every lead — `LR1/DDN` runs 1.04–1.27 and
`LR7/DDN` 1.01–1.21. That is one of six bands, it is the band with the shortest correction
timescale (`T_int(dQ)` = 0.0081 TU, effectively white), and it is the one place where an i.i.d.
sampler is the right model of the correction. Worth stating rather than averaging away.

⚠️ **No band saturates inside the grid for either LRS cell** (0 of 6), so every LRS number here is a
pre-saturation measurement and the grid does not bound how much further the error would grow.
Reported, not extrapolated. The DDN saturates only in the two bands where it is already worse than
climatology.

⚠️ **The validation was never run** (`--array=0`): all three directories contain 0 `d6_valid_*`
files. `compare_validation` is the correctness check on the whole D6 path against a trajectory
produced by different code years earlier, and it costs one task. It should be run against LinReg1,
whose oracle is R2's own LinReg1 replica 1, before these numbers are quoted anywhere final.


## 5. Findings that change the plan

1. **SC-48 is settled: same-index pairing.** The lag-*k* block is `(q^{n-k}, q_star^{n-k})`, the
   same physical step, asserted independently in the batch builder and in the deployed ring buffer
   by planting step indices as array values. ⇒ C̃ is `h·N_Q` with first block row
   `{A₁+A_*+B₁, A₂+B₂, …, A_h+B_h}`; the `(h+1)·N_Q` one-step-older form does not apply.
   `plan.md` §8a's `DEFERRED-TO-G1` closes.

2. **RH-1's null is the in-sample histogram, not flatness.** §9's outcome table reads flat-vs-U
   directly against zero; the residual's excess kurtosis of +0.9 to +2.0 makes that wrong by −3.7
   to −8.7 units of convexity before any model defect is involved.

3. **The archive is five configurations at h ∈ {5, 10, 40}, not four at h ∈ {5,20} × λ ∈ {0,0.01}**
   (`plan.md` §7). h = 5 exists only at λ = 0; h = 40 only in the `_rand_initial_dQ` family;
   LinReg74 has 2 replicas. The "≥6 further rung-spanning configurations" arithmetic needs
   restating on that basis.

4. **§12's G1 acceptance of `rtol = 1e-6` on `c` is unattainable and should be restated in
   prediction space.** Measured: coefficients reproduce to 2.8e-5–7.6e-5, predictions to
   6.4e-7–8.3e-7. On a Float32 archive with κ ≈ 1e7 the coefficient vector is not the reproducible
   object; the fitted subspace is.

5. **Coefficient-level diagnostics require Float64** (§4 above). Any ρ(C̃) or gain quoted from a
   Float32 fit — including §25's session-log values — should be recomputed.

6. **Metric #25 as specified has no power**; use the rank deficit.

7. **The intercept does not carry the level** under an uncentred convention (§4, #24).

8. **`autocorr` had a broken degeneracy guard** — `den == 0` never fires, so a constant series
   returned `[1.0, 0.999]` from rounding noise. It reported DDN's constant predictive mean as having
   lag-1 autocorrelation 1.0000 when the truth is that it has none. Now guarded relative to the
   series' own magnitude, with a regression test.

9. **Phase 0.4's gap is measured: 1.89e-2 relative overall**, 3.1–13.8 % of a `dQ` standard
   deviation per QoI, correlation 0.999996. This is the structural O(‖sgs‖²) term (§0 item 8), and
   at O(10⁻²) it fires 0.4's stated consequence: later rungs should buffer recomputed QoIs rather
   than `q_star + dQ`.

10. **`t_int` is not one number** (§1): 0.008–0.302 TU across QoIs, and $T_{\exp}$ vs
    $T_{\text{int}}$ disagree by up to 4× within a QoI. Nothing that scales in `t_int` should quote
    a single value.

11. **The $N_{\text{eff}}$ correction is a no-op on RH-1's ranks** (§2) — needed in principle,
    inert here. And the χ²-on-N_eff rescaling *over*-corrects: on a synthetic AR(1) null the raw
    test rejects 87 % at nominal 5 % and the rescaled one 0 %. `chi2_eff` is a conservative guard,
    not a calibrated p-value; the contrasts are the primary statement.

12. 🔴 **The "always score `dQ`, never `q`" rule is regime-B-scoped, and applying it to regime C
    was an error in an earlier version of this report** (Rik). O7's measurement came from the
    offline unrolled driver, where the replayed `q*` pins the level. Free-running, the level is
    exactly what the claims are about, and it is what paper 2 scores. Two things followed from
    fixing it: **G1's online acceptance became checkable and passes exactly** (§3), and the level
    turned out to be a near-null diagnostic for temporal structure while remaining the right
    target — so regime C now reports both. `plan.md` §9 spec 1 and `metrics.md` §1 already say
    this correctly; the error was in reading them as unconditional.

13. **Δρ₁ on the level is not a weak statistic but a null one** (§3): 0.000–0.003 across five
    configurations spanning a factor 5 on KS. `plan.md` §8b argues for the integral-timescale lag
    on the grounds that lag 1 has *little* range at ρ₁ ≈ 0.94; on the level the range is gone
    entirely. The pre-registered Δρ₁ adequacy statistic should be specified on the correction, or
    the adequacy test restated.

14. 🆕 **The two HIT tracked records are two realisations, and only one of them reproduces paper
    2's fit** (header, §4, #21/#22). They share an initial field and a reference, so the level `q`
    agrees to 1e-5–1e-4 of a standard deviation — less than either one's own tracking error — but
    the correction `dQ` decorrelates to **0.7–1.1 sd** beyond step 1000, across most of the
    (400, 4000) training window. Refitting `LinReg1` in Float64 over that window reproduces the
    archived coefficients to **7.6e-5** from the 10 TU record and **0.282** from the 100 TU one.
    Consequences: the M0 scored here is a *different fit of the same configuration*, not paper 2's
    model; `test/test_g1.jl` must keep using the 10 TU record, which is the only one that can
    reproduce the archive; and **ρ(C̃) and the H∞ gain are record-dependent** — 2.534/60.2 on the archive 10 TU refit, 2.174/108.6 on the archive 100 TU one, and 2.689/269.8 on R1, all in Float64. Neither `plan.md` nor `metrics.md`
    distinguishes the two records; both should, wherever a coefficient-level number is
    pre-registered.

15. 🆕 **`_rand_initial_dQ` is a stale filename, not a warm-start treatment.** The two archive
    roots hold the **same runs**: frozen `data_online_tsim100.0_replica<i>.jld2` is bit-identical
    in `q` over all 40 001 columns to the working repository's `..._replica<i>_rand_initial_dQ.jld2`
    (LinReg1 replicas 1–3, LinReg63/64 replicas 1–2), and `paper_runs/online_sgs.jl:62,84` seeds
    `spinnup_data` from `data_track.dQ[:, 1:100]` exactly as `6_online_TO_LRS.jl:62,83` does. Three
    consequences, all favourable: LinReg73/74 are on the **same** warm start as LinReg1/63/64 and
    as D6, so regime C mixes nothing; `compute_ks.jl`'s split between `isfile` on one naming and
    `load` on the other is harmless, so **G1's online acceptance stands unqualified**; and the
    `replica_groups` guard is still required, but to stop replica 1 being **counted twice** in a
    six-file glob rather than to separate treatments.

---

16. 🆕 🔴 **D6 REVERSES THE FREE-RUNNING RANKING, and that is the point of the experiment** (§4c).
    On 100 TU marginal KS, LinReg7 (λ = 1) beat LinReg1 (λ = 0), 0.648 against 0.836. Paired over
    87 shared initial conditions, **LinReg1 is better on skill at 9 of 10 short- and mid-lead cells
    and is calibrated where LinReg7 is not**: 23 of 30 (band, lead) cells inside S7's [0.8, 1.25]
    against 2, median spread–skill ratio 0.974 against 0.600. LinReg7 is systematically
    under-dispersed. ⇒ **#61's diagnosis is confirmed by a positive result, not just by a power
    argument**: the free-running statistic ranked the two cells backwards, and the λ ladder's "null"
    (#63) was a null on the wrong statistic. Any cell selection made on marginal KS must be redone
    on D6 before it enters the paper.
    🔴 **RESOLVED 2026-09-16: they diverged** (`slurm-26795092_67.out`). So the finding is **S2′'s
    stability–accuracy Pareto front, not a clean win.** LinReg1 is more skilful and better
    calibrated *on the ICs it survives*, and it is the only cell that breaks: member stability
    fraction **0.9966 against 1.0000** for LinReg7 and the DDN, ≥3 of 90 ICs affected. Since the
    excluded ICs are exactly the ones LinReg1 broke on, the skill table is conditioned in LinReg1's
    favour and must never be quoted without the stability numbers beside it.

17. 🆕 **The DDN is worse than climatology in both smallest-scale bands** (§4c): skill/climatology
    1.13–1.98 at every lead in `Z[16,32]` and `E[16,32]`, the only configuration for which that is
    true and the only one saturating inside the grid. ⚠️ Part of the gap is structural rather than
    the noise model — `MVG_sampler` cannot apply `TURBULENCE_GATE` (its method takes no `q_star`),
    and an LRS run would have gated 2.33% of steps in 50% of runs. ⇒ Report the DDN's small-scale
    deficit with that census attached, or re-run a gated DDN; do not quote the raw gap alone.


## 6. Verdict on `plan.md` §9's branch

§9 asks whether M0's RH-1 is U-shaped (L2's motivation is measured), flat (**stop — talk to Rik**,
L2 demotes to confirmatory) or ∩-shaped (check the clamp census and tracking-noise injection
first).

**It is U-shaped in four of six QoIs and biased in the other two — read against the correct null,
with the two ∩-candidates ruled out.**

- The four smaller-scale bands are under-dispersed: held-out residual 14–18 % wider than the fitted
  Σ, convexity +10 to +19 against the in-sample null, all intervals excluding zero. **L2's
  motivation is measured, on the axis it was proposed for.**
- The two largest-scale bands are **biased, not mis-dispersed**: variance right to 1 %, slope +9.1
  and +7.5. That is an L0/L1 target, not an L2 one.
- §9's ∩ branch asked for two checks before believing over-dispersion. Both are now closed:
  the **clamp never fires** (rate 0 on every record) and the archived configurations have
  `tracking_noise = 0.0`. So the low-band ∩ is not an artefact of either.

**No stop.** L2 stays as proposed, with a sharper claim available than the plan assumed — the defect
is band-selective, and the paper can say which mechanism each band needs.

---

## 7. What is still blocked

| | Blocker |
|---|---|
| **RH-2** | The archive has `n_replicas = 5` ⇒ 6 bins, 4-dof spread estimate. Computable, not quantitative. |
| **#17 spread–skill vs lead, RH-3** | ✅ **CLOSED 2026-09-16.** D6 ran three closures at K = 90, M = 10; §4c reports spread–skill by lead and the rank histograms for all three. Scored on the 87 ICs common to all of them. |
| **DDN in regime C** | ✅ **CLOSED.** R2 ran the DDN online, 5 × 100 TU, and §3 scores it. The archive never had these runs; these are new measurements, not a reproduction. |
| **λ > 0 offline reproduction** | 🔴 **The parity check is no longer "not run" — it is RUN and it FAILED.** Measured on R1's record 2026-09-15: against the exact ridge minimiser the `RegularizedLeastSquares` ADMM iterate differs by a relative **0.970 at λ = 1e-5, 0.903 at 1e-4, 0.452 at 1e-2**, and its training RMSE is ~0.00714 at every one of those λ — it is iteration-limited, not λ-limited, so a sweep through it is not a sweep in λ. `5_train_LinReg.jl` now solves `:l2` exactly (`ridge_solver = :exact`) and keeps ADMM only for reproducing paper 2 and for `:nuclear`. G1's λ = 0.01 *coefficient* acceptance against the archive is still not run and now needs the `:admm` path explicitly. λ = 0 is reproduced for all three available configurations, and G1's **online** acceptance passes for all five (§3). |
| **The h and λ sweep on the new system** | 🔴 Only h = 5 exists rebaselined (λ ∈ {0, 1e-5, 1e-4}). h ∈ {10, 40} and λ = 0.01 have never been run post-merge, so §3's archive subsection is the only place a sweep can be read — on the old system. |
| **Which `T_int` estimator is right** | ⚠️ The Sokal-window estimator here gives 0.25–0.54 TU on the level; `report_marginals` gives 0.94–1.08 TU on the same record (#58). A factor ~2, and D6's grid is sized on the larger one. |
| **D6's validation** | 🔴 **NOT RUN.** All three run directories hold 0 `d6_valid_*` files — the `--array=0` task was never submitted. `compare_validation` checks the whole D6 path against a trajectory produced by different code years earlier, and it costs one task. It belongs to LinReg1, whose oracle is R2's own LinReg1 replica 1. §4c's numbers are unvalidated until it runs. |
| **LinReg1 diverges on ~0.3% of members** | 🔴 **CONFIRMED 2026-09-16**, not infrastructure: `Unreasonable large QoI at n = 712` then NaNs (`slurm-26795092_67.out`), at lead 612 = 1.53 TU, inside the scored grid. 3 of 886 attempted members; LinReg7 and DDN 0 of 900. ⚠️ `run_d6.jl:413` aborts the whole task on one bad member, so 14 more were never attempted and the per-IC stability fraction is a bound (≤0.967), not a measurement. **Fix the driver to record and continue before re-running.** ⚠️ The gate does not prevent it — `k` = 170 has the 2nd-highest gate rate of 90. |
| **A gated DDN** | ⚠️ `MVG_sampler` cannot apply `TURBULENCE_GATE`; see §4c. Either report the 2.33% / 50% counterfactual census alongside the DDN's small-scale deficit, or re-run the DDN with a gate. Open decision. |
| **#5, #6, #3** | Deferred by `metrics.md` §6. |
| **Channel, Taylor-Green** | Only the channel tracked QoI cache is present; no channel or TG fits or online runs were scored. |

---

## 8. Test coverage

`julia --startup-file=no --project=test test/runtests.jl` — **1461 tests, all passing.** Run
directly rather than through `Pkg.test`, because the suite deliberately has no RikFlow dependency:
the `ts_*` layer is stdlib-only, so CI never loads IncompressibleNavierStokes, CUDA, Makie or Lux.

| Test | Status |
|---|---|
| **V0** harness | ✅ |
| **V1** `build_history` vs all five archived `create_history` copies | ✅ HIT / channel / `time_solvers` bit-identical; TG = builder + an exact column mask; the inline online copy row-by-row; plus the real tracked record at h ∈ {5, 10} |
| **V1** archived coefficients | ✅ LinReg1, 64, 74 at λ = 0 — `rel_pred` 6.4e-7 to 8.3e-7 |
| **V2** batch ≡ online buffer | ✅ |
| **V16** ridge parity, unpenalized intercept | ✅ incl. QR vs normal equations on a collinear design |
| **V17** NLL, CRPS vs Monte Carlo, Δρ on a known AR(1), erf to 4e-16 | ✅ |
| **V18** companion blocks, total block sum, starred gain, Float64 invariance | ✅ |
| **V23** Gram spectrum branches | ✅ |
| **V25** rank deficit vs pinv gap | ✅ asserts the specified metric's failure |
| **V26** level ≡ increment ranks | ✅ |
| **V27** uniformity coverage, signed dispersion, χ² over-rejection | ✅ |
| **V28** D6's lead-resolved scorer — planted-index truth alignment, the finite-`M` correction (1 corrected against √(M/(M+1)) uncorrected), signed dispersion, flatness as coverage over 30 replications, saturation reported not extrapolated, and `score_d6.jl`'s whole driver on synthetic members | ✅ `test_d6_score.jl`. ⚠️ `plan.md` §14 files V28 under `test_rollout.jl`, which does not exist |
| **V29** D6's IC selection and packaging — both pool constraints per IC, `K = 180` at 0.4818 TU, `K = 400` fails loudly, the warm-up slice reducing to the archived driver's `dQ[:, 1:100]` at `k = 1`, and the step→column convention measured against the record | ✅ `test_d6_ics.jl` |
| **V30** the OU replay — equality with the real `OU_forcing_step!`, the Markov property, `n = 0` a provable no-op, grid independence | ✅ `test_ou.jl`; the advance count itself is measured by `analysis/ou_replay.jl` on a CPU mini-solve (`nstep + 1`) |
| **SC-48** index convention | ✅ |
