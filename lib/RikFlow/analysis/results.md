# Results — M0 and DDN scored on HIT

**What this is.** The first round of measurements for paper 4: TO+LRS (**M0**) and paper 1's
data-driven noise model (**DDN**) scored on one axis, on existing data. No new model was built and
no simulation was run — **0 SBU**. Companion to `meta_files/plan.md` (design) and `meta_files/metrics.md` (definitions);
where a number here disagrees with either of those, this file is the measurement and they are the
prediction.

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

**Data.** HIT. `D1` the 100 TU tracked record, `D3` the 100 TU HF reference (40 001 points),
`D5` the archived online ensembles, `D8` a split of D3 for the KS noise floor. Paper 2's archive is
gitignored and lives outside the repository; `RIKFLOW_ARCHIVE` and `RIKFLOW_DEV_ARCHIVE` point at
it.

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

![HF reference and tracked LF QoIs](figures/fig1_trajectories.png)

The left column is the 100 TU overview with the training window in blue and the held-out window in
pink. The right column is the same data at full resolution across the boundary, and it exists
because the overview shows **one line, not two**: nudging holds the tracked low-fidelity QoIs on
the reference to

| | Z[0,6] | E[0,6] | Z[7,15] | E[7,15] | Z[16,32] | E[16,32] |
|---|---|---|---|---|---|---|
| `rms(track − ref) / sd(ref)` | 8e-5 | 8e-5 | 4e-5 | 3e-5 | 2.3e-3 | 1.4e-3 |

So the tracked record *is* the reference for practical purposes, and the interesting quantity is
the correction that holds it there. In the right column the orange dashed predictor `q*` separates
visibly from the corrected `q` only in the two smallest-scale bands — which is exactly the
statement that `dQ` is a 0.2–1.5 % correction on the level.

![The SGS correction dQ](figures/fig2_dQ.png)

### `t_int`, both ways, per QoI

🔴 Every archived document says 0.04 TU. Two different timescales were being conflated: an
exponential fit to the lag-1 autocorrelation, $T_{\exp} = -\Delta t/\ln\rho_1$, and the integral
$T_{\text{int}} = \Delta t\,(\tfrac12 + \sum_k \rho_k)$ to the first zero crossing. Measured on the
reference `dQ`:

| QoI | $\rho_1$ | $T_{\exp}$ [TU] | $T_{\text{int}}$ [TU] | `sd(dQ)` |
|---|---|---|---|---|
| Z[0,6] | 0.9556 | 0.0551 | 0.1118 | 2.79 |
| E[0,6] | 0.7435 | 0.0084 | 0.0082 | 8.62e-3 |
| Z[7,15] | 0.9802 | 0.1253 | 0.0923 | 3.60 |
| E[7,15] | 0.9899 | 0.2474 | 0.0669 | 5.37e-4 |
| Z[16,32] | 0.9976 | 1.0560 | 0.2926 | 30.0 |
| E[16,32] | 0.9964 | 0.7007 | 0.3017 | 6.63e-4 |

**Neither number is 0.04 TU, and they disagree with each other by up to a factor 4 within a single
QoI** (E[7,15]: 0.247 against 0.067). The median $T_{\text{int}}$ is **0.102 TU**, so the Δρ lag
used below is **41 steps**. ⚠️ The spread across QoIs is a factor 37 in $T_{\text{int}}$, so a
single project-wide `t_int` is not a well-defined object; P2c's cost, which scales in it, should be
quoted per QoI or against the median with the range stated.

⚠️ **`sd(dQ)` spans four orders of magnitude.** Anything that pools QoIs in raw units is an
enstrophy statistic with the energy bands contributing nothing. Every pooled number below is
normalised.

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

![RH-1](figures/fig3_rank_histograms.png)

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

![Dynamics against calibration](figures/fig4_dynamics_vs_calibration.png)

The lag-1 autocorrelation of the predicted mean `dQ`: M0 gives 0.735–0.999, tracking the realised
values; **DDN gives exactly 0.0000 in every band**, because its predictive mean is a constant. Set
beside its near-flat histograms and its climatological CRPS, that is the trap `metrics.md` states
abstractly — **flat histogram, good marginal score, no dynamics at all** — demonstrated without
running a single large-λ cell.

---

## 3. Marginal and temporal accuracy — regime C (online, coupled)

Scored on the **QoI level** against the HF reference level, with the correction alongside. ⚠️ The
five archived configurations are **not one experiment**: h ∈ {5, 10} come from paper 2's frozen
archive (`tsim100.0` family) and h = 40 exists only in the working repository's `_rand_initial_dQ`
family, a different warm start. LinReg74 has **2 replicas, not 5**.

| config | h | λ | M | family | summed KS `q` | ens KS | Δρ₁(q) | Δρ₄₁(q) | summed KS `dQ` | Δρ₁(dQ) | stab |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LinReg1 | 5 | 0 | 5 | frozen | 0.142–0.371 | 0.198 | 0.000 | 0.051 | 0.170–0.269 | 0.022 | 1.00 |
| LinReg64 | 10 | 0 | 5 | frozen | 0.213–0.392 | 0.197 | 0.000 | 0.057 | 0.203–0.308 | 0.026 | 1.00 |
| LinReg63 | 10 | 0.01 | 5 | frozen | 0.400–0.580 | 0.466 | 0.001 | 0.050 | 0.293–0.373 | **3.459** | 1.00 |
| LinReg74 | 40 | 0 | 2 | dev | 0.281–0.436 | 0.250 | 0.000 | 0.063 | 0.213–0.351 | 0.504 | 1.00 |
| LinReg73 | 40 | 0.01 | 5 | dev | 0.472–0.694 | 0.603 | 0.003 | 0.162 | 0.740–0.806 | **5.288** | 1.00 |

![Regime C](figures/fig6_online.png)

### #10 / #11 Summed and ensemble KS, and G1's online acceptance

$$\text{KS}_r = \sum_{i=1}^{N_Q}\sup_x\big|F_i^{(r)}(x) - F_i^{\text{ref}}(x)\big|$$

summed over QoIs, **never over replicas**; the ensemble form pools replicas into one EDF first.
Both are reported and never averaged together.

🔑 **G1's online acceptance passes, exactly.** `plan.md` §12 asks for the summed ensemble KS per
configuration to fall within paper 2 Fig. 6's replica min–max range. Read from the archived
`ks_dists_*.jld2` tables and compared against this round's level-based numbers:

| config | this round | paper 2's archived table | archived ensemble |
|---|---|---|---|
| LinReg1 | 0.142–0.371 | 0.142–0.371 | 0.198 |
| LinReg63 | 0.400–0.580 | 0.400–0.580 | 0.466 |
| LinReg64 | 0.213–0.392 | 0.213–0.392 | 0.197 |
| LinReg73 | 0.472–0.694 | 0.472–0.694 | 0.603 |
| LinReg74 | 0.281–0.436 | 0.281–0.436 | 0.250 |

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

![Autocorrelation](figures/fig7_autocorr.png)

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

**5/5 replicas complete 100 TU for every configuration; 2/2 for LinReg74.** Stability fraction 1.00
across the board, so this data cannot discriminate on stability at all. Paper 2's own table agrees:
`n_unstable = 0`.

🔑 Set that against **ρ(C̃) = 2.174** for the h = 5, λ = 0 cell (§4). The standalone QoI process has
a fastest-growing mode more than doubling every step, and the coupled system runs 100 TU without a
single failure. `plan.md` §8a's caveat — *ρ(C̃) < 1 is neither necessary nor sufficient for coupled
stability* — is no longer a caveat but a measurement.

⚠️ **The two halves of that sentence come from different fits, and the conclusion survives it.**
The 2.174 is this round's refit on the 100 TU record; the 5/5 stable replicas were produced by the
*archived* model, i.e. by the 10 TU fit, which refits in Float64 to **ρ(C̃) = 2.534** (H∞ gain 60.2
against 108.6). Either number is ≫ 1, so the measurement stands whichever fit is paired with the
replicas — but ρ(C̃) is record-dependent as well as precision-dependent and must be quoted with its
record, never bare. See finding 14.

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

## 4. Mechanism diagnostics — regime 0 (fit-time)

### 🔴 A precision result that gates three of the metrics below

The tracked records are **Float32** and this design has $\kappa(X) = 1.7\times10^6$ (harmonized) to
$9.5\times10^6$ (faithful), so $\kappa\cdot\varepsilon_{32} = 0.20$: **in single precision the
coefficient vector is not resolved.**

The clean way to see it: at λ = 0 the two normalization conventions differ only by subtracting a
constant from every design column and from the target, and a least-squares fit carrying an
intercept is invariant to exactly that, so their slope blocks *must* agree.

| | Float32 | Float64 |
|---|---|---|
| slope-block relative difference between conventions | **3.32** | **4.7e-11** |
| slope-block difference, Float32 vs Float64 | — | **0.993** |
| ρ(C̃) | 1.012 | **2.174** |
| H∞ starred-block gain | 25.3 | **108.6** |
| total block sum ‖S − I‖ | 3.07e-2 | 3.15e-2 |

**Consequences.** Everything reading individual coefficients — #21 ρ(C̃), #22 gain — must be
computed in Float64, and **cannot be taken from an archived Float32 fit**; ⚠️ nor is Float64 alone
enough, because those two quantities also move with *which tracked record* the fit came from
(2.174 against 2.534 — see #21/#22 and finding 14). 1.012 versus 2.174 is
the difference between "marginally unstable" and "violently unstable" from the same data. Aggregate
diagnostics survive, because they average the errors away (‖S − I‖ agrees to 3 %). The density
metrics in §2 also survive — recomputing them in Float64 moved the convexities by at most 1.1 and
changed no conclusion. This also explains why G1's coefficient reproduction is good (7.6e-5): it
compares two Float32 fits computed the same way, which agree with each other without either being
close to the true minimiser. All regime-0 numbers below are Float64.

### #23 Gram spectrum — which of three stories λ tells

$$\alpha_j = \frac{\sigma_j^2}{\sigma_j^2+\lambda}$$

![Gram spectrum](figures/fig5_gram_spectrum.png)

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

**ρ(C̃) = 2.1741**, ℓ2 starred gain 1.03, **H∞ starred gain 108.6** at ω = 0 — all three on the
**100 TU** record's refit, in Float64.

🔴 **Both numbers move with the record, not only with the precision.** Refitting the same
configuration and window on the **10 TU** record — the one paper 2 fitted, see the header — gives
**ρ(C̃) = 2.534** and an **H∞ gain of 60.2**. Against the archived Float32 coefficients themselves
the pair reads 1.0002 and 10.3, which is the precision collapse of §4. Three values of ρ from one
configuration:

| coefficients | ρ(C̃) | H∞ starred gain |
|---|---|---|
| archived `LinReg1`, Float32 promoted | 1.0002 | 10.3 |
| refit, 10 TU record, Float64 | **2.534** | 60.2 |
| refit, 100 TU record, Float64 | **2.174** | 108.6 |

The precision effect (§4) and this realisation effect compound: the design has κ ≈ 1e7 and the
training target differs by ~1 `dQ` sd between the two records, so the coefficient vector is neither
well determined nor record-independent. **The aggregate conclusion is robust and the digits are
not** — quote ρ with its record and its precision, or quote it as a range.

The sweep-adequacy test's free end is therefore already reached at λ = 0 — §8b requires ρ > 1, and
every one of the Float64 fits gives 2.2–2.5 — so no additional λ points are needed at that end for
this cell.

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

## 4b. The rebaselined pipeline — R1 and R2 on the regenerated reference 🆕 2026-09-15

🔴 **These runs are NOT comparable with §3's numbers.** Everything here is on the regenerated HF
reference and a new tracking record, i.e. the post-`09954be1` Nyquist convention, which changed `∂`,
`tau` and therefore the dynamical system (memory #45, #46). §3 scores paper 2's archive against
paper 2's reference. The two tables measure different systems with the same statistic.

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

🔑 **The two top bands track ~60× worse than the other four** — means 3–5e-4 against 4–8e-6. Expected
in kind (smallest scales, least LF skill) but it is the margin that matters: the driver's gate ships
at a deliberately loose 1e-1 and the real bound should come from this table.

### R2 — four closures, 100 TU each, scored against the regenerated reference

All stable; all launched from the same initial field as the tracking record (rel diff ≤ 1.2e-16).

| model | replicas | stable | summed KS, per replica | ensemble |
|---|---|---|---|---|
| **LinReg1** (h = 5, λ = 0, `:normal`) | 5 | 5/5 | 0.586 – 1.010 | **0.836** |
| **DDN** | 5 | 5/5 | 1.285 – 1.382 | 1.332 |
| no model | 1 | 1/1 | — | 2.487 |
| Smagorinsky `c_s = 0.07` | 1 | 1/1 | — | 3.710 |

Per-band KS, ensemble:

| model | Z[0,6] | E[0,6] | Z[7,15] | E[7,15] | Z[16,32] | E[16,32] |
|---|---|---|---|---|---|---|
| LinReg1 | 0.1448 | 0.1107 | 0.1507 | 0.1513 | 0.1390 | 0.1397 |
| DDN | 0.1663 | 0.0869 | 0.1530 | 0.1679 | 0.3945 | 0.3639 |
| no model | 0.1759 | 0.0279 | 0.2120 | 0.3304 | 0.9013 | 0.8396 |
| Smagorinsky | 0.4658 | 0.3070 | 0.5461 | 0.4758 | 0.9672 | 0.9479 |

**Ordering: LRS < DDN < no model < Smagorinsky.** The LRS beats the DDN on the two top bands by ~2.6×
and is flat across all six; the DDN and both deterministic baselines fail in `[16,32]`.

### The trajectories, one figure per closure

`analysis/plot_rebaseline.jl` writes four figures, `fig8_online_<model>.png`, each showing that
closure's online QoI trajectories against the regenerated reference across all six bands, with the
marginal the KS statistic actually scores drawn beside each band.

⚠️ **Band naming, because §4b above uses the other convention.** "The two top bands" there means the
two *highest-wavenumber* ones, `Z[16,32]` and `E[16,32]`. Below they are called **the `[16,32]`
pair** and the other four **the larger-scale bands**, so that nothing turns on which end of the
spectrum "top" points at.

⚠️ **How to read them.** Regime C is free-running — nothing is replayed, `q*` comes from the solver,
and two runs launched from the same field decorrelate within an eddy turnover (~0.3 TU). So
*pointwise* agreement past the opening is not expected and its absence is not a defect. The
trajectory panel is read for the **envelope**: does the closure hold the right band of amplitudes,
does it drift, does it collapse. The marginal beside it carries the claim.

🔑 **The direction of each failure, which KS discards.** KS is a distance and has no sign, so the
ratio of means is reported beside it — it takes one line and it separates two closures that KS ranks
adjacently:

| model, mean(q) / mean(q_ref) | Z[0,6] | E[0,6] | Z[7,15] | E[7,15] | Z[16,32] | E[16,32] |
|---|---|---|---|---|---|---|
| LinReg1 | 0.941 | 0.943 | 0.926 | 0.928 | 0.911 | 0.913 |
| DDN | 0.941 | 0.967 | 0.947 | 0.941 | 0.821 | 0.842 |
| no model | 0.939 | 1.012 | 0.918 | 0.870 | **2.391** | **2.015** |
| Smagorinsky | **1.169** | **1.157** | **1.206** | **1.166** | **2.153** | **1.942** |

**The two stochastic closures are low everywhere and the two deterministic ones are high in
`[16,32]`.** Under-dissipation at the smallest resolved scales is the baselines' failure; the TO
closures have the opposite sign, and nothing in the summed KS says so.

![LinReg1 online](figures/fig8_online_LinReg1.png)

🔑 **The LRS's error is a modest low bias carried by a left tail.** Its marginal sits on the
reference's through the body of the distribution — the modes line up in all six bands — but the mean
runs **6–9% low in every band**, and the trajectories show where that comes from: downward
excursions to values the reference never visits. `Z[16,32]` reaches **181** against a reference
minimum of **721**; `E[16,32]` reaches **6.2e-3** against **0.0243**. The flat ~0.14 KS profile
across all six bands is therefore not six independent near-misses but one failure mode expressed six
times.

![DDN online](figures/fig8_online_DDN.png)

🔑 **The DDN fails differently, and only in `[16,32]`.** Its four larger-scale bands are as good as
the LRS's or better (`E[0,6]` KS 0.0869 against 0.1107, mean ratio 0.967 against 0.943), but `Z[16,32]` and
`E[16,32]` both overshoot the reference — max **7 967** against **3 791** — *and* repeatedly collapse
toward zero: minimum `Z[16,32]` **0.42** and `E[16,32]` **8.4e-6**, the latter more than three orders
below the reference's own minimum of 0.0243. The marginal is bimodal, with mass piled at the bottom
of the range where the reference has none. **These are the excursions the stabilizer catches on the
LRS**, and the DDN has no stabilizer.

![no model online](figures/fig8_online_nomodel.png)

![Smagorinsky online](figures/fig8_online_smag.png)

🔴 **The 3.710-against-2.487 ordering does not come from `[16,32]`, where KS is saturated.** Both
baselines pile up there — no model reaches `Z[16,32]` = **10 170** against the reference's median of
2 023 — and the KS values (0.84–0.97) sit where the distributions barely overlap either way, so the
statistic cannot rank them in that band. What the trajectories add is that the eddy viscosity *does*
cut the pile-up: max `Z[16,32]` **7 135** against **10 170**, mean ratio 2.15 against 2.39.

**The gap is paid in the four larger-scale bands.** Decomposing the 3.710 − 2.487 = 1.223:
**1.048 of it (86%) comes from those four** — 1.795 for Smagorinsky against 0.746 for no model —
and only 0.174 from the saturated `[16,32]` pair. No model is close to the reference there in the
mean (0.918–1.012 on three of the four, with `E[7,15]` the exception at 0.870, and `E[0,6]` KS
**0.0279**, the lowest number anywhere in this table); `c_s = 0.07` is biased **16–21% high on all
four**. A statement about this untuned `c_s` on these kernels, not about Smagorinsky.

🔴 **Three things stop this being a result yet.**

1. **The stabilizer clamp fires on the LRS and cannot fire on the DDN** (memory #59). 160 / 539 / 359
   / 465 / 736 identically-zero `dQ` columns per replica out of **40 000** — 0.4–1.8% of steps. The
   clamp lives only in the `LinReg` path (`time_series_methods.jl:162,165,190,193`); `MVG_sampler`
   never receives `q_star`. So **LinReg1's KS is model + stabilizer, and the LRS/DDN difference
   carries that asymmetry.**
   ✅ **Two things sharpened here, both by measurement rather than by re-reading.** `E[16,32]` is
   under the threshold on **100% of fired steps in all five replicas** — memory #59 checked only
   the worst one — and no other band is under it on any fired step, in any replica. And the events
   are **clustered, not a uniform tax**: 13 / 20 / 31 / 20 / 79 contiguous bursts, so replica 1's
   160 steps are 13 bursts inside a single 0.78 TU window (steps 32 728–33 040). A stabilizer that
   sits on one excursion is a different object from one that shaves 1% of every step, and only the
   burst count distinguishes them.
2. **The threshold is too close to the physics.** The reference's own `E[16,32]` minimum is 0.0243,
   only 2.4× the 1e-2 clamp. Reconstructed `|q*|` minima: LRS 6.2e-3 – 8.6e-3, DDN down to 4.57e-4.
   ⚠️ Memory #59 calls the DDN figure *"three orders below the threshold"*; the value is right and
   the characterisation is not — 1e-2 / 4.57e-4 = **22**. Three orders is the right description of a
   different quantity, the DDN's minimum on the **level**: `E[16,32]` reaches **8.4e-6** against the
   reference's own minimum of 0.0243.
3. 🔴 **LinReg1's 0.836 is ~4× §3's archived 0.198**, well outside the 0.1848 noise floor. Untested
   candidates: the clamp; the changed `tau`; the split now being a continuation of one record instead
   of two; a different reference realisation. **Not reportable until understood.**

⚠️ **Smagorinsky here is an untuned point, not paper 2's baseline.** Paper 2 tuned `c_s` against its
own reference with its own kernels and reported 0.705; this is `c_s = 0.07` on upstream's rewritten
kernels against a different reference. Never quote the two side by side.

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
    reproduce the archive; and **ρ(C̃) and the H∞ gain are record-dependent** — 2.174/108.6 on the
    100 TU refit against 2.534/60.2 on the 10 TU one. Neither `plan.md` nor `metrics.md`
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
| **#17 spread–skill vs lead, RH-3** | Needs D6 (K ≫ 1 initial conditions). Every archived run is one trajectory from one IC. P2c. |
| **DDN in regime C** | 🔴 **The DDN online runs do not exist.** `7_online_DDN.jl` writes to `output/TO_DDN/`; no such directory or file exists in either archive root, only a precomputed `ks_dists_DDN_smag_lf.jld2`. DDN's offline half — everything in §2 — is complete; its KS, stability and spread–skill would need a re-run of 5 × 100 TU ≈ 51 SBU. |
| **λ > 0 offline reproduction** | `RegularizedLeastSquares`' ADMM is not in the stdlib-only test environment, so G1's λ = 0.01 *coefficient* acceptance and the QR-vs-ADMM parity at λ ∈ {0, 0.01, 0.1} are not run. λ = 0 is reproduced for all three available configurations. ✅ G1's **online** acceptance is no longer blocked and passes for all five (§3). |
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
