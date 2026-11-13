# Card: step 5 reaching step 4 — a generator held at k1a's noise level and response; opened 2026-11-13

Paper: `sec:lever-stoch`, `sec:res-stoch`. Builds on `step5_skips.md` (B = `k1_CNN_gen_nt`, the generator on k1's frozen
mean): B passed its offline gate (−6.8 %) but drew noise ~8 % wider than k1a's (middle band 1.29 / 1.50 against 1.18 /
1.40 of k1a's clean rms) and responded ~5 % more strongly to its own draws (0.81 / 0.87 against 0.77 / 0.81); coupled, its
long runs collapsed (D 1.03 against k1a's 0.32, b −11.4 %, guard 2 %).

**Origin.** Rik, via the paper session (2026-11-13): "let the step 5 agent try to at least reach the performance of step 4
(this should be possible right?)". The generator's class contains k1a (k1's mean + a constant L ε), so an exact start at
k1a reaches step 4 by construction; the task is to train from there without leaving k1a's noise level and response,
which the offline score did not protect in B.

⚠️ **Motivated post hoc**, after step 4 and step 5 (all three skips) were scored coupled. The variants and the gate below
are chosen knowing B's failure.

🔒 **Once any coupled number of these closures exists, nothing above "Post hoc" changes.**

- [ ] Rik has seen this card (gate summaries go to the paper session; every Snellius step is Rik's go here).

## Closures (one skip: k1, the old fit `skipnth_LinReg1_nt_crps_wd1_s1`, step-4 z0 verdict KEEP)

All start at **W0**: k1's mean frozen (as B: `Cv.mean_net` + LinReg1_nt's Ws), L₀ = chol of k1a's deployed Σ (the
refitted head's sds with the correlation of `S_noisy`), the generator's output layer at zero (so g ≡ 0 at update 0).
W0 IS k1a: its mean is k1a's (B's export check: frozen mean + Ws z = k1a's deployed mean, 0.0) and its noise is L₀ ε
with L₀ L₀' = k1a's Σ. History noise in training: A2c (the generator's total sample covariance) throughout.

| variant | L | network output | loss | extra |
|---|---|---|---|---|
| W0 | k1a's | zero | — | the start; exported to test it against k1a |
| a | frozen at k1a's | g(z, q*, ε) (may move the mean) | energy score | — |
| a-c | frozen at k1a's | **centred**: g(z, q*, ε) − g(z, q*, 0) (cannot move the mean at ε = 0) | energy score | needs a deployment change (src) |
| b | frozen at k1a's | g | summed fair CRPS (step 4's loss family) | — |
| c | trainable | g | energy score | **sd cap**: penalty on the drawn sd per QoI above 1.02 × k1a's |

Each on the protocol's grid (wd {0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10} × seeds 1–3, the rule = seed-mean validation loss,
early stop 8.2–10 TU, refit 1–10 TU, seed 1 deployed). Stage 1 on k1's stage-1 network (as B).

Partner: **k1a** (`LinReg1_nt_CNN_crps_nt_headcorr`). Also reported against k1b and B (`k1_CNN_gen_nt`).

## Offline gate (new; from B's lesson). A variant passes only if all three hold, at its candidate (rule wd, seed 1)

1. the fair CRPS under its own noise (paired, 256 draws, 8.2–10 TU) at or below W0's;
2. the drawn sd per QoI at most 1.02 × k1a's (the generator's sample sd under its own noise, pooled over the validation
   rows, against √diag Σ_k1a);
3. the response to its own draws per QoI at most 1.02 × k1a's (0.51 0.42 0.77 0.81 0.42 0.57, `gen_owndraws.jl`).

🔒 **Amended 2026-11-13, after the offline grid's wave 1 and before any coupled number (Rik: "yes remove this gate it is
not based on a real issue"): gate 2 is removed.** The step-4 session's k1a noise test (k1a's mean, its sd ×1.08 / ×0.92,
5 × 100 TU each) showed that a noise level 8 % wider does not collapse the long runs (×1.08: D 0.32, b −2.6 %; ×1.00: D
0.32, b −3.3 %; ×0.92: D 0.38, b −3.8 %; side-session notes `k1a_noise_test.md`), so B's collapse is specific to its
generator noise (its correlation or its state-dependent part), not to the noise level. The gate is now gates 1 and 3; the
drawn sd, the samples' correlation against Σ_k1a and the network's share of the variance are reported beside it. Variant
c (the sd cap) was already running and is kept as a reported variant.

🔴 **Found 2026-11-13 while applying gate 3: the response measure was biased against generators.** `gen_owndraws.jl` took the
generator's clean and noisy means as two INDEPENDENT 16-draw sample means, so rms(noisy − clean) carried their Monte Carlo
noise, about 2/16 of the noise variance in quadrature, while a LinReg / step-4 comparator's mean is exact. Fixed with
`RIKFLOW_OWNDRAWS_CRN=1` (the same deviates for the clean and the noisy means) and `RIKFLOW_OWNDRAWS_M=64`; the defaults
reproduce the old tables. Gate 3 is applied with the corrected measure, and W0 (= k1a) is run through it as its check.
Every earlier generator response (step 5, s5n, s5k, s5r; Table `tab: s5-offline` columns 7–8) is biased upward the same way
and is to be recomputed.

The best passing variant (lowest CRPS under its own noise) goes coupled. If none passes, **W0 is the result** ("the
generator reaches k1a only at its start") and nothing is run coupled.

## Coupled (gpu_a100)

Long runs first: 5 × 100 TU with k1a's replica seeds (`run_online.sh lstm 2 <r>`, ≈ 125 SBU). Only if the summed KS D ≤ 0.4
(k1a 0.32, k1b 0.29): the training hindcast (defect rule against k1a) and the full hindcast. If the full hindcast does not
fit before Tue 17 Nov, long runs alone.

## Expected, and why (written 2026-11-13, before any run)

- **W0 reproduces k1a** (mean exactly, noise in distribution; the random stream differs).
- **Freezing L (a, b) does not hold the noise level by itself**: the network's ε-response adds variance (B: share ≤ 0.21)
  or cancels part of L ε. Expected drawn sd 0.95–1.08 × k1a's; some fits fail gate 2. The centred variant (a-c) keeps
  the mean exactly k1's, so whatever it learns is noise structure only.
- **The cap (c)** holds gate 2 by construction (up to the batch estimate); gate 3 (response) is not trained against and
  is the likely binding gate if the network moves the mean.
- **Offline gains are small** (< 2 % under own noise): k1a's mean is already noise-trained and its Σ is fitted; what is
  left is a state-dependent spread, which no step-5 fit has shown on k1 so far (B's spread was flat in the level).
- **Coupled, a gate-passing variant should give long runs close to k1a's (D 0.3–0.4)** and short leads within ±2 % of
  k1a's CRPS₀.₅. Expected verdict against k1a: about equal, not "improves".

**Refutation rule:** if a gate-passing variant's long runs give D > 0.4, the gate (noise level and response, offline) does
not capture what makes B's long runs collapse; no hindcast is run, and the result is reported as such. If no variant
passes, the result is W0.

## Disclosures

- A2c injects the generator's average total covariance, not its state-dependent part per window.
- Stage 1 on k1's stage-1 network over LinReg1_nt fitted on 1–10 TU (as B and k1).
- The variants and the gate were designed after B's coupled failure.

## Built

## Offline (desktop 2026-11-13; grid 11:23–11:59, `TO_LSTM/step5/gen_k_<v>_wd<wd>_s<s>`, logs `tmp/step5k/`; gate logs `tmp/step5k_gate/`)

- **W0 check: PASS.** B's fit that stopped at update 0 (wd 10, seed 2), exported with k1's frozen mean as `k1_CNN_gen_w0`
  (translation 7.6e-7; frozen mean + Ws z = k1a's deployed mean, 0.0): L₀ L₀' against k1a's Σ max rel. diff 4.1e-8
  (Float32 L); the generator's sample against k1a's mean + L₀ ε on 100 inputs, max |diff| 0.0. Through the corrected
  own-draws measure its response is 0.511 0.416 0.769 0.808 0.423 0.566 = k1a's.
- **Rule: wd 1 for every variant** (exact seed-mean validation loss; a 1.9106, a-c 1.8908, b 3.6885 (CRPS), c 1.8537; the
  other weight decays 1.888–1.97 / 3.92–4.05). Candidates = seed 1. Exports: a `74ab3d2c…`, a-c `5a6f82fe…`, b `2214cdce…`,
  c `548500a6…` (translation 6.9e-7 – 9.8e-7; frozen mean = k1a's, 0.0).

| variant (wd 1, seed 1) | gate 1: CRPS under own noise vs W0 (stage 1) | gate 3: response (corrected) | drawn sd / k1a's | corr. off Σ | mean error / k1a's | network share (refit) |
|---|---|---|---|---|---|---|
| k1a = W0 | 0 | 0.512 0.416 0.769 0.808 0.423 0.566 | 1 | 0 | 1 | 0 |
| a (L frozen) | −3.79 % pass | 0.461 0.424 0.668 0.768 0.367 0.477 pass | 1.01 0.99 1.11 0.91 1.02 1.08 | 0.22 | 1.01 1.01 1.12 1.02 1.03 1.02 | ≤ 0.14 |
| a-c (L frozen, centred) | −2.93 % pass | 0.477 0.415 0.720 0.773 0.382 0.482 pass | 0.96 1.02 1.03 1.01 0.97 1.02 | 0.14 | 1.01 1.01 1.09 1.05 1.02 1.05 | ≤ 1.2 |
| b (L frozen, CRPS) | −6.36 % pass | 0.473 0.399 0.653 0.765 0.355 0.465 pass | 1.01 1.06 1.08 0.98 1.08 1.05 | 0.37 | 1.01 1.02 1.09 1.04 1.03 1.03 | — |
| c (L trainable, sd cap) | −6.79 % pass | 0.466 **0.427** 0.722 0.816 0.382 0.515 fail by 0.003 | 0.93 0.97 0.99 0.71 0.96 0.94 | 0.17 | 1.00 1.00 1.04 0.98 1.01 1.00 | ≤ 0.11 |
| (B, collapsed coupled) | −6.83 % | 0.468 0.421 0.733 0.799 0.401 0.542 (would pass) | 1.01 1.03 1.07 1.05 1.02 1.03 | 0.15 | 1.01 1.01 1.05 1.02 1.01 0.99 | ≤ 0.21 |

  (drawn sd and mean error from the corrected own-draws run: 64-draw means, common random numbers; units k1a's clean
  rms. The 64-draw mean adds ~0.8 % to the error by itself.)
- 🔴 **B passes gates 1 and 3 when measured correctly**: its response is not stronger than k1a's (0.73 / 0.80 against
  0.77 / 0.81 in the middle band; the earlier 0.81 / 0.87 was the measurement bias). So the offline gate does not separate
  the variants from B, whose long runs collapsed; nothing offline here predicts the long runs. Rik decides which go
  coupled (asked 2026-11-13).

## Post hoc
