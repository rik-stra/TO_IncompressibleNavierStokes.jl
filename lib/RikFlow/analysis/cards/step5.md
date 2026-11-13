# Card: step 5 — stochastic skip network (paper `sec:lever-stoch`), opened 2026-11-06

**Amended 2026-11-06 (Rik, 2026-11-06: "actually the new step 5 only really covers the old step 6. I suggest we keep the old step 5 as optional, move it to the appendix for now.")** This closure replaces step 6 only. The old step 5 (5a–5c) is
kept as an optional variant, not run, in the paper's App. `app:lever-state`. The methods text is merged into
`closures.tex` (`sec:lever-stoch`); Rik has not yet read it. The paragraphs below that say "replaces the old step 5"
are superseded by this note.

WORKFLOW.md Stage 0. **Methods text drafted, not merged:** `paper_tighten/sections/closures_step5_draft.tex`
(2026-11-06; Eq. `eq: stochastic net`; merge notes for the paper session at its top). It replaces `closures.tex`'s Step 5
(the old LSTM head, 5a–5c) and Step 6. It compiles in a scratch copy of the paper with no errors, no undefined or
duplicate labels, and no swallowed text. Plan: `paper_tighten/claim_maps/runs_to_finalize.md` ("Step 5, new").

**This card replaces:**
- the old step 5: 5a/5a₀, 5b/5b₀ and 5c/5c₀, an LSTM state-dependent scale on frozen linear means;
- step 6a's closure g1 (card `step6a.md`): this is that card's D5 variant, the generator on a linear skip;
- 6a_nt, which becomes the optional s5_nt below.

**Origin** (discussion with Rik, 2026-11-06):
- "I was thinking of replacing step 5 by a stochastic skip network. It should have the linear branch, the CNN and should
  concatenate a vector of random numbers. Such that it produces a random sample imediatly, this can then be trained
  with CRPS? Likelihood does not work right?"
- On the noise skip: "keep it and make L fully learnable".
- On the sketch: "This looks good log it as the new step 5".

⚠️ **Motivated post hoc.** It was proposed after steps 1–3c were scored coupled and after step 4 was scored offline
(sequential design, paper Sec. 3.5). No evaluation-window result of this network exists.

🔒 **Once any full-hindcast number of this closure exists, nothing above "Post hoc" changes.**

- [x] Rik approved the architecture (2026-11-06; quotes above)
- [x] Rik decided items (a)–(d) at the defaults (2026-11-06, asked "Shall I record them as decided?": "yes")
- [x] **GATE 0 passed 2026-11-06** (Rik, asked to read `sec:lever-stoch` and this card: "Read; use LinReg1 + MVG"). The
  offline check's pass reference is **LinReg1 + MVG on the same steps** (the same mean, so only the learned residual
  differs); least squares with its constant Gaussian refitted on 1–8.2 TU is reported beside it. The methods text said
  least squares; the paper session changes it. Item (c) checked: left out (below). Stage 1 started; step 4's CNN skip
  is committed (`a38aec13`).

## Closure

| run | closure | the one change | matched partner | artefact |
|---|---|---|---|---|
| s5 | stochastic skip network (`LinReg1_CNN_gen`, name provisional) | the residual is learned by a generator on LinReg1's frozen mean, instead of a constant Gaussian | LinReg1 (`D6_LinReg1`, `D6train_LinReg1`, its long runs; reused) | not built |
| s5_nt (**optional**) | the same, trained with its own noise on LinReg1_nt's skip (6a_nt's construction) | noise training of the generator | k1 (step 4, `step4.md`) | not built; only if k1 improves on LinReg1_nt coupled |

LinReg1 + power law (`LinReg1_pl`, step 2) is reported beside s5 as the fixed-form scale, without a clause. With
LinReg1 as partner, the partner comparison and the criterion against the unregularized baseline coincide.

## Model

In standardized correction units (target = dQ, as step 3a):

    ŷⁿ = W_s zⁿ + b + g_θ(zⁿ, q*ⁿ, εⁿ) + L εⁿ,    εⁿ ~ N(0, I₆) i.i.d. in time,    q̂ⁿ = q*ⁿ + D ŷⁿ

- **Linear skip.** W_s zⁿ + b is LinReg1's map, frozen, written on the CNN's mapped input zⁿ (97 inputs, h = 5). It is
  step 4's `linreg_skip_map` (W_s is 6 × 97, plus the bias).
  - 🔴 **Deploy in Float64.** LinReg1 is a least-squares map. LinReg1_nt's map reaches 9.4e3 target sd on the mapped
    input, and in Float32 the deployed skip was 4.6e-3 sd off (step 4, `claude_memory.md`). LinReg1's own map must
    be checked the same way.
- **CNN g_θ** is step 3a's: input map 97 → 5 steps × 18 channels; two Conv1D layers (kernel 2, 16 channels, tanh);
  flatten; [·; q*ⁿ; εⁿ] → Dense 64 tanh → Dense 64 tanh → Dense 6.
  - The output layer starts at zero (as step 4).
  - The six deviates enter the first dense layer only, beside q*ⁿ, which adds 6 × 64 = 384 weights.
  - 9 190 + 384 trainable weights ((c) checked: the skip's prediction is not an input).
- **Noise skip L εⁿ.** L is 6 × 6 lower triangular (21 parameters). It starts at chol(Σ) of LinReg1's MVG, in
  standardized target units, and is **fully learnable** (Rik), with no weight decay.
- **At the start** (g_θ ≡ 0), the closure is exactly LinReg1 + MVG. This must be checked by a test.
- **Nothing is added after the network,** and no Σ is fitted after training.

## Decided

- **D-A Replace the old step 5 and 6a** by this closure (Rik, 2026-11-06). Amended the same day: 6a only; the old
  step 5 stays as an optional variant (note at the top).
- **D-B The noise skip L ε is kept; L is fully learnable** (Rik, 2026-11-06). Reasons:
  - the closure starts at its partner;
  - it brings in LinReg1's correlation between the QoIs, which the marginal CRPS does not constrain;
  - an early stop leaves the closure at about LinReg1 + MVG, not at too little spread;
  - it separates the constant part (L) from the learned, state-dependent part (the CNN's response to ε).

  The consequence of a fully learnable L: the correlation can drift under the marginal CRPS. Offline, report the
  correlation of the samples and of L Lᵀ against Σ. If it misses Σ badly, Rik decides on a variant before any coupled
  run.
- **D-C No likelihood.** With the noise inside the network, the density of q̂ has no closed form. The loss is
  sample-based (6a card D6).
- **Carried over from `step6a.md`** (Rik accepted its defaults on 2026-10-29):
  - D2: M = 4 samples per training step, pure fair CRPS; validation with one fixed set of draws.
  - D3: current-step deviates in the first dense layer.
  - D4: N_Q = 6 deviates from the member's stream before the guard, so the random numbers are common with LinReg1.
  - D7: random init (except the zero output layer and L at chol Σ), seeds 1–3, seed 1 coupled.
  - D8: settings by the rules of `sec:closures-training` (Adam, lr 1e-3, plateau schedule, batch 256, weight-decay
    grid {0, 0.01, 0.03, 0.1, 0.3, 1} by the seed-mean validation CRPS, early stop on 8.2–10 TU, refit on 1–10 TU).
    Weight decay acts on the CNN, the noise weights included, not on L.
  - D9: a failed offline check is reported, and the closure is still run.
  - D5 is flipped: **with** the linear map (LinReg1's).

## (a)–(d): decided at the defaults (in bold; Rik, 2026-11-06: "yes"), alternatives kept for the record

- **(a) Base.** **The clean LinReg1 skip, with partner LinReg1** | LinReg1_nt's skip trained with its own noise, with
  partner k1 (= s5_nt).
  - The default gives the cleanest pair: one change, and the partner's runs are reused.
  - In step 4, the clean CRPS network on this skip stayed at update 0, so the mean stays about LinReg1's.
- 🔒 **(b) amended at GATE 1 (Rik, 2026-11-06, shown the grid below: "Energy score, rerun"): the fair ENERGY SCORE of
  M = 4 samples of the six-QoI standardized correction** (`RF.energy_fair_samples`, `RIKFLOW_CNN_GEN_SCORE=energy`, the
  default since), because the summed marginal CRPS let the network couple the bands (correlations ±0.9 where Σ has
  ≤ 0.09). Everything else unchanged; the grid is rerun. The offline check stays the CRPS against LinReg1 + MVG (the
  methods text); the energy score on the same draws is printed beside it.
- **(b) Loss, as first decided.** **Σ over the six QoIs of the fair CRPS of the standardized correction** | the same plus a variogram
  term for the correlation, with its weight fixed before training.
  - The default is 6a's D1 and the evaluation's score per QoI.
  - The noise skip carries the correlation in from Σ.
- **(c) The skip's prediction W_s z as an extra input** to the first dense layer, scaled to unit sd on 1–10 TU.
  - **First the offline check:** LinReg1's residual sd on 10–100 TU, binned by its predicted correction at a given
    level, next to Fig. `colour-scale`. No training is needed.
  - **If the sd depends on the predicted correction:** train with and without the input, and choose by the seed-mean
    validation CRPS (rule fixed here before training). **If not, leave it out.**
  - Risk: online, the skip's prediction responds about 2.4 times to the closure's own draws in the middle band (`tab:
    skip-offline`). A spread that grows with it feeds back on the draws. If it is included, report the spread's
    response to its own draws.
  - **Check run 2026-11-06: no dependence beyond the level → left out** (the rule above). `analysis/skip_pred_scale.jl
    LinReg1 --fig` (figure `analysis/figures/skip_pred_scale_LinReg1.png`, gitignored). Fits on 1–10 TU, read on 10–100 TU:
    - log-variance model, Gaussian about zero: step 2's power law over a constant gains 0.483 nats/step summed (the
      0.48 of Sec. 5.2); adding the predicted correction c and c² to the power law gains **0.002** (per QoI −0.001 to
      +0.002; 0.005 in sample);
    - the residual over the power law's sd, per decile of c: spread 0.94–1.10 of its overall spread, a mild U (the
      outer deciles 1.03–1.10) in the large scales, flat elsewhere;
    - binned (level quintile × c quintile within it), the spread / the level bin's: 0.92–1.11, except the small scales'
      lowest and highest level quintiles (down to 0.60), where c and the level are the same variable (corr(c, log q*)
      −0.96/−0.95 on 1–10 TU) and the power law absorbs it.
    So the methods sentence on reading the prediction (closures.tex `sec:lever-stoch`) is to be cut (paper session told).
- **(d) May the CNN also move the mean?** **Yes**, through the part of g_θ that does not depend on ε, as in step 4 |
  no: centre the network on ε by using g_θ(z, q*, ε) − g_θ(z, q*, 0).

## Expected, and why

- **Offline:** a CRPS on 8.2–10 TU a little below LinReg1 + MVG's. The residual's scale depends on the level (Sec. 5.2;
  step 2's power law gained 0.48 nats per step offline). The samples' spread should grow with the level, about as the
  power law does. The mean should stay about LinReg1's.
- **Coupled against LinReg1:** as step 2 on LinReg1, narrower, state-dependent ensembles and fewer guard firings at
  about the same skill. The network is trained on the recorded history, so it keeps LinReg1's response to its own
  draws, and the long runs may stay over-dispersed (C1c).
- **Expected verdict against LinReg1: does not improve.** The case for running it is the story: does a residual learned
  inside a network, on the best linear mean, do more than step 2's fixed form? It also carries the title's
  "conditional density" (S7) and intro contribution 3.
- **s5_nt (optional):** as steps 2b and 3c, the response to its own draws largely removed, narrower short-lead
  ensembles, better long runs.

## Clauses (fixed: paper Sec. 3.5 `sec:criterion`; as `step6a.md`)

Hindcast K = 90, M = 10, 100 + 1200 steps; CRPS₀.₅ (steps 1–200), CRPS_all (1–1200); e₀.₅; r, the calibrated count and
the rank histograms per lead (36 cells); the lead-wise divergence rule; long runs 5 × 100 TU.
- **G:** the KS guard against **LinReg7**. FAIL iff all five single-run summed KS distances lie above all five of
  LinReg7's.
- **S:** the CRPS₀.₅ paired difference against LinReg1, with the 90 % CI upper end < 0.
- **C:** the calibrated cells against LinReg1, with the paired CI of the count difference > 0, AND the CRPS₀.₅ CI upper
  end < +10 %.
- **Verdict:** IMPROVES iff G passes and (S or C). s5_nt against k1: the same rules with k1 in place of both baselines.

Reported, without a clause:
- as `step3a.md`;
- beside LinReg1_pl (step 2);
- the offline numbers below.

## Offline check and what is reported beside it

- **Check:** the CRPS on 8.2–10 TU (after training on 1–8.2 TU) against LinReg1 + MVG's closed-form CRPS on the same
  steps. It passes when it reaches it, with no margin (6a D9, with LinReg1 + MVG in place of least squares; Rik at
  GATE 0). Beside it, least squares with its constant Gaussian refitted on 1–8.2 TU (`LS_CRPS` in `cnn_mean_fit.jl`).
- **Reported beside it:**
  - the samples' sd against their rms error per QoI;
  - the samples' sd per tenth of the level (as Fig. `colour-scale`, with the power law beside it);
  - the share of the sample variance from the CNN's response to ε rather than from L, and how it varies with the state;
  - the samples' correlation and L Lᵀ's against Σ (D-B);
  - the error of the samples' mean against LinReg1's;
  - the response to its own draws and the error on noisy inputs (as `tab: own-draws`);
  - for (c), the spread's response to the skip's prediction.

## Implementation (Stage 1, after GATE 0 and after step 4's commit)

- **`cnn_mean_fit.jl`:**
  - `RIKFLOW_CNN_LOSS=crps_gen` (name provisional): noise inputs in the first dense layer, M draws per window, the
    sample fair CRPS;
  - the noise skip L (lower triangular, from chol Σ of a deployed LinReg, no weight decay);
  - the frozen skip (`RIKFLOW_MLP_SKIP=LinReg1`, as step 4);
  - fixed validation draws, and the second training as now;
  - option (c) if chosen.
- **`StochLSTM` (`arch = :cnn`, with `skip` from step 4):** an emission mode in which the six deviates are drawn from
  the member's rng before the forward pass, fed to the network and to L, with nothing added after. The old modes stay
  bit-identical, random stream included. Float64.
- **`cnn_closure_export.jl`:** export the generator. The translation check compares the fit with the deployed closure
  using the same εⁿ.
- **Tests** (next free V-number: check the branch first, #74):
  - the sample CRPS against `crps_ensemble(fair = true)`;
  - with g_θ ≡ 0 the deployed closure reproduces LinReg1 + MVG exactly, step by step, on the same seed;
  - six deviates are drawn per step from the member's stream before the guard;
  - zero noise weights and L = 0 give a deterministic closure.
- **s5_nt only (optional):** 6a_nt's own-draw construction (`step6a.md` D10) on LinReg1_nt's skip.

### Built (Stage 1, 2026-11-06; uncommitted, Rik commits)

| methods text (`sec:lever-stoch`) / card item | code | test or check |
|---|---|---|
| ŷ = W_s z + b + g_θ(z, q*, ε) + L ε, deployed | `LSTMSpec` `emission = :generator` (`:cnn` only), `is_generator`; `cnn_window!(…; eps)` appends ε to [conv features; q*] and adds `L ε` (`L = LR`, lower), `bd = Wd = 0` | V98 spec / forward-pass items (`test/test_cnn_generator.jl`) |
| W_s z + b: LinReg1's map, frozen, Float64 | step 4's `linreg_skip_map`, `cnn_closure_export.jl … gen … skip=LinReg1` (Float64 with a skip) | V95; export: Ws vs LinReg1's own correction on every record step |
| ε joins q* in the first dense layer; output layer starts at zero | `cnn_mean_fit.jl` `mkmodel_gen`: `Parallel(vcat, conv, NoOp, NoOp)` on (A, q*, ε), `Dense(nflat + 2N_Q => 64)`; output weights `zeros32` AND output bias `zeros32` (🔴 Lux 1.x's default bias is PyTorch's uniform ±1/√64: with OUTINIT alone the start was off by up to 0.11 target sd, found by the smoke; steps 3–4's `mkmodel` have that default, not changed here) | `fit_gen` asserts at update 0 that the samples are L0 ε exactly; export translation check (deployed vs Lux, the fit's own ε, 300 steps of 10–100 TU): smoke max 2.2e-7, rms 2.9e-8 target sd |
| L lower triangular, starts at chol Σ of LinReg1, fully learnable, no weight decay | `ps = (; net, L)`, `gen_L(p) = L .* tril`, `L0 = chol(diag(σ_out/so) Σ diag(σ_out/so))`; `adjust!(opt.L; lambda = 0)` | fit prints L sd / L0 sd |
| at the start exactly LinReg1 + MVG | target centred at LinReg1's MVG mean (`RIKFLOW_MLP_SKIP_CENTER=mvg`, step3_data.jl; required by crps_gen) | V98: the Float64 generator at its start vs the real `LinReg` path, same seed, step by step: max 1.5e-7 sd (synthetic), 2.8e-6 sd (LinReg1's file; LinReg rounds its level to Float32, #77) |
| N_Q deviates per step from the member's stream, as LinReg + MVG | `StochLSTM` generator branch: N_Q scalar `randn` before the forward pass; none in the warm-up; nothing added after (`ar_phi`, `scale_*`, `eta_ar`, `noise_scale`, `emission_scale`, `dq_offset` refused) | V98: stream position after a run; gated steps; refusals; `rand(g.rng) === rand(l.rng)` against LinReg |
| loss: fair energy score of M = 4 samples (GATE 1; first: fair CRPS per QoI, summed) | `RF.energy_fair_samples` / `RF.crps_fair_samples` (ts_score.jl), `RIKFLOW_CNN_GEN_SCORE` = energy (default) / crps; fresh ε per update from the noise stream | V98: the energy score by explicit sums, = the fair CRPS for N_Q = 1, prefers the right correlation where the summed CRPS cannot tell; the CRPS = `crps_ensemble(fair = true)` summed / averaged |
| validation: one fixed set of draws | `Eva` fixed per seed, **M_val = 16** draws per row (an implementation choice, unbiased like M = 4, less noisy for the early stop) | — |
| weight decay on the network (noise weights included), not on L; grid, 3 seeds, early stop, refit | `fit_gen` / `refit_gen` (the protocol's loop, written out for the generator) | smoke on the desktop |
| nothing fitted after training | no Σ refit in crps_gen | — |
| offline check vs LinReg1 + MVG (Rik, GATE 0) and LS beside it | fit prints CHECK: the fair CRPS of M = 256 fixed draws per row on 8.2–10 TU against LinReg1 + MVG **on the same draws** (the start model, g = 0, L = L0, is LinReg1 + MVG: a paired comparison without Monte Carlo noise; PASS iff not above it); the closed form beside it (any sample estimate differs from it by Monte Carlo noise: sd 0.5 / 0.22 / 0.11 % at M = 16 / 64 / 256, scratch simulation, unbiased); LS refit `LS_CRPS` | smoke: start = LinReg1 + MVG on the same draws exactly (3.4082 = 3.4082) |
| reported beside: spread vs error, vs level, net share, correlation | fit prints them on 8.2–10 TU; `analysis/gen_offline.jl` per decile of the level beside the power law (from `pred.jld2`) | — |
| zero noise weights, L = 0 → deterministic | — | V98 |

Test ID **V98** (V96 unused, V97 is the q1-q2 session's).

### Offline: the CPU grid (desktop, 2026-11-06 15:32–15:38; fits `TO_LSTM/step5/gen_wd<wd>_s<s>`, logs
`the desktop's tmp/step5_grid/`; `analysis/gen_offline.jl`)

- **Grid** (6 weight decays × 3 seeds, 8 000–13 800 updates each, ~27 ms per update):
  - wd 0, 0.01, 0.03, 0.1, 0.3: **every seed stops at update 0**, i.e. the closure IS LinReg1 + MVG (validation CRPS
    = the start's).
  - wd 1: seeds 1 and 3 learn (best at update 5 800 / 5 700), seed 2 stays at update 0.
  - Rule (seed-mean validation fair CRPS, 16 fixed draws): wd 1 (3.370 against 3.398 for every other wd) — the edge
    of the grid.
- **Check** (stage 1 on 1–8.2 TU, 8.2–10 TU, 256 fixed draws, against LinReg1 + MVG on the same draws): seed 1
  3.4066 against 3.4082 (**PASS by 0.05 %**), seed 3 3.3900 against 3.4038 (−0.4 %), seed 2 = the start. The 16-draw
  validation loss that chose the iterate flatters it (seed 1 −0.75 %, seed 3 −1.7 %). LS refitted on 1–8.2 TU 3.4970.
- **Spread against the level** (seed 1 refit, 10–100 TU, per decile of q*/q_ref, over LinReg1 + MVG's constant sd):
  Z[7,15] 0.51 → 1.12 (power law 0.39 → 1.49, LinReg1's residual spread 0.42 → 1.65); the small band alike
  (0.52 → 1.09); the large band 0.89 → 1.00. The generator learns the level dependence, flatter than the power law at
  both ends.
- **Share of the sample variance from the network's part**: 0.11 (large band), 0.40–0.46 (middle, small); L shrinks to
  0.47–0.86 of L0's sd.
- **Spread against error** (10–100 TU): error / sd 1.11–1.14 (LinReg1 + MVG 1.05–1.08): narrower, under-dispersed.
  The mean's error / LinReg1's 1.013–1.024 (sampling noise of a 32-draw mean alone gives 1.016).
- 🔴 **Correlation (D-B): misses Σ badly.** Max |samples − Σ| on 10–100 TU 1.07 (seed 1), 0.91 (seed 3); L Lᵀ 0.31.
  Σ's correlations between bands are ≤ 0.09; the samples' are ±0.5–0.98, with a seed-dependent sign: seed 1 middle–
  small −0.97, large–middle +0.55; seed 3 middle–small +0.90, large–middle −0.68. The network puts its noise response on
  one common direction across bands, which the summed marginal CRPS does not see. **By D-B, Rik decides on a variant
  before any coupled run (GATE 1, asked 2026-11-06).**

### Offline: the energy-score grid (Rik at GATE 1, "Energy score, rerun"; desktop 2026-11-06 15:48–15:55; fits
`TO_LSTM/step5/gen_es_wd<wd>_s<s>`, logs `the desktop's tmp/step5_grid_es/`)

- **All 18 fits stop at update 0**: every weight decay, every seed. The closure is LinReg1 + MVG exactly (check PASS
  trivially, energy score +0.00 %, correlation = Σ's, network share 0).
- The validation energy score rises from the first 100 updates: wd 0 1.630 → 1.952 at 5k, wd 0.1 → 1.883 at 8k; wd 1
  hovers 0.5–1 % above the start (1.637–1.651). The marginal-CRPS wd-1 fits had dipped below the start only by
  fluctuation (seed 1 at 3k: 3.386 against 3.411; their honest 256-draw gain was 0.05 %).
- **Why (diagnosis, 2026-11-06):** LinReg1, the frozen skip, is fitted on all of 1–10 TU, so the early-stopping block
  (8.2–10 TU) is in its training window. A correction of its mean learned on 1–8.2 TU is then anti-correlated with its
  residual on 8.2–10 TU: least squares on the same input fitted on 1–8.2 TU scores CRPS 3.4970 on 8.2–10 TU against
  LinReg1 + MVG's 3.4082 (**+2.6 %**). Any move of the network's mean looks worse on validation, so the early stop
  returns the start. The methods' "the effect is small: residuals ~1 % too small in sd" holds for the size of the
  residual, not for this. (Step 4's clean CRPS network on the same skip stopped at update 0 too.)
- **Asked of Rik at GATE 1 (2026-11-06)**: (a) accept — at the protocol's settings step 5 is LinReg1 + MVG, reported
  offline, no coupled run (it would reproduce LinReg1); (b) card (d)'s alternative — centre the network on ε,
  g(z, q*, ε) − g(z, q*, 0), so it cannot move the mean and learns only the residual; (c) stage 1 on LinReg1 refitted on
  1–8.2 TU as the skip (a held-out early stop), the refit on LinReg1.

### Held-out early stop (Rik at GATE 1, 2026-11-06: "please try this: Held-out early stop"; then "If that does not
work; try using random intialization instead of 0")

- 🔒 **Protocol change for step 5** (Rik notes it affects step 4 too; step 4 is not changed here): stage 1 (settings,
  early stop, offline check) uses as its skip LinReg1 + least squares of LinReg1's residual on the fit rows (together:
  least squares on 1–8.2 TU), with that model's residual as the target and L starting at the Cholesky factor of its
  fit-row residual covariance (about zero). Stage 1 thus starts as least squares on 1–8.2 TU + its MVG, and 8.2–10 TU
  is held out from everything stage 1 uses. The refit (stage 2) is unchanged: LinReg1's skip and target, L0 from
  LinReg1's Σ, so the deployed closure still starts as LinReg1 + MVG. Code: `RIKFLOW_CNN_GEN_STAGE1=heldout`.
- The check then compares with stage 1's start on the same draws (paired; at the smoke 3.5059, closed form 3.4970 =
  `LS_CRPS`); LinReg1 + MVG's closed form (3.3994, in-sample on 8.2–10 TU) is printed beside it. ⚠️ The pass reference
  Rik set at GATE 0 (LinReg1 + MVG) is in-sample there by 2.6 %; which reference decides the check is for Rik.
- Fallback built, not run (not needed): `RIKFLOW_CNN_GEN_OUTINIT=random` (Glorot output weights, bias zero; the start
  check skipped).
- **Grid, held-out + energy score** (desktop 16:01–16:05; `TO_LSTM/step5/gen_ho_wd<wd>_s<s>`, logs
  `the desktop's tmp/step5_grid_ho/`): wd 0–0.3 every seed at update 0 again (the validation score rises from
  the first 1k updates: wd 0 1.678 → 1.972 at 5k); **wd 1 all three seeds learn** (best at 1 300 / 2 500 / 3 600
  updates). Rule: **wd 1** (seed-mean validation ES 1.657 against 1.669), the edge of the grid as in both earlier grids.
- **Check, wd 1** (256 draws, same draws as the held-out start = LS on 1–8.2 TU + its MVG): CRPS seed 1 **3.4764
  against 3.5059 (−0.84 %, PASS)**, seed 2 3.4808 / 3.4968 (−0.46 %), seed 3 3.4742 / 3.5007 (−0.76 %); energy score
  −0.59 / −0.28 / −0.59 %. Against LinReg1 + MVG's closed form (3.3994, in-sample there by 2.6 %): above, all seeds.
- **Seed 1 refit, 10–100 TU** (`gen_offline.jl`): the spread follows the level — Z[7,15] 0.50 → 1.14 across deciles
  (power law 0.39 → 1.49, LinReg1's residual 0.42 → 1.65), small band 0.62 → 1.13, large band 0.85 → 0.96 (power law
  0.81 → 1.13); correlation kept (max |samples − Σ| 0.074, 1–10 TU 0.062; the worst large–middle +0.11–0.14 against
  0.04–0.09); the network's share of the sample variance 0.03 / 0.14 / 0.07 (large / middle / small), up to 0.89 in the
  lowest decile of Z[7,15], where it narrows L ε; L L' sd 0.91–0.97 of Σ's; the mean is LinReg1's (error 1.013–1.018 of
  LinReg1's, the 32-draw sampling noise alone 1.016); error / sd 1.07–1.17 (LinReg1 + MVG 1.04–1.08): narrower,
  the large band under-dispersed at every level (spread 0.85–0.96).
- **Exported 2026-11-06 (desktop build copy, code = f94f4217 + the step-5 files, Float64)**:
  `cnn_closure_export.jl step5/gen_ho_wd1_s1 gen LinReg1_CNN_gen skip=LinReg1` → `TO_LSTM/step3/closures/LinReg1_CNN_gen/
  StochLSTM_seed1.jld2` (163 978 bytes, sha256 `19ce5462137f25cd1af78a48b7f032b2768abeb50805b62a25b12d5bb844d714`); Ws
  against LinReg1's own correction on 39 596 steps 1.1e-13; translation check (the fit's own deviates, 300 steps of
  10–100 TU) max 3.8e-7, rms 8.2e-8 target sd. Log `TO_LSTM/step3/closures/export_step5_build.log`.
- **Training loss on 1–10 TU** (Rik asked; `RIKFLOW_CNN_GEN_TRAINLOG=20000`, the refit's setup at constant lr 1e-3, = the
  refit up to its update count; logs `the desktop's tmp/step5_trainlog/`): wd 1 beats LinReg1 + MVG on its
  training data by ~2 % in energy score within ~1 000 updates and stays there to 20 000 (−1.8 to −2.5 %; at the refit's
  update: seed 1 −2.4 %, CRPS −2.7 %); 10–100 TU −0.5 to −0.8 %. wd 0 memorises (−65 % training, +101 % held out at
  20 000 updates).
- 🔒 **Grid extended by wd 3 and 10** (Rik, 2026-11-06: "also add two stronger wait decays to the grid to check if they
  do better (with early stopping)"; held-out + energy score; `gen_ho_wd{3,10}_s{1,2,3}`, logs
  `the desktop's tmp/step5_grid_ho2/`): wd 10 every seed at update 0; wd 3 all seeds learn (best at 10 400 /
  8 500 / 1 100; check −0.68 / −0.57 / −0.68 % CRPS against the held-out start). Seed-mean validation ES: wd 1 1.6570,
  wd 3 1.6585, the rest 1.6692 → **the rule still picks wd 1**, now inside the grid. The candidate stays
  `LinReg1_CNN_gen` (wd 1, seed 1).
- Tests at GATE 1: V98 81/81; stdlib suite 3745 pass, 1 fail = the pre-existing V36 (`m4_control.jl`); Lux suite
  933/933 (desktop copy).

### GATE 1

- [x] **GATE 1 passed 2026-11-06** (Rik, asked "Go to Snellius with LinReg1_CNN_gen, judging the check against the
  held-out start?": "Yes, go"). 🔒 The offline check's pass reference is **the held-out start** (least squares on
  1–8.2 TU + its constant Gaussian, same draws), replacing GATE 0's LinReg1 + MVG, which is in-sample on 8.2–10 TU once
  stage 1 is held out. Decided at GATE 1 as well: the energy score (b), the held-out early stop, the grid with wd 3 and
  10. The closure: `LinReg1_CNN_gen` = `step5/gen_ho_wd1_s1` (wd 1, seed 1, refit 1 300 updates), partner LinReg1.

## Runs (as step 3a no-wd; gpu_a100)

- Training hindcast (the defect rule against LinReg1), the long-run smoke, the full hindcast (`run_d6_packed.sh`) and
  5 long runs (`run_online.sh lstm 2 <r>`, `-t 00:30:00`).
- Cost: the four Snellius jobs per closure, plus the CPU grid (six weight decays × three seeds, doubled if (c) is tried; CRPS on CPU about 3.5 ms
  per update). Storage ≈ 1.6 GB.

## Post hoc

- 2026-11-06: code committed as `cb189c9f` (Rik: "Yes, commit + push"; only the step-5 files, plus the PARTNER entry
  in `train_hindcast.jl`) and pushed (origin `cd759ad3` → `cb189c9f`, with Rik's `f94f4217`, `8e1ed477`, `ca84c79e`).
  Snellius fast-forwarded `cd759ad3` → `cb189c9f`; its `TO_LRS/LinReg1/LinReg.jld2` sha256 `4e94fb7f…b8ba12` = the
  laptop's and the desktop's. Artefact copied laptop → Snellius, sha256 `19ce5462…d714` equal on desktop, laptop and
  Snellius.
- Jobs (gpu_a100, at `cb189c9f`): training hindcast `28947513` (array 1, `run_d6_train_lstm.sh
  step3/closures/LinReg1_CNN_gen`), long-run smoke `28947514`.
- ⚠️ **Production submitted BEFORE the defect rule, chained on success** (Rik, 2026-11-06: "also submit the longruns and
  full hindcasts (with dependency on these)"): `--dependency=afterok:28947513:28947514`, at `cb189c9f`, gpu_a100. Full
  hindcast (`run_d6_packed.sh`, array 1–10, `D6_OUT=output/D6_LinReg1_CNN_gen`) `28947607`; long runs (`run_online.sh
  lstm 2 <r>`, `-t 00:30:00`, `RIKFLOW_M4_MODEL_DIR=output/TO_LSTM/step3/closures/LinReg1_CNN_gen`) r = 1–5:
  `28947609`, `28947611`, `28947612`, `28947613`, `28947614`. The defect rule is checked on the training hindcast as
  soon as it lands; if it flags, Rik decides whether to cancel.
- Long-run smoke `28947514` COMPLETED (2:34, `cb189c9f`): `PASS: 1 closure(s)` — complete (200 of 200 steps), warm-up
  dQ bit-identical to the record, q and dQ finite, the guard fired 0 / 100 steps after the warm-up, model file the fit
  dir's.
- **2026-11-11, scored.** All 16 job records COMPLETED (training hindcast 17:46; hindcast tasks 30–34 min; long runs
  10.5 min each). Copied to the laptop: 50 + 900 files, summed file bytes equal on both sides (9 880 650 / 177 852 810);
  the 5 replicas' sha256 equal (`b85a464d…`, `866febc2…`, `eca1eec6…`, `7013f86f…`, `eaa7534b…`). Snellius home 83.6 %.
  Scratch logs (session scratchpad): `train_step5.log`, `full_step5.log`, `score_step5.log` (driver `score_step5.jl`,
  `paired_compare` × 3), `longrun_step5.log`, `gate_hind_step5.log`, `gate_long_step5.log`.
- **Defect rule** (training hindcast `D6train_LinReg1_CNN_gen`, `train_hindcast.jl LinReg1_CNN_gen LinReg1`): **none
  flagged** — no member diverged; spread-skill ≤ 0.5 TU 0.73–0.81 (LinReg1 0.61–0.78), ratio to the partner 1.18 1.10
  1.07 1.08 1.03 1.04; guard 0 of 60 000 forecast steps (LinReg1 246); bias at 0.5 TU −0.3 % (LinReg1 −6.1 %), at 2.5 TU
  +5.4 % (+2.0 %).

  | closure | CRPS_0.5 | CRPS_3 | e_0.5 | cal. | b_0.5 | b_2.5 | guard | b | D | D range | CRPS_100 | e_100 | SS ratio | guard |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | LinReg1 (λ = 0) | 0.221 | 0.390 | 0.455 | 28 | −2.9 | −5.9 | 0.42 | −7.3 | 0.84 | 0.59–1.01 | 0.387 | 0.649 | 1.53 | 1.13 |
  | LinReg1 + power-law scale (step 2) | 0.220 | 0.393 | 0.455 | 27 | −2.8 | −5.0 | 0.20 | −6.1 | 0.79 | 0.57–0.95 | 0.402 | 0.684 | 1.36 | 0.41 |
  | **s5 LinReg1 + stochastic CNN (`LinReg1_CNN_gen`)** | 0.275 | 0.404 | 0.543 | 14 | +2.8 | −0.6 | <0.01 | −1.4 | 0.48 | 0.42–0.68 | 0.381 | 0.642 | 1.35 | 0 |
  | LinReg7 (λ = 1) | 0.263 | 0.408 | 0.498 | 5 | −4.0 | −5.1 | 0 | −4.9 | 0.65 | 0.54–0.91 | 0.386 | 0.691 | 0.99 | 0 |

  Six digits: CRPS_0.5 0.27452 (LinReg1 0.22079, LinReg1_pl 0.22016); CRPS_3 0.40394 (paired with LinReg1_pl, where no
  member diverges; 0.40360 paired with LinReg1); e_0.5 0.5426. Guard: hindcast 5 of 1 080 000 steps (1 member, 1 IC, none
  in leads 1–200; LinReg1 4 498, 35 members); long runs 0 of 199 500 (LinReg1 2 259, LinReg1_pl 814). No member diverged
  (LinReg1: 3). Hindcast ensembles against LinReg1's (`--full`, ≤ 0.5 TU): spread 0.84–0.91 ×, error of the ensemble
  mean 1.15–1.20 ×; r at leads 25–200 0.62–0.98 (LinReg1 0.78–1.07), at 400–1000 0.97–1.36 (1.14–1.48). Long runs: per-QoI
  spread–skill 1.34–1.36 (LinReg1 1.49–1.59), one run with b_i of both signs.

  Paired differences (90 % IC-block bootstrap, 10 000 resamples; % of B). G: the single-run D range 0.42–0.68 overlaps
  LinReg7's 0.54–0.91 (and lies below LinReg1's 0.59–1.01): G passes.
  | A vs B | CRPS_0.5 | CRPS_3 | e_0.5 | cal. A − B | S | C | verdict |
  |---|---|---|---|---|---|---|---|
  | s5 vs LinReg1 (partner = criterion) | +24.3 [+13.9, +33.0] (worse) | +3.4 [−0.3, +7.2], n.r. | +19.2 [+9.9, +27.4] (worse) | −14 [−16, −4] | fail | fail | **does not improve** |
  | s5 vs LinReg1 + power law (reported, C3) | +24.7 [+14.1, +33.5] (worse) | +2.9 [−1.3, +6.8], n.r. | +19.2 [+9.9, +27.4] (worse) | −13 [−17, −6] | fail | fail | no |
  | LinReg1 + power law vs LinReg1 (for reference) | −0.3 [−1.0, +0.5] | +0.5, n.r. | −0.0, n.r. | −1 [−2, +6] | fail | fail | no |

  Against "Expected": the verdict is as expected (does not improve), and as expected the ensembles are narrower and the
  guard nearly silent. Not as expected: (1) **the skill at 0.5 TU is clearly worse**, not "about the same": the ensemble
  mean's error is 15–20 % above LinReg1's at leads ≤ 0.5 TU, although offline (one step, recorded history, 10–100 TU) the
  sample mean's error was LinReg1's (1.013–1.018 ×, the sampling noise of 32 draws) and the training hindcast (in the
  training window) has a smaller bias than LinReg1's; the short-lead ensembles are under-dispersed (14 calibrated cells
  against 28). (2) **The long runs are not over-dispersed as feared and are closer to the reference than LinReg1's**: D
  0.48 against 0.84 (single runs 0.42–0.68), mean bias −1.4 % against −7.3 %, CRPS_100 0.381 and e_100 0.642 at or below
  LinReg1's, spread–skill 1.35 against 1.53, no guard firing. Explanation not yet checked: the card's own-draws diagnostic
  (the network's response to a history carrying its own samples) was not run before the coupled runs.
- **2026-11-11, the own-draws diagnostic** (Rik: "Yes, run it"; `exp_square_HIT/tools/gen_owndraws.jl`, laptop, the
  exported closure in plain arrays, no Lux; 10–100 TU, 3 paths, means of 16 draws; units LinReg1's clean rms; the desktop
  was unreachable, so the Lux mode first written into `cnn_mean_fit.jl` was removed again):

  | per QoI Z[0,6] E[0,6] Z[7,15] E[7,15] Z[16,32] E[16,32] | error, recorded history | error, own draws in the history | drawn noise sd | response |
  |---|---|---|---|---|
  | LinReg1 + MVG | 1.000 (all) | 1.31 1.32 2.58 2.50 1.39 1.59 | 0.95 0.94 0.97 0.97 0.97 0.97 | 0.89 0.92 2.45 2.37 1.00 1.28 |
  | s5 generator | 1.03 (all; a 16-draw mean's noise) | 1.30 1.31 2.37 2.30 1.38 1.55 | 0.91 0.89 0.90 0.91 0.98 0.96 | 0.93 0.97 2.40 2.30 1.01 1.26 |

  LinReg1's row reproduces `tab: own-draws` (response ~2.4 in the middle band). **The generator responds to its own
  draws as LinReg1 does** (it keeps LinReg1's skip, as expected), and one step ahead with its own draws in the history
  its error is no worse (a little lower, its noise being smaller). So the coupled short-lead loss is **not a one-step
  own-draw effect**: it builds up over the coupled steps (the ensemble mean's error over LinReg1's is ~1.0 at lead 25,
  1.04–1.16 at lead 100, 1.13–1.28 at lead 200). Unexplained.
- Not run (Rik, 2026-11-11: "add it as suggestion to the results text"): a hindcast of LinReg1 with the learned noise
  skip L alone (network part off), to separate the learned noise from the network's state-dependent part. Passed to the
  paper session as a suggestion sentence for `sec:res-stoch`.
