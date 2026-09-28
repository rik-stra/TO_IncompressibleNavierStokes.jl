# Plan — the search for a good stochastic LSTM (M4), 2026-09-24

Branch `m4-short-lstm`. Results go to `results_LSTMS.md` §7e/§7f; this file is the plan and its
checklist. Phase A uses **only `q*`** as input; Phase B adds `q_{n-1}`, `q*_{n-1}` and longer
history (Rik).

## What "good" means (fixed before searching)

**Online screen — decides.** 20 TU × 3 replicas on the desktop GPU, scored by
`TSCREEN=20 M4_SCREEN_SUBDIR=window analysis/m4_screen.jl` against the reference's own 20 TU windows:

| | reference band |
|---|---|
| stable (finite 20 TU) | 3/3 |
| clamp firings | 0 |
| flat fraction | 0.2–6.9% |
| time above 2900 (Z16) | 1.9–12.2% |
| Z16 min / max | 721–1080 / 3100–3790 |
| sd ratio | 0.86–1.14 |
| summed KS | 0.27–0.77 |
| `dQ` lag-1 (E[0,6]) | 0.73–0.75 |

A candidate **passes** when it is stable, never clamps, and sits inside the band on flat, >2900,
sd ratio and lag-1 in at least 2 of 3 replicas.

**Offline — only filters what goes online.** Held-out 50–100 TU: mean error (0.5·SSE per step,
standardised `dQ`) against the linear floor on the same input, and — to be added — an ensemble CRPS
for the latent models, which have no density. The offline ranking is **not** trusted on its own:
§7d and today's increment features both reversed online.

**Confirmation** of a winner: 5 seeds (S6, the median seed goes online), then 100 TU × 5 replicas
against LinReg1 and the DDN.

## Where it stands (facts, 2026-09-24)

- q*-only mean is **linear-limited**: floor 1.752 on `q*_n`, 1.458 with `q*_{n-1}` (the
  increment), and nothing beyond 2 lags. No LSTM beats the floor; a frozen least-squares skip
  guarantees it and an unbounded output path.
- 🔴 The increment enters with gain ~130 and **carries the model's own previous correction**
  (`q*_n = S(q*_{n-1} + dQ_{n-1})`): every model with the `[q*, Δq*]` skip diverged online, 0/9.
- Best online so far: **`:vrnn` + frozen skip on raw `q*_n`** — 3/3 stable, clamp 0, flat and
  >2900 in band; over-dispersed (sd ratio 1.25, min 521–654) and too persistent (lag-1 0.93).
- Tail validation selects better than blocked (1.736 vs 1.95–2.0 held out); KL in the source's form.

## Phase A — q* only

- [x] Window mode (reset + replay of the last W inputs, latent tied to the step), V61
- [x] Source-form KL (`kl_mode = :reference`), V62
- [x] Increment features (`input_map`), V63; explicit split, V64; linear skip on by default, frozen
- [x] Online round 1: raw/diff × skip/no-skip × latent/noise head (18 runs)
- [x] **A1 — where the lag-1 of 0.93 comes from.** Not the tying: W = 5/10/20 give 0.98/0.93/0.97. Online W = 5 / 20 against W = 10 (running). If
      persistence tracks W, it is the noise tying (consecutive predictions share W−1 draws).
      Diagnostic ablation: an untied closure (fresh draws per window) on the same fit.
- [x] **A2 — the dispersion (sd ratio 1.25, low minimum).** The latent scale sets it: `beta`
      (1e-4 → 1e-3 → 1e-2), `n_latent` (4 → 2 → 8), `:storn` vs `:vrnn` (running), and the
      emission-only baseline raw linear + η (running).
- [x] **A3 — a probabilistic offline score.** Done (CRPS + spread/skill); found the source latent 10-20x too narrow. Ensemble CRPS on 50–100 TU (K latent draws, tied
      exactly as online), so latent models can be ranked before GPU time.
- [ ] **A4 — the increment without the runaway.** Only if A1–A2 leave room: `diff` features with no
      skip were stable (3/3) but over-dispersed 1.8–2.7; try them with a larger `beta` / smaller
      latent.
- [ ] **A5 — seeds.** 3 seeds of the best Phase-A configuration online before calling it.

## Phase B — add the level history

- [x] **B0 — driver support.** `RIKFLOW_W_H` / `RIKFLOW_W_HIST_VAR` (`q_star_q`), so `x_t =
      [q*_t; q_{t-1}; q*_{t-1}; 1]` in window mode. Online the replayed `q_{t-1}` is the
      closure's own `q* + dQ` (the history buffer), so this reintroduces the closed-loop
      exposure §6 measured; the window tests (V61) extend to `h = 1`.
- [x] **B1 — h = 1, frozen least-squares skip** (floor 0.284) + latent (`:vrnn`/`:storn`) vs an
      LSTM noise scale — the q*-only winner's recipe on the richer input.
- [x] **B2 — the bias.** (λ 3e-6/1e-5: bias gone, spread 1.3–2.3 — §7g) §7d: the unregularised linear map runs +1 sd high online and ridge
      λ ≈ 1e-5 zeroes it. Seed the skip with the ridge map (needs a `LAMBDA` option) at
      λ ∈ {0, 3e-6, 1e-5}.
- [~] **B3 — h = 2 and deeper** (h = 2 done: linear + η nearest, 2/3 in band on 3 of 7 criteria), against §7d's lead `r3_lin_sd_h2` (persistent state, h = 2) as the
      baseline to beat.
- [ ] **B4 — a bounded encoder** (`n_encoder > 0`, the source's tanh layer) for the latent
      architectures: §7d's `:vrnn` on a skip diverged 6/6, read as its linear encoder
      extrapolating.

## Phase A′ — the conditional-VAE encoder (Rik, 2026-09-24)

The source's encoder `q(z | x)` never sees the target, so the latent carries no information about
it: offline spread/skill 0.05–0.11 (the ensemble 10–20× too narrow for one step) at any `beta`.
`posterior = :xy` trains `q(z_t | x_t, dQ_t)` and deploys `z_t ~ N(0, I)`, tied across windows.

- [x] `LSTMSpec(; posterior = :xy)`, training forward, deployed step, save/load; parity 7e-7
- [x] First fits (`SCORE=all`, constant head, `BETA=1`): spread/skill 0.71–0.89, CRPS 0.381 ≈ linear + η
- [x] Online, `:vrnn` and `:storn`, 20 TU × 3 — calibrated one-step noise over-disperses q*-only (1.7–1.8)
- [x] Sweep what sets the learned noise: `beta` (0.3/1/3), `n_latent` (2/4/8), W (5/10/20), head
      (`:constant` vs `:state_dependent`), `SCORE=last` vs `all`
- [x] Tests for `:xy` (V65) (deployed step ignores the encoder; `elbo` reads the target; save/load)

## Next, after §7g (2026-09-24)

- [x] **Surrogate go/no-go (§7h a): NO-GO** — a least-squares QoI Jacobian amplifies deviations
      (spectral radius 1.013 / 1.60) and gets the bias's sign wrong for 5 of 6 fits.
- [x] **Restart vs persistent (§7h b): restart wins** — consistent spread 0.91–1.17, lag-1 0.79–0.80;
      persistent is over-dispersed or erratic and brings back the flat state.
- [x] **Replayed-q* rollout (§7h c): wrong environment** — every λ = 0 fit explodes (loop gain ~119
      once q* stops responding); the old 1.00x evidence came from fits that ignored the lag.
- [x] **MLP surrogate S(q*_n, dQ_n) (§7h d): no-go** — no attractor: 18/18 diverge, even replaying the
      record's corrections; the linear head 14/18.
- [x] **Measured solver response (§7h e)** — an integrator over 10–25 steps; not a usable surrogate.
- [x] **Closed-loop calibration against 1–10 TU (§7i)** — per-QoI offset, one Newton step: bias gone in 3/3,
      KS in band in 3/3 (`calib/xy1_N1`). Next: noise scale for the spread, more replicas, 100 TU.
- [x] **Phase D, dense VAE (§7i)** — same as the LSTM offline and online.
- [ ] **The bias needs a closed-loop objective.** One-step training cannot see it (the LS map IS the
      one-step optimum); rollout / solver-in-the-loop fine-tuning of the mean (L5) is the lever, or an
      online choice of lambda (online selection — Rik's call).
- [ ] Seeds and 100 TU for the two nearest candidates (linear + eta h = 2, CVAE h = 1) before ranking them
- [ ] Phase D (dense/conv VAE over the window) — the reset window already makes the LSTM a
      feed-forward map of its window; B4 (bounded encoder) dropped: the CVAE never runs its encoder online

## Phase C — confirm and write up

- [ ] Best 1–2 overall → 5 seeds → 100 TU × 5 replicas vs LinReg1 and the DDN
- [ ] `results_LSTMS.md` §7f (q*-only search) and §7g (with history); `claude_memory.md` entry
- [ ] Stage; propose the commit (Rik commits)

## Phase D — later, only if the LSTM stalls (Rik)

- [ ] Drop the recurrence: a dense or convolutional VAE over the same window of inputs (and past
      latents), with the same conditional encoder, tying and online protocol.

## Housekeeping

- Fits: `exp_square_HIT/tools/m4_window_fit.jl` (`RIKFLOW_W_*`), ~1–2 min each on the CPU, 3–4 in
  parallel. Online: at most 5 sims on the shared 3090 (~4.4 GiB each, ~10 min per 20 TU).
- Every online candidate gets the closure-vs-forward parity check first (as done for round 1).
