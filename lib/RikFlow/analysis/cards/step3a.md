# Card: step 3a — the CNN mean (paper Sec. 7.4 `sec:lever-net`, offline Sec. 6 `sec:net-offline`), opened 2026-10-23

WORKFLOW.md Stage 0. Methods text: `closures.tex` §Step 3 (Eq. `eq: net mean`; the run variants below,
**draft 2026-10-23 for Rik**, paper branch `step3-cnn-closure`); offline evidence: Sec. 6 and
`results_LSTMS.md` §31–33; implementation: §33, tests V92–V94.

⚠️ **Opened after the training hindcast of a1 was submitted** (job 28503399, 2026-10-23), before any of
its numbers were read. The other five configs have not run.

🔒 **Once any full-hindcast number of these closures exists, nothing above "Post hoc" changes.**
Late observations go under "Post hoc", dated.

- [x] Rik has seen this card (date: 2026-10-23; "I've read the card")

## Closures

Network path (`D6_CLOSURE=lstm`, `StochLSTM`, Float32 model, Float64 solver), `arch = :cnn`: the CNN of
Sec. 6 (h = 5; two tanh convolutions, kernel 2, 16 channels; dense 64-64; 9 190 weights), **seed 1**,
trained again on 1–10 TU, weight decay 0.3 (each loss's own validation choice; Table `tab: net-loss`).
Deployed with the turbulence gate after the draw, the 100-step warm-up replayed verbatim (no draw), member
seeds `member_seed(k, m)` as LinReg1: every predicted step draws N_Q = 6 standard normals, mapped by the
head (`D LR ε`), as LinReg1's `MvNormal` draw maps its six. a1 against LinReg1 crosses the two
implementations, as the paper's step 3a always did (Table `tab: closures` caption; the implementation
check, App. `app:implementation`, measures that difference).

| run | closure | the one change | matched partner | artefact (`TO_LSTM/step3/closures/`, sha256) |
|---|---|---|---|---|
| a1 | CNN (RMSE) + MVG | the linear mean replaced by the CNN; Σ refitted about zero | LinReg1 (`D6_LinReg1`, `D6train_LinReg1`, its long runs; reused) | `CNN_rmse_mvg` c1a3b099… |
| a2 | CNN (CRPS) + MVG | the loss: CRPS instead of RMSE | a1 | `CNN_crps_mvg` ef1a774c… |
| a3 | CNN (CRPS) + learned head | the residual as learned with the mean: per-QoI sds (0.96 0.97 0.92 0.94 0.93 0.93 of the residual's rms), **no correlations** | a2 | `CNN_crps_head` cb2e1d87… |
| a3' | CNN (CRPS) + learned sds | the learned sds with the refitted correlations | a2 | `CNN_crps_headcorr` 4f76319a… |
| b1 | CNN (RMSE) + AR(2) | a coloured residual, step 1's AR(2) fit | a1 | `CNN_rmse_ar2n` 4f51c675… |
| c1 | CNN (RMSE) + power-law scale | a state-dependent residual of fixed form, step 2's | a1 | `CNN_rmse_pl` 96f0b007… |

Built 2026-10-23 on the laptop (`exp_square_HIT/tools/cnn_closure_export.jl <run> <noise> <name>`, from
`TO_LSTM/step3/grid_cap_{rmse,crps}_wd0.3_s1`, copied from the desktop). b1: φ1 0.019 0.010 0.071 0.156
0.097 0.072, φ2 −0.002 / 0 (the constraint binds in five QoIs: an AR(1) there). c1: β 0.52 0.55 1.62 1.48
1.23 1.32. Only a1 is on Snellius so far (sha256 equal).

**Expected, and why** (Sec. 6). Offline the CNN improves on LinReg1 nowhere: on 10–100 TU its one-step
error is 11–40 % larger in the middle and the small scales (a1: Z/E[7,15] 1.11 / 1.25, Z/E[16,32] 1.23
/ 1.15; within 2 % in the large scales), most of it at the 6.3 % of states outside the 1–10 TU range,
where its mean is biased (too large a correction in the small scales). Its residual is coloured there.
Expected, a1 vs LinReg1: CRPS_0.5 a few % worse; no gain in calibration; possibly more gate firings or
divergences at deep excursions, where the network extrapolates. The offline result need not decide the
coupled one: ridge (LinReg7) is worse offline and better in the long runs (Sec. 4). a2 vs a1: alike
(Table `tab: net-loss`: equal within the seeds on 10–100 TU). a3 vs a2: 3–8 % narrower ensembles in the
middle and small scales, and the six QoIs' noise uncorrelated (the Z–E correlations per band are 0.89–0.94
in the residual); a3' separates the two. b1 vs a1: no difference to speak of (φ ≤ 0.16; the AR adds
almost no colour, `results_LSTMS.md` §33 offline AR check). c1 vs a1: as step 2 on LinReg1 — narrower
ensembles at low states, fewer guard firings, no score gain. **Expected verdict vs LinReg1, every
config: does not improve.**

## Clauses (fixed: paper Sec. 3.7; round-1 card amendments 1–7 apply unchanged)

Hindcast K = 90 ICs, M = 10, 100 + 1200 steps; window CRPS_0.5 (steps 1–200) and CRPS_all (1–1200);
δ₀.₅ (1–200); r, the calibrated count and the rank histograms per lead (36 cells); lead-wise divergence
rule; long runs 5 × 100 TU.

- **G** KS guard against **LinReg7**: FAIL iff all 5 single-run summed KS lie above all 5 of LinReg7's.
- **S** CRPS_0.5 paired difference against LinReg1: 90 % CI upper end < 0.
- **C** calibrated cells (0.8 ≤ r ≤ 1.25, 36 cells) against LinReg1: paired CI of the count difference
  > 0, AND the CRPS_0.5 CI upper end < +10 % of LinReg1's CRPS_0.5.
- **Verdict:** IMPROVES iff G passes and (S or C).

Against the matched partner (the mechanism, secondary; same rules): paired CRPS_0.5 and calibrated count
for a2 vs a1, a3 and a3' vs a2, b1 and c1 vs a1.

Reported for every closure, no clause: CRPS_all, δ₀.₅, the calibrated count's reference, rank histograms,
mean bias by lead, long-run bias and spread–skill (per run), divergences, gate firings (hindcast: % of
steps, members, ICs; long runs: % of steps), and, from Sec. 6, the share of member steps whose level lies
outside its 1–10 TU range. Tool: `score_d6.jl --compare <D6_CNN_…> <D6_LinReg1>` (and vs the partner).

## Settings and where they come from

- No offset and no noise scale fitted against coupled runs: `lstm_knobs = input_map` only (the fixed
  input map of Sec. 7.4); the noise model is recorded in `lstm_noise`, `lstm_ar_phi`, `lstm_scale_beta`.
- The network: Sec. 6 (architecture study on the validation window); seed 1 by the rule fixed in advance;
  weight decay by each loss's own validation value, averaged over three seeds (0.3 for both losses).
- The loss: **RMSE first** (Rik, 2026-10-23: the likelihood and the CRPS "are mixing the network mean with
  the covariance head", so they are extensions, examples of learning head and mean together; the CRPS is
  run, the likelihood not in this campaign). ⚠️ Disclosure: the choice was made after Table `tab:
  net-loss` existed, where the RMSE and CRPS networks are the closest to LinReg1 on 10–100 TU, the
  evaluation window; the reason given is the loss's form, not that column.
- Σ (a1, a2): the second moment of the network's residual on 1–10 TU, about **zero** (Rik), as the closure
  draws noise centred at zero; the residual's means there are 0.00–0.04 of its rms.
- AR(2) (b1): step 1's fit, `fit_ar(Z, 2; nonneg = true)` (ACF lags 1–20, φ1 ≥ 0, φ2 ≤ 0), Σ_ξ = D R D with
  the AR's stationary variance matched to the residual's; warm start from the last two replayed steps.
- Power law (c1): step 2's fit, `fit_scale` (ML per QoI, the clip the 1–10 TU range of q*), Σ_ε the
  zero-mean second moment of the residual divided by f.
- ⚠️ Disclosures:
  - The CNN does **not pass the offline check** of Sec. 7.8 (validation likelihood −6.08 RMSE / −6.11
    CRPS against −6.95 for least squares); by that check a negative coupled result could come from the
    network's training (Sec. 6). It goes to production anyway (WORKFLOW Stage 2: a closure on the run
    list runs even if it looks bad offline).
  - The deployed lag window is Float32: the deployed closure reproduces the trained network to 8.9e-5
    sd(dQ) at most, rms 7.5e-6 (300 steps, all six configs).
  - The warm-up replays the RECORDED correction (`dQ` of the IC package), while the network's history in
    training was q^{n-k} − q^{(n-k)*} from the record's levels; the two differ by up to 0.12 sd(dQ) per
    step (0.04 rms in the small scales, `step3_data.jl`). This touches the first five predictions and
    b1's warm start only.
  - a3 draws uncorrelated noise across the six QoIs (the CRPS head learns no correlation).

## Prerun checks (WORKFLOW.md Stages 0–3)

- **GATE 0:** methods text — draft for Rik (paper branch `step3-cnn-closure`); this card — to be seen by Rik.
- **GATE 1 (implementation matches the text):**

  | text (closures.tex Step 3a) | checked by |
  |---|---|
  | Eq. `eq: net mean`: q̂ = q* + f(regressor) + η, target the correction | V93 (`target = :dQ`), the export's translation check |
  | f = the CNN of Sec. 6 on the fixed input map | V92 (deployed == Lux), V91 (input map), `step3_data.jl` input checks, translation check (8.9e-5 sd) |
  | η constant correlated Gaussian, Σ refitted about zero on 1–10 TU | export `mvg` (Σ = Z'Z/N; prints the residual's mean) |
  | the learned head (a3, a3') | export `head` / `headcorr` (prints learned sd / rms) |
  | AR(2) as step 1 | `fit_ar` shared with step 1, V94 (the recursion is LinReg's), V73 + bit check (LinReg unchanged) |
  | power-law scale as step 2 | `fit_scale` shared with step 2, V94, V83 |
  | warm-up verbatim, no draw; N_Q normals per step | V93, V72, V44 |
  | gate after the draw; the AR state advances on a gated step | V94, V44 |

  Suites: stdlib 3359 pass at `0a51d821` (1 pre-existing failure, V36 on `m4_control.jl`), and
  `test_d6_lstm.jl` 70 / 70 after the noise record was added (2026-10-23); Lux 933 / 933. LinReg
  byte-identical on 7 deployed files × 1600 steps (old vs new code, bit patterns).
- **GATE 2:** fits complete and reproducible (`cnn_mean_fit.jl`, `RIKFLOW_CNN_REFIT=1`, 3 seeds; seed spread
  in `results_LSTMS.md` §32); offline check failed (above, disclosed); residual colour on 1–10 TU: §33.
- **GATE 3 (smoke):** to do, on Snellius (Rik, 2026-10-23), in two halves:
  - hindcast path: the **training hindcasts** — the same `run_d6.jl` on the production device, which
    aborts unless every member is `warm_identical`; read from their files: finite (`diverged = false`),
    `model_name` / `lstm_noise` / `lstm_ar_phi` / `lstm_scale_beta` as this card, `gate_nfired = 0`;
  - long-run path: `batch_scripts/smoke_online_lstm.sh` with the six fit dirs, one gpu_a100 job: one
    replica each at 0.5 TU (100 warm-up + 100 forecast steps), then `tools/smoke_online_check.jl`
    (complete, warm-up `dQ` bit-identical to the record, finite, gate 0 after the warm-up, the fit
    dir's model; the log's `Deploying` / `Emission noise` lines as this card). Output in
    `output/smoke_online/`, deleted before production.
  
  A desktop CPU smoke (`run_d6.jl 1`, 2 members, then 1 TU long runs) was started and stopped on
  2026-10-23 before any member finished (~55 min for six; the hindcast half duplicates the training
  hindcast).
- **Training hindcast (amendment 7):** a1 job 28503399 (`run_d6_train_lstm.sh`, out `D6train_CNN_rmse_mvg`);
  the other five to submit; the defect rule against each matched partner (a1's: LinReg1).

## Production plan (after GATE 3 and Rik's go)

Per config: hindcast `D6_OUT=$PWD/output/D6_<name> D6_CLOSURE=lstm
D6_MODEL=$PWD/output/TO_LSTM/step3/closures/<name>/StochLSTM_seed1.jld2 sbatch batch_scripts/run_d6_packed.sh`
(~650 SBU); long runs `RIKFLOW_M4_MODEL_DIR=$PWD/output/TO_LSTM/step3/closures/<name> sbatch -t 00:30:00
batch_scripts/run_online.sh lstm 2 <r>`, r = 1…5 (~125 SBU; the short limit because the dispatch check
reserves the full limit, #81). Six configs ≈ 4 650 SBU: check the budget first.

Storage (checked 2026-10-23): ~1.6 GB per config (hindcast 169 MB as `D6_LinReg1`, training hindcast ~10
MB, long runs 5 × 289 MB as the LinReg7_pl / Splice1 runs of 20–22 Oct), ≈ 10 GB for six. Snellius home
85.4 % of 200 GiB (~29 GiB free; it was 76.3 % on 2026-10-20).

## Post hoc

**2026-10-23, a1 training hindcast** (job 28503399, gpu_a100, 17.5 min ≈ 37 SBU; code `9b4e6e48`; 50
files, copied to the laptop with the log; `train_hindcast.jl LinReg1 CNN_rmse_mvg`, PARTNER entries for
the six added):
- GATE 3, hindcast half: 50 / 50 `warm_identical`, 0 diverged, `model_name` step3/closures/CNN_rmse_mvg,
  the log's `M4: arch cnn, emission constant, window 1, target dQ, knobs [input_map], deploy seed 1`.
  `lstm_noise` is **not recorded**: the job ran before `ff4a7ab3`; a1's noise is white by construction
  (the export's `mvg`). ⚠️ The gate fired: 261 of 60 000 forecast steps (0.43 %), members 8 and 10 of IC
  k = 11 (t = 2.5 TU), first at leads 1042 and 987. LinReg1 fires at the same IC: 246 steps (0.41 %),
  members 2 and 8, first at leads 1041 and 1018. Read as the IC's (a deep excursion late in the
  forecast), not the CNN's.
- Defect rule (amendment 7): **no defect**. Spread–skill ratio over leads 1–200 / LinReg1's: 1.25 1.17
  1.20 1.22 1.17 1.14.
- In-sample, no clause: r ≤ 0.5 TU 0.77–0.91 (LinReg1 0.61–0.78), 0.5–3 TU 1.13–1.17 (LinReg1
  1.22–1.30); the ensemble mean's error ≤ 0.5 TU 0.86–0.95 of LinReg1's; bias at lead 200 −2.4 %
  (LinReg1 −6.1 %), at lead 1000 +4.5 % (+2.0 %); member steps below the training range 0.028–0.030
  (LinReg1 0.020–0.034).
