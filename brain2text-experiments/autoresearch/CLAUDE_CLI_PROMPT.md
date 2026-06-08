# Master Prompt — Brain2Text A100 Autoresearch (paste into Claude CLI)

> Copy everything inside the fenced block below into a Claude CLI session running **from the
> `brain2text-experiments/` repo root** on your local machine. Fill in the INSTANCE DETAILS and
> the SLACK WEBHOOK first. Claude CLI drives the A100 via the `jl` CLI: smoke tests → sweep
> Tracks A/B/C/D/E/F (all **short** toy runs, **no full runs**) → Tufte charts + interpretation
> reports → auto-heal on errors → Slack pings → pause the GPU whenever idle.

---

```
You are an autonomous ML research engineer. Run the Brain2Text autoresearch sweep on a
JarvisLabs A100 and produce ranked, interpreted, beautifully-charted results. ALL compute runs
on the A100 — nothing local. Every experiment is a SHORT toy run; you NEVER do a full/150-epoch
run. The point is to quickly learn which component (encoder / decoder / loss / projector) is the
best building block. Long-term full training is a separate goal I will handle later.

═══════════════════════════════════════════════════════════════════════════════
INSTANCE DETAILS  (I fill these in)
═══════════════════════════════════════════════════════════════════════════════
  machine_id:   <FILL IN, e.g. 417023>
  region:       <FILL IN, e.g. IN2>
  data on box:  <"yes — data/ has the hdf5s"  OR  "no — upload from local">
  SLACK webhook: <FILL IN, e.g. https://hooks.slack.com/services/XXX/YYY/ZZZ>

═══════════════════════════════════════════════════════════════════════════════
READ THESE FIRST  (they define the plan — do not re-derive it)
═══════════════════════════════════════════════════════════════════════════════
  autoresearch/program.md            ← steering: search order, metric, promote/discard rules
  autoresearch/EXPERIMENT_CATALOG.md ← the experiments (A/B/C/D/E/F), why each, the inspiration paper
  autoresearch/FEASIBILITY_AUDIT.md  ← bottlenecks (B3_mamba, bnb-4bit, HRM cost, MoE aux)
  autoresearch/A100_RUNBOOK.md       ← setup + time/cost reference
  autoresearch/notify_slack.py       ← Slack notifier helper (already in repo)
  registry.yaml                       ← experiment IDs, local_ok, depends_on, expected_wer_band
  README.md                           ← run.py usage

═══════════════════════════════════════════════════════════════════════════════
HARD RULES  (non-negotiable)
═══════════════════════════════════════════════════════════════════════════════
  1. RUN ALL TRACKS: A, B, C, D, E, F. (Track C — C1/C2/C3. Track F — JEPA, now ACTIVE with real
     wav2vec2-1D / DINOv2-2D / native backbones; it is a PRETRAINING track — see Phase 2F.)
     Skip only G, H (separate thesis line).
  2. EVERYTHING RUNS ON THE A100. Do not run experiments locally.
  3. SHORT RUNS ONLY. Always --profile toy. NEVER run --profile full or any 150-epoch run.
     "Winners" are just FLAGGED for my separate long-term-implementation goal — you do not train
     them to convergence. If a result is conclusive at toy scale, that is enough.
  4. RANK by slope (epoch 2→20) + WER@10 vs the track baseline (B → B0_baseline; D/E → BIT+
     default-loss+MLP). Label each: STRONG (≥5% slope gain) / PROMISING (3–5%) / WEAK (1–3%) /
     INERT (<1%). KEEP negative/inert results — they are findings, not failures.
  5. COMPUTE DISCIPLINE — never leave the GPU idle-Running (see COMPUTE DISCIPLINE section).
  6. AUTO-HEAL errors (see ERROR HANDLING section). One bad experiment must NOT halt the sweep.
  7. SLACK-NOTIFY at every run end + track end + heal action + pause + sweep end.
  8. Diagnose with `jl run logs` / `jl exec` before acting. Never fabricate a result or a fix.

═══════════════════════════════════════════════════════════════════════════════
PHASE 0 — Connect, sync, UPLOAD DATA, arm notifications
═══════════════════════════════════════════════════════════════════════════════
  - `jl status --json` — confirm auth + instance Running.
  - Upload the repo CODE (REQUIRED — Phase 1 needs run.py, specs/, stages/, tests/, profiles/,
    results/leaderboard.py, autoresearch/):
      jl upload <machine_id> . /home/brain2text-experiments
    VERIFY: `jl exec <machine_id> -- ls -la /home/brain2text-experiments/` shows run.py,
    registry.yaml, specs/, stages/, tests/, autoresearch/. If any are missing, re-run upload.
  - Upload the DATA explicitly (REQUIRED — every run reads these, ~2.4 GB total). Even if the code
    upload above already swept in data/, run this and VERIFY:
      jl upload <machine_id> data/toy_train.hdf5 /home/brain2text-experiments/data/
      jl upload <machine_id> data/toy_val.hdf5   /home/brain2text-experiments/data/
      jl exec   <machine_id> -- ls -lh /home/brain2text-experiments/data/
    EXPECT toy_train.hdf5 (~2.0 GB) + toy_val.hdf5 (~0.5 GB). If absent, STOP and re-upload — the
    sweep cannot run without them. (These are the only data files; there is no full data_train.hdf5
    in this repo, and the toy runs use toy_val.hdf5 for validation.)
  - On the box, export the webhook so every script can ping:
      export SLACK_WEBHOOK_URL="<the webhook from INSTANCE DETAILS>"
  - Send a start ping:
      python autoresearch/notify_slack.py "Autoresearch sweep starting on A100 <machine_id> — Tracks A/B/C/D/E/F, toy only" --status start

═══════════════════════════════════════════════════════════════════════════════
PHASE 1 — Setup + smoke tests  (MANDATORY before any training)
═══════════════════════════════════════════════════════════════════════════════
  a. pip install -r requirements.txt
     pip install mamba-ssm causal-conv1d        # Linux+CUDA, installs cleanly on A100
  b. PYTHONIOENCODING=utf-8 python autoresearch/preflight.py
     EXPECT on A100: bnb 4-bit ok, mamba-ssm available, shape gate PASS. NOTE: preflight buckets by
     registry `local_ok` (sized for the 6 GB local box), so it WILL mark A4 / C1 / C2 / B3_mamba as
     "deferred/cloud". On the A100 you DO run those (per Phase 2) — preflight here is an environment
     + shape check, NOT the run gate; Phase 2's explicit list is the gate.
  c. One-epoch dry run to catch OOM/NaN before the real sweep:
       python run.py --expt B0_baseline --profile toy --override epochs=1 \
         --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5
  - If preflight or the dry run fails: AUTO-HEAL (next section). If unrecoverable, Slack-notify,
    PAUSE the instance, and stop for me.

═══════════════════════════════════════════════════════════════════════════════
PHASE 2 — Track sweeps  (toy profile; `jl run` so each run is backgrounded + logged)
═══════════════════════════════════════════════════════════════════════════════
  Run order cheap→expensive. After EACH run: `python results/leaderboard.py --list`, then a
  per-run Slack ping (ID, WER@10, slope, label, pass/fail). Run 2–3 in parallel (40GB) to keep
  wall-clock — and thus cost — down.

  Track A (analysis; run.py routes A1/A2/A3 to their analysis tools — dispatch is fixed):
    # A3 phoneme probe — self-contained (uses toy_val):
    python run.py --expt A3 --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5
    # A1 CKA — needs a B0 checkpoint for neural projections; run AFTER B0_baseline (Track B), or
    # set encoder_ckpt in specs/A1_cka.yaml. Downloads several LLMs (fine on the A100):
    python run.py --expt A1 --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5
    # A4 audio-vs-vision E2E (downloads 7B; A100):
    python run.py --expt A4 --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5
    # A2 perplexity — PREREQUISITE: corpora under $BCI_DATA_ROOT (transcripts_val.txt, ptb_test.txt).
    # If absent, SKIP A2 and log "A2 deferred — corpora not provided" (not on the core path). Else:
    #   export BCI_DATA_ROOT=<dir with the corpora>   then:
    python run.py --expt A2 --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5

  Track B (B0_baseline FIRST — it is the control), then: B1 B2 B3 B3_mamba B4 B5
    - B2 (HRM) may be slow (DEQ). If epoch-1 > 3× B0, it's fine at toy scale — just let it finish;
      do NOT switch to full. Note the cost in the report.

  Track C (now included — all fit on the A100):
    C1 (Qwen2-Audio-7B), C2 (Phi-4-MM), C3 (Whisper-Qwen).  All --profile toy.

  Track D (independent, any order): D1b D1d D2a D2d D3b D3c D4

  Track E (E2b before E3; E3 is a ~6-cell grid): E1a E1b E2b E3

  Track F — JEPA PRETRAINING (ACTIVE; controlled A/B/C). DIFFERENT flow from B/C/D/E — it pretrains
  a backbone (NO LLM built) and produces NO single toy WER; rank it on pretraining health (no
  collapse) + the downstream fine-tune WER. stages/encoder/jepa.py has real per-modality backbones
  (wav2vec2-style 1D conv = audio, DINOv2-style 2D conv = video, native patch-embed = neural
  control). run.py routes the jepa encoder to its self-supervised pretraining path (masked-latent +
  EMA + VICReg) — verified working. Steps:
    1. Integrity + smoke gate FIRST (this IS the controlled-experiment guarantee):
         python -m pytest tests/test_jepa_smoke.py -v
         python tools/diff_specs.py specs/F1_audio_jepa.yaml specs/F2_video_jepa.yaml
         python tools/diff_specs.py specs/F1_audio_jepa.yaml specs/F3_neural_jepa.yaml
       The specs MUST differ ONLY in `encoder.modality` (the pytest already checks all three pairs).
       If any pair differs elsewhere, STOP — the controlled experiment is compromised.
    2. PRETRAIN each modality (F1 audio FIRST; F2/F3 depend on it). Runs masked-latent + EMA +
       VICReg, saves results/runs/<F*_dir>/pretrained_encoder.pth, logs collapse health (std > 0.5 OK):
         python run.py --expt F1 --profile toy --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5
         (then --expt F2, then --expt F3)
    3. DOWNSTREAM fine-tune each pretrained backbone for the real cross-modality comparison — feed
       the saved backbone through the standard E2E loop via --override (this is how F1/F2/F3 get
       COMPARABLE WERs; the pretrain step prints the exact command + path in its log):
         python run.py --expt F1 --profile toy \
             --override encoder.pretrained_ckpt=results/runs/<F1_dir>/pretrained_encoder.pth \
             --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5
    4. Report per modality: (a) NO collapse (VICReg variance healthy); (b) pretraining loss curve;
       (c) downstream WER. The winning modality is the JEPA lens whose backbone best decodes neural
       signal — the mechanistic complement to A4's pragmatic decoder-swap result.

═══════════════════════════════════════════════════════════════════════════════
PHASE 3 — After EACH track: charts + interpretation report
═══════════════════════════════════════════════════════════════════════════════
  Pull data from results/leaderboard.sqlite directly (table `runs`: expt_id, profile, best_wer,
  wer_at_ep10, slope, run_dir, notes) — a small sqlite3 query per track — or `python
  results/leaderboard.py --list` for a full dump. (leaderboard.py exposes --list / --frontier /
  --promote only; there is NO --track or --json flag.)

  CHARTS — use the /render-tufte-chart skill (tufte-vdqi plugin). If unavailable, produce
  Tufte-style seaborn: minimal ink, no chartjunk, direct labels, muted palette, 300dpi PNG+SVG
  to results/figures/. Per track:
    A: CKA score per backbone (A1); spoken/written PPL ratio (A2); phoneme probe acc (A3).
    B: WER@10 + best-WER bar per encoder with B0 reference line; slope-vs-WER scatter (size=params).
    C: WER bar per decoder (C1/C2/C3) vs the text-only Qwen baseline.
    D: WER bar per loss config vs default-loss baseline; annotate INERT ties.
    E: WER bar per projector; E3 → patch_size × n_queries heatmap.
    F: JEPA pretraining curves per modality (F1/F2/F3) — VICReg loss + variance + covariance terms
       over steps; bar of downstream decoding quality per backbone (audio vs video vs neural).

  REPORT — results/reports/<track>_interpretation.md, ≤1 page:
    - Ranked table: ID, name, WER@10, slope, Δslope% vs baseline, label (STRONG/PROMISING/WEAK/INERT).
    - 2–3 sentences PER experiment interpreting it against its hypothesis + inspiration paper from
      EXPERIMENT_CATALOG.md (e.g. "B1 slope +7% vs B0 → jitter-correction prenet helps, per the
      ConformerXL paper; STRONG." / "D2a (no contrastive) ties baseline → InfoNCE is INERT here").
    - Flag SUSPECT rows (best_wer outside registry expected_wer_band) for a re-run.
  Slack-notify track completion with the ranked winner.

═══════════════════════════════════════════════════════════════════════════════
PHASE 4 — Combination phase  (still toy, still short)
═══════════════════════════════════════════════════════════════════════════════
  - Pick winning encoder (B), decoder (C), loss (D), projector (E) by slope.
  - Run CMB-1 / CMB-2 from EXPERIMENT_CATALOG.md via --override on the winning encoder spec, e.g.:
      python run.py --expt B1 --profile toy \
        --override projector.variant=qformer projector.n_queries=32 \
                   loss.ctc_anneal.anneal_epochs=75 loss.contrastive.weight=0.0 \
        --train_h5 data/toy_train.hdf5 --val_h5 data/toy_val.hdf5
      (override keys MUST match the specs: projector.variant=qformer (not "kind"); loss entries are
       keyed by variant — loss.ctc_anneal.* and loss.contrastive.* — verified against the specs.)
  - Chart combination vs each single lever. State whether winners stack or interact negatively.
  - Still NO full run — the combination is evaluated at toy scale only.

═══════════════════════════════════════════════════════════════════════════════
PHASE 5 — Final report + handoff + PAUSE
═══════════════════════════════════════════════════════════════════════════════
  - results/reports/SWEEP_SUMMARY.md: master ranked leaderboard across A/B/C/D/E/F, the combination
    result, embedded key figures, and a "BEST BUILDING BLOCKS" section — the single best encoder,
    decoder, loss, projector to carry into the separate long-term implementation. NO full-run list.
  - Bundle: results/figures/ + results/reports/ + leaderboard.sqlite + cka_results.json.
  - Slack-notify the summary (winners + inert levers + suspects).
  - `jl pause <machine_id> --yes` and confirm Paused. Slack-notify the pause + (if visible) cost.

═══════════════════════════════════════════════════════════════════════════════
ERROR HANDLING & AUTO-HEAL  (rule 6 — keep the sweep alive)
═══════════════════════════════════════════════════════════════════════════════
  On any run failure: read the log (`jl run logs <id> --tail 80` or the run_dir log), classify,
  apply the matching fix, retry AT MOST twice, then mark the experiment FAILED/DEFERRED, Slack-
  notify with `--status heal` (or `fail`), and CONTINUE to the next experiment. Log every heal
  action in the track report.

  | Symptom in log | Diagnosis | Auto-heal action |
  |---|---|---|
  | `CUDA out of memory` | batch too big for the arch | retry with `--override batch_size=1 gradient_checkpointing=true`; if still OOM, halve `max_batches_per_epoch`, else DEFER |
  | `bitsandbytes` import/quant error | bnb wheel mismatch | `pip install bitsandbytes==0.44.1`; re-probe with preflight; if still broken, run that arch with `--override decoder.quantize=false` |
  | `No module named 'mamba_ssm'` (B3_mamba/Cx) | Mamba not installed | `pip install mamba-ssm causal-conv1d`; if build fails, SKIP B3_mamba (B3 GRU already covers it) |
  | `loss is nan` / SmokeAssert NaN | lr too high / bad init | retry `--override lr=2e-5`; if still NaN, mark SUSPECT, DEFER |
  | `Shape mismatch` in stack build | spec/stage contract bug | run `pytest tests/test_stage_shapes.py -k <arch>`; report the failing stage; SKIP that arch (do NOT hand-edit architecture code) |
  | `nvidia-smi` errors / 0 GPUs | instance unhealthy | `jl get <id>`; if not Running, Slack-notify + STOP for me (don't thrash resume) |
  | `No space left on device` | disk full | delete old `results/runs/*` dirs from prior attempts, retry once |
  | Slack post fails | network/webhook | ignore (notify_slack never raises) — never let this block a run |

  Never: invent a WER, silently skip without logging, retry the same failing command unchanged,
  or hand-edit model/architecture source to force a pass.

═══════════════════════════════════════════════════════════════════════════════
SLACK NOTIFICATIONS  (rule 7)
═══════════════════════════════════════════════════════════════════════════════
  Use `python autoresearch/notify_slack.py "<msg>" --status <ok|warn|fail|heal|start|pause>`.
  Ping on: sweep start; every experiment end (ID, WER@10, slope, label, pass/fail); every track
  completion (ranked winner); every auto-heal action; instance pause; sweep complete (summary).
  Keep each message one line. Example:
    python autoresearch/notify_slack.py "B1 ConformerXL done — WER@10 0.41, slope +7% vs B0 → STRONG" --status ok

═══════════════════════════════════════════════════════════════════════════════
COMPUTE DISCIPLINE  (rule 5 — do not waste GPU)
═══════════════════════════════════════════════════════════════════════════════
  - Keep the GPU BUSY or PAUSED — never idle-Running.
  - Run experiments back-to-back; batch 2–3 in parallel to shorten total GPU wall-clock.
  - Do NOT pause/resume between individual short runs (a resume costs 2–5 min — more than the idle
    it saves). Batch the work instead.
  - PAUSE immediately (`jl pause <machine_id> --yes`) when: the sweep finishes, OR you hit a STOP
    condition and must wait for me, OR an unavoidable idle gap > ~15 min appears.
  - Before pausing, make sure leaderboard.sqlite + figures + reports are written to disk (they
    persist across pause; ephemeral /tmp does not).
  - After pausing, confirm with `jl get <machine_id>` and Slack-notify.

═══════════════════════════════════════════════════════════════════════════════
DELIVERABLES
═══════════════════════════════════════════════════════════════════════════════
  results/leaderboard.sqlite                  ← every toy run, ranked
  results/figures/*.svg + *.png               ← Tufte charts per track + combination
  results/reports/<track>_interpretation.md   ← one per A/B/C/D/E/F (F = JEPA pretraining)
  results/reports/SWEEP_SUMMARY.md            ← master summary + BEST BUILDING BLOCKS
  autoresearch/runnable.json                  ← preflight output
  Slack thread                                ← live progress; final summary
  Chat: winners, inert levers, suspects, and the best building block per stage.

Work autonomously through the phases. After each track, post a 3-line chat status. Stop and ask
me only at the STOP conditions above. Never run a full/150-epoch run. Pause the GPU when idle.
```

---

## Notes for you (Pratik) — not part of the prompt

- **Slack webhook:** paste your webhook into the INSTANCE DETAILS block. The `notify_slack.py`
  helper reads `SLACK_WEBHOOK_URL` from the env (Phase 0 exports it). It never raises, so a Slack
  hiccup can't break a run. Keep the webhook out of any committed file.
- **No full runs:** this sweep is purely diagnostic — short toy runs to find the best building
  blocks. The `--profile full` path and `profiles/full.yaml` are untouched and left for your
  separate long-term-implementation effort.
- **Track C on A100:** C1 (7B) and C2 (5.6B) fit comfortably in 40 GB, so they're in scope now.
- **Pause behavior:** the agent batches runs and pauses at the end / when blocked — it won't
  thrash pause↔resume between every 5-minute run (that would waste more time than it saves).
