# Ready-to-paste Claude Code prompts

---

## Prompt 1 — Upload to the A100, set up, run the tests, and report

> I have an A100 instance at `<USER>@<IP>` (SSH key `~/.ssh/id_rsa`). Set up my DietCorp-sleep
> study there and verify it. Steps:
>
> 1. **Sync code.** `rsync -avz` (fall back to `scp -r` if rsync is unavailable on Windows)
>    `C:\Projects\Brain2Text2025\brain2text2025\dietcorp-sleep-study\` →
>    `/workspace/dietcorp-sleep-study/`, excluding `.venv`, `__pycache__`, `results`, `.pytest_cache`.
> 2. **Upload artifacts** into `/workspace/dietcorp-sleep-study/data/`:
>    - `C:\Users\Pratik\Downloads\ctc\best_model_per (1).pth` → `data/best_model_per.pth`
>      (rename — drop the space and `(1)`).
>    - `C:\Projects\Brain2Text2025\brain2text2025\brain2text-modeltraining\data\session_stats.json`
>      → `data/session_stats.json`.
>    - The 17 GB `preprocessed_data.h5` → `data/preprocessed_data.h5` (this is large; show progress).
> 3. **Environment.** SSH in and run `bash a100/env_setup.sh`. Report the torch/CUDA line.
> 4. **Run the unit tests** and paste the full output:
>    `source .venv/bin/activate && python -m pytest tests -q`. I expect **10 passed**. If any
>    fail, show the traceback and stop.
> 5. **Verify data.** Run `python a100/prepare_data.py --data data/preprocessed_data.h5` and paste
>    the session inventory + whether `seq_class_ids` are present (they will NOT be — that's expected;
>    see DATA.md).
> 6. **Smoke the pipeline (C0/C1, no labels needed).** Run:
>    `PYTHONIOENCODING=utf-8 python run_study.py --checkpoint data/best_model_per.pth \
>    --data data/preprocessed_data.h5 --session_stats data/session_stats.json \
>    --conditions C0 C1 --n_steps 0 1 2 --max_sessions 4 --max_trials 16 --out results`
>    and paste the printed PER table.
> 7. **Report a summary**: tests pass/fail, #sessions found, the C0/C1 PER-vs-day numbers, and any
>    warnings. Do **not** run the full grid or C2/C3/C4 yet — I'll decide after seeing this.

---

## Prompt 2 — Generate labels for C2/C4 (run after Prompt 1 if C0/C1 looks right)

> On the A100 in `/workspace/dietcorp-sleep-study` (venv active): I want to enable conditions
> C2/C3/C4, which need `seq_class_ids`. Use the local-generation path in DATA.md (Option A):
>
> 1. `python a100/make_seq_class_ids.py --in data/preprocessed_data.h5 --out data/sessions_g2p.h5`
> 2. **Validate the phoneme inventory against the trained head** (decisive):
>    `python a100/make_seq_class_ids.py --in data/sessions_g2p.h5 \
>    --validate_with_model data/best_model_per.pth --session_stats data/session_stats.json`
>    Paste the `[validate] ... PER = ...` line. If PER is low (says "OK"), continue. If it says
>    "WARNING ... ordering likely WRONG", STOP and tell me — we'll switch to Option B (authors' data).
> 3. If validated, build the LMs: `python a100/build_lm.py --data data/sessions_g2p.h5 --out_dir data/lm`
> 4. Point `configs/study.yaml` `data:` at `data/sessions_g2p.h5` and run the full grid:
>    `bash a100/run_matrix.sh`. Paste the self-check WER line, the per-condition tables, and confirm
>    `results/wer_vs_day.png` was written. Then `jl pause` the instance.

---

## Prompt 3 — Produce a detailed, visual explainer of the study

> Read the entire `dietcorp-sleep-study/` folder (every file in `core/`, `run_study.py`,
> `a100/`, `configs/`, `tests/`, plus `README.md`, `DATA.md`) and also skim the parent
> `RESEARCH_NOTES.md`. Then produce a single self-contained **`EXPLAINER.html`** (inline CSS,
> Mermaid diagrams via the mermaid CDN) that explains this study in depth for a thesis-defense
> audience. It must cover, with a diagram for each where noted:
>
> 1. **The concept being tested.** State the hypothesis H_main in plain language and formally:
>    deeper N-step "sleep" consolidation reduces WER under day-to-day electrode drift at constant
>    wake latency. Explain *why electrode drift is a sequential problem* and why that makes the
>    sleep-paper prediction apply. [Diagram: the drift problem — WER 22.7%→66.5% over 8 days.]
> 2. **The three papers and exactly which parts we use.** A table mapping
>    DietCorp / "Do Language Models Need Sleep" / ZenBrain → the specific mechanism we adopted,
>    and (important) what we deliberately did NOT take (the sleep paper's *learned* local rule —
>    we use fixed gradient descent instead). [Diagram: the three papers collapsing onto the single
>    axis N = consolidation depth.]
> 3. **The architecture & wake/sleep loop.** [Mermaid flowchart]: wake path
>    (neural → patch_embed → transformer(frozen) → episodic-memory read → CTC head → phonemes →
>    words) vs sleep path (between trials: confidence-gated write → priority replay selection →
>    N AdamW steps on patch_embed only → memory-anchored). Emphasize that wake latency is
>    N-independent and consolidation cost is ~linear in N.
> 4. **The significant code changes**, file by file under `core/`: `consolidator.py` (the N-step
>    loop + memory anchor + LM/oracle pseudo-labels + replay batch), `episodic_memory.py`
>    (session-keyed + confidence-gated ring + cross-attn read + learnable gate),
>    `lm_refine.py` (phoneme n-gram + CTC beam — the fix for the N>1 collapse we observed),
>    `replay.py` (Simulation-Selection priority = |surprise|+(1−conf)+novelty),
>    `drift_eval.py` (real chronological-session loader), `wer_decode.py` + `phonemes.py`
>    (lexicon DP → word WER, with the oracle self-check). Quote the key function signatures.
> 5. **The experiment grid** C0–C4 × N∈{0,1,2,4,8} × days, what each condition isolates, and the
>    decision gates (H_main supported iff C2 N>1 < C1 N=1 at later days with flat wake latency;
>    memory helps iff C3 < C2; C4 = ceiling). [Diagram: the grid as a table/heatmap mockup.]
> 6. **What we already learned** (from `RESEARCH_NOTES.md`): the N>1 self-label collapse on toy
>    data and the two root causes we fixed (CTC head must load from the checkpoint; real neural
>    not random; lr=1e-5), and why that motivates LM-refined labels + the episodic anchor.
> 7. **Honest limitations**: synthetic-vs-real drift, approximate g2p oracle (DATA.md Option A),
>    PER vs canonical WER, and the stability ceiling at large N.
>
> Keep it accurate to the code (read it; don't invent). Use clear prose a committee can follow,
> and make every Mermaid diagram render. Output only `EXPLAINER.html` in the folder root.
