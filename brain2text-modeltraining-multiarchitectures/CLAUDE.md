# CLAUDE.md — Brain2Text Multi-Architecture Lab

> Read this file completely before touching any code. Contracts in §2 are non-negotiable.

## 1. What this repo is

A multi-architecture fork of `brain2text-modeltraining/`. The goal: swap the `BIT_Transformer` neural encoder for **seven alternative architectures** while keeping `MLPProjector`, the quantized Qwen2.5-1.5B LLM, and all training scripts intact. Each architecture lives in its own folder under `src/models/architectures/`.

### Critical tensor shapes

| Stage | Shape | Notes |
|---|---|---|
| Raw collated input | `(B, T_bins, 512)` | 20 ms bins, 512 channels, Z-scored + Gaussian-smoothed |
| After read-in | `(B, T_bins, 512)` | per-session or universal Linear(512,512) |
| **Encoder output** | `(B, T_patch, 384)` | `T_patch = ceil(T_bins / patch_size)` — **every arch must produce this** |
| Projector output | `(B, T_patch, llm_dim)` | `llm_dim=1536` for Qwen2.5-1.5B |
| CTC head output | `(B, T_patch, 42)` | blank=0, phonemes 1–41 |

---

## 2. Hard contracts — never break these

### 2.1 Encoder interface (every new encoder must satisfy this exactly)

```python
class YourEncoder(nn.Module):
    embed_dim: int = 384    # MLPProjector.input_dim is hardcoded to 384
    input_dim: int = 512
    patch_size: int         # used by bit_e2e.py and train_ctc.py for patched_lengths

    def forward(self, x, session_id=None, mask_patches=None, neural_lengths=None):
        # returns (B, T_patch, 384) — no exceptions
```

- Accept `session_id: list[str] | str | None` — mirror the branching in `encoder.py:143–172`.
- Accept `mask_patches: (B, T_patch) bool` for SSL. If unsupported, accept and ignore.
- Pad `T_bins` to multiple of `patch_size` with `F.pad(x, (0,0,0,pad_len))` before patching.
- Compute `key_padding_mask: (B, T_patch) bool` (True=pad) from `neural_lengths`. See `encoder.py:193–199`.
- End with `nn.Linear(hidden, 384) + nn.LayerNorm(384)` if backbone width ≠ 384.

### 2.2 Frozen downstream (do not edit these files)

- `src/models/projector.py` — `MLPProjector(input_dim=384, hidden_dim=1024, output_dim=llm_dim)`
- `src/models/encoder.py` — baseline reference, kept untouched
- `src/preprocessing/dataloader.py` and `bci_collate_fn`
- LLM: `Qwen/Qwen2.5-1.5B-Instruct`, 4-bit NF4, LoRA `r=8, α=32, dropout=0.2` on `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]` — unless §Arch-5 explicitly swaps the LLM.

### 2.3 Loss composition (baseline; arch-specific additions are additive only)

```
loss = ce_loss + 0.3 * ctc_loss + contrastive_loss
# Arch-3 adds:  + τ * topo_loss          (CLI: --topo_weight, default 0.0)
# Arch-6 adds:  + λ * aux_load_balance   (CLI: --aux_loss_weight, default 0.0)
```

---

## 3. Folder layout

```
src/models/
├── encoder.py          # baseline BIT_Transformer — READ ONLY
├── projector.py        # MLPProjector — READ ONLY
├── bit_e2e.py          # add encoder selection (§5.1 below) — minimal edit only
├── registry.py         # NEW — maps --encoder name → class
└── architectures/
    ├── conformer/      # Arch-1: ConformerEncoder (iPhoneme/ConformerXL)
    ├── mamba_possm/    # Arch-2: MambaPOSSMEncoder (POSSM hybrid SSM)
    ├── topoloss/       # Arch-3: TopoLoss module + hook collector (not an encoder)
    ├── zenbrain_memory/# Arch-4: ZenBrainEncoder (dual-memory attention + FIFO buffer)
    ├── audio_llm/      # Arch-5: LLM-swap variants (Qwen2-Audio, Whisper+Qwen, Phi-4-MM)
    ├── moe/            # Arch-6: MoEEncoder (SSMoE FFN blocks, Top-K router)
    └── hrm/            # Arch-7: HRMEncoder (L/H dual-timescale, DEQ 1-step gradient)

benchmarking/           # NEW — toy dataset + multi-arch eval (§6)
scripts/
├── train_e2e.py        # add --encoder, --topo_weight, --aux_loss_weight, --llm flags
├── train_ctc.py        # no changes needed
├── train_ssl.py        # no changes needed
├── train_topo.py       # NEW thin wrapper — enables topo_weight > 0
├── train_moe.py        # NEW thin wrapper — enables aux_loss_weight > 0
├── train_zerocal.py    # NEW — freeze weights, run buffer-update inference loop
└── train_hrm.py        # NEW — DEQ-style training (no standard BPTT)
```

Each `architectures/*/` folder must contain:
- `__init__.py` exporting the main class
- `encoder.py` (or `topo_loss.py` for Arch-3)
- supporting module files (blocks, router, buffer, etc.)
- `README.md` with: paper citation, component diagram, full implementation spec, hyperparameter table, and pitfalls. **The README is the authoritative spec — read it before implementing the arch.**

---

## 4. Shared engineering rules

1. **No edits to frozen files.** Only `bit_e2e.py` gets one targeted edit (encoder selection), `train_e2e.py` gets new CLI flags.
2. **Output projection always ends at 384.** `nn.Linear(hidden, 384) → nn.LayerNorm(384)`.
3. **patch_size propagation.** If your arch adds downsampling, set `self.patch_size = patch_size * factor` so CTC length math stays correct in `bit_e2e.py` and `train_ctc.py`.
4. **SSL mask token.** Implement `self.mask_token = nn.Parameter(torch.zeros(1, 1, 384)); nn.init.normal_(self.mask_token, std=0.02)` and substitute at masked positions. Required by `train_ssl.py`.
5. **Memory budget.** All archs must fit A100-40GB at `batch_size=8, seq_len≈400 bins` alongside 4-bit Qwen2.5-1.5B. Use `torch.utils.checkpoint.checkpoint` on encoder blocks if needed.
6. **No silent dtype changes.** Encoder runs under bfloat16 autocast. Only CTC log-softmax upcasts to fp32 (already in `bit_e2e.py`).
7. **Padding for Mamba/GRU.** These lack a key-padding mask. Multiply a `(B, T_patch, 1)` float mask into `x` after each block to zero padded positions.
8. **Comments.** Write only when the WHY is non-obvious (paper reference, subtle invariant). No docstrings restating the type signature.

---

## 5. Minimal edits to existing files

### 5.1 `src/models/registry.py` (create new)

```python
from src.models.encoder import BIT_Transformer
from src.models.architectures.conformer import ConformerEncoder
from src.models.architectures.mamba_possm import MambaPOSSMEncoder
from src.models.architectures.zenbrain_memory import ZenBrainEncoder
from src.models.architectures.moe import MoEEncoder
from src.models.architectures.hrm import HRMEncoder

ENCODER_REGISTRY = {
    "bit": BIT_Transformer, "conformer": ConformerEncoder,
    "mamba": MambaPOSSMEncoder, "zenbrain": ZenBrainEncoder,
    "moe": MoEEncoder, "hrm": HRMEncoder,
}
def build_encoder(name, **kwargs):
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Unknown encoder '{name}'. Available: {list(ENCODER_REGISTRY)}")
    return ENCODER_REGISTRY[name](**kwargs)
```

### 5.2 `bit_e2e.py` — one-line change (line ~67)

```python
# Replace:
self.neural_encoder = BIT_Transformer(session_ids=session_ids, patch_size=patch_size)
# With:
from src.models.registry import build_encoder
self.neural_encoder = build_encoder(encoder_name, session_ids=session_ids, patch_size=patch_size)
```

Add `encoder_name="bit"` to `BrainToTextE2E.__init__` with default `"bit"` → zero regression.

### 5.3 New CLI flags for `train_e2e.py`

```python
parser.add_argument("--encoder", choices=list(ENCODER_REGISTRY), default="bit")
parser.add_argument("--topo_weight", type=float, default=0.0)
parser.add_argument("--topo_sigma",  type=float, default=1.0)
parser.add_argument("--aux_loss_weight", type=float, default=0.0)
parser.add_argument("--llm", choices=["qwen2.5-1.5b","qwen2-audio-7b","whisper+qwen","phi4-mm"],
                    default="qwen2.5-1.5b")
```

---

## 6. Benchmarking folder (§ NEW)

Full spec: [`benchmarking/README.md`](benchmarking/README.md). Summary:

```
benchmarking/
├── README.md                    # full spec for this section
├── toy_dataset/
│   ├── generate_toy_data.py    # creates synthetic HDF5 files with known phoneme patterns
│   ├── toy_config.json         # dataset knobs (n_trials, n_sessions, seq_len_range, snr_db)
│   └── README.md
├── run_benchmark.py            # runs all registered archs on the toy dataset end-to-end
├── metrics/
│   ├── collect_metrics.py      # WER, CER, PER, forward latency (ms), peak VRAM (MB)
│   ├── results.json            # auto-written after each run
│   └── aggregate.py            # merges results.json files across multiple runs/seeds
└── plots/
    ├── plot_wer_comparison.py  # grouped bar chart: WER per arch
    ├── plot_training_curves.py # loss + WER/CER curves per arch (overlay mode)
    ├── plot_efficiency.py      # scatter: WER vs. VRAM and WER vs. latency
    ├── plot_topo_ablation.py   # WER vs. τ for Arch-3 TopoLoss
    ├── plot_expert_load.py     # MoE expert utilization heatmap (Arch-6)
    ├── plot_zerocal_curve.py   # ZenBrain WER vs. trial index (Arch-4 inference)
    └── figures/                # all PNG/SVG outputs land here
```

### Toy dataset spec (`benchmarking/toy_dataset/generate_toy_data.py`)

- **n_trials = 200** (split 160 train / 40 val).
- **n_sessions = 3** with distinct signal-to-noise ratios (`toy_config.json` default: `[30, 20, 15]` dB).
- **seq_len** sampled uniformly from `[80, 200]` bins (20 ms each → 1.6–4.0 s).
- **Signal model:** ground-truth phoneme sequence (8–15 phonemes, drawn from the 41-phoneme set) → per-phoneme Gaussian bump on a random subset of the 512 channels, summed, then additive white Gaussian noise at the session's SNR.
- **Outputs:** `data/toy/session_{0,1,2}/data_train.hdf5` and `data_val.hdf5`, matching the exact HDF5 schema expected by `Preprocessed_BCI_Dataset` (keys: `neural`, `text`, `phonemes`, attr: `session`).
- **Goal:** A correctly wired encoder should achieve PER < 40% on this dataset within 20 epochs. This is a smoke-signal, not a real benchmark.

### `benchmarking/run_benchmark.py` contract

```
python benchmarking/run_benchmark.py \
    --data_dir data/toy \
    --output_dir benchmarking/metrics \
    --encoders bit conformer mamba moe hrm \   # subset or "all"
    --epochs 20 \
    --seed 42
```

- Loops over selected encoders, calls `train_e2e.py` in subprocess with `--no_quantize --epochs N --batch_size 4`.
- After training, calls `collect_metrics.py` to log WER, CER, PER, forward-pass latency (median of 50 runs on a single batch), and peak VRAM.
- Appends a row to `benchmarking/metrics/results.json`.

### `benchmarking/metrics/collect_metrics.py` contract

```python
def collect(encoder_name, checkpoint_path, val_h5, device, n_latency_trials=50):
    # loads the model (no_quantize=True), runs validation → WER, CER
    # runs n_latency_trials forward passes on a fixed (4, 200, 512) tensor → median ms
    # reads torch.cuda.max_memory_allocated() → peak VRAM MB
    # returns dict; caller appends to results.json
```

### Plots spec (`benchmarking/plots/`)

Every plot script reads `benchmarking/metrics/results.json`. All figures saved to `benchmarking/plots/figures/` as both PNG (300 dpi) and SVG. No Jupyter notebooks — standalone scripts only.

| Script | Chart type | X-axis | Y-axis / Color |
|---|---|---|---|
| `plot_wer_comparison.py` | Grouped bar | Encoder name | WER (↓ better) |
| `plot_training_curves.py` | Line (overlay) | Epoch | Train loss + Val WER, one line per arch |
| `plot_efficiency.py` | Scatter | Forward latency (ms) | WER; point size = VRAM MB |
| `plot_topo_ablation.py` | Line | τ (0.01 → 0.5) | WER, Arch-3 only |
| `plot_expert_load.py` | Heatmap | Expert index | Token fraction routed, Arch-6 only |
| `plot_zerocal_curve.py` | Line | Trial index (0→40) | WER, Arch-4 buffer-filling curve |

Use `matplotlib` (no seaborn to avoid dependency bloat). Style: grid off, clean axes, figure size `(10, 5)` default. Save path hardcoded relative to this file's `__file__` — no CLI path args needed.

---

## 7. Compute budget & training protocol

- **150 epochs** hard ceiling for full runs; **20 epochs** for toy-dataset benchmarking.
- Default: `batch_size=16, accumulation_steps=4, lr=5e-5, weight_decay=1e-5, patience=50, val_interval=5`.
- Each arch gets its own `output_dir` (e.g. `outputs/e2e_conformer/`). Never share output dirs.
- Seed: `torch.manual_seed(42)` in every training script.
- Launch example per arch in the full spec — see each `architectures/*/README.md`.

---

## 8. Testing & verification

### Contract test (run before any full training)

```bash
pytest tests/test_encoder_contracts.py -v
```

```python
# tests/test_encoder_contracts.py
import torch, pytest
from src.models.registry import ENCODER_REGISTRY

@pytest.mark.parametrize("name", list(ENCODER_REGISTRY))
def test_shape_and_finite(name):
    enc = ENCODER_REGISTRY[name](session_ids=["s1","s2"], patch_size=4)
    out = enc(torch.randn(2,100,512), session_id=["s1","s2"],
              neural_lengths=torch.tensor([100,80]))
    assert out.shape == (2, 25, 384) and torch.isfinite(out).all()

@pytest.mark.parametrize("name", list(ENCODER_REGISTRY))
def test_mask_token(name):
    enc = ENCODER_REGISTRY[name](session_ids=["s1"], patch_size=4)
    mask = torch.zeros(1,25,dtype=torch.bool); mask[0,5:10]=True
    out = enc(torch.randn(1,100,512), session_id=["s1"], mask_patches=mask)
    assert out.shape == (1, 25, 384)
```

### Dry-run (per arch before 150-epoch commit)

```bash
python scripts/train_e2e.py --encoder <name> --train_h5 data/ --val_h5 data/ \
    --output_dir outputs/dry_<name> --epochs 1 --batch_size 2 --no_quantize
```

Smoke test inside `train_e2e.py` (lines 197–227) runs automatically. If it prints a WER value, the wiring is correct.

---

## 9. Reference papers

All PDFs in `../Reference Papers/` (one level up from this repo root):

| Arch | File |
|---|---|
| Baseline BIT | `Zhang et al. (2025).pdf` |
| 1 Conformer | `iPhoneme-Brain-to-Text Communication for ALS Using ConformerXL Decoding.pdf` |
| 2 Mamba/POSSM | `Generalizable, real-time neural decoding with hybrid.pdf` |
| 3 TopoLoss | `TopoNets-High performing vision and language models.pdf` |
| 4 ZenBrain | `ZenBrain_A_Neuroscience-Inspired_7-Layer_Memory_Ar.pdf` |
| 4 DietCORP | `Time-Masked Transformers with Lightweight.pdf` |
| 6 EEGMoE | `EEGMoE_A_Domain-Decoupled_Mixture-of-Experts_Model_for_Self-Supervised_EEG_Representation_Learning (1).pdf` |
| 6 NeuroMoE | `NeuroMoE.pdf` |
| 7 HRM | `HRM.pdf` |

**Read the paper before implementing each arch.** The `architectures/*/README.md` files contain detailed implementation specs — read those next.

---

## 10. Implementation order (recommended)

1. `registry.py` + `bit_e2e.py` one-line edit + `train_e2e.py` CLI flags  
2. `tests/test_encoder_contracts.py` — drives TDD  
3. Arch-1 (Conformer) — most similar to baseline; validates registry  
4. Arch-3 (TopoLoss) — pure loss module, no encoder changes; fast win  
5. Benchmarking folder + toy dataset generator  
6. Arch-6 (MoE) — replaces FFN inside existing block  
7. Arch-2 (Mamba) — needs GPU with mamba-ssm installed  
8. Arch-4 (ZenBrain) — two-stage training; attempt after pipeline is solid  
9. Arch-5 (Audio-LLM swap) — VRAM-constrained; confirm 7B fits first  
10. Arch-7 (HRM) — custom autograd; highest risk, save for last  
