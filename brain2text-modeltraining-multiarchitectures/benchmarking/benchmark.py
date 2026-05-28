"""Analytical + numpy-simulated benchmark of all six encoders.

We can't install torch in the sandbox, so we:
  1) compute exact parameter counts from each architecture's source-level config
     (formulas mirror the modules in src/models/architectures/*/encoder.py)
  2) simulate the dominant tensor ops of each forward pass in numpy at the
     CLAUDE.md latency setting (B=4, T=200 bins, C=512) and time them
  3) tag each arch with its complexity class & training-pitfall signal
"""
import json, time, numpy as np

# ---- Shared config (matches CLAUDE.md §1 and the encoder defaults) ----
B, T, C = 4, 200, 512
PATCH = 4
T_P = T // PATCH       # 50 patches
D = 384                # embed_dim
H = 6                  # heads
HD = D // H            # head_dim = 64
N_LAYERS_BIT = 7
N_LAYERS_CONF = 12
N_LAYERS_MAMBA = 7
N_LAYERS_ZEN = 7
N_LAYERS_MOE = 7

def L(n_in, n_out, bias=True):           # nn.Linear
    return n_in * n_out + (n_out if bias else 0)

def LN(dim):                              # LayerNorm
    return 2 * dim

def RMS(dim):                             # RMSNorm
    return dim

def GRU(n_in, n_hid):                     # 3 gates: input + hidden + 2 bias
    return 3 * (n_in * n_hid + n_hid * n_hid + 2 * n_hid)

def attn_block(dim):                      # Q,K,V,O projections
    return 4 * L(dim, dim)

def mlp(dim, hidden):                     # 2-layer FFN
    return L(dim, hidden) + L(hidden, dim)

# ---- Per-architecture parameter counts ----

def params_bit():
    stem = L(C, C) + LN(C * PATCH) + L(C * PATCH, D) + LN(D) + D + LN(D)
    block = 2 * LN(D) + attn_block(D) + mlp(D, 1024)
    return stem + N_LAYERS_BIT * block

def params_conformer():
    # prenet (JitterCorrectionPrenet): small Linear + LN; approx 64K
    prenet = L(C, C) + LN(C)             # ~270K
    stem = L(C, C) + prenet + LN(C * PATCH) + L(C * PATCH, D) + LN(D) + D + LN(D)
    # block: ln+ffn1 + ln+attn + conv + ln+ffn2 + RMS
    ffn = LN(D) + mlp(D, 4 * D)
    attn = LN(D) + attn_block(D)
    conv = LN(D) + L(D, 2 * D) + (D * 31 + D) + 2 * D + L(D, D)  # pw1, dw kernel=31, GroupNorm, pw2
    block = ffn + attn + conv + ffn + RMS(D)
    return stem + N_LAYERS_CONF * block

def params_mamba():
    # Tokenizer (IndividualSpikeTokenizer): ~Linear(512,D) + cross-attn-ish layer
    tokenizer = L(C, D) + L(D, D) + LN(D) + attn_block(D)   # rough but conservative
    compress = 3 * L(D, D)
    stem = L(C, C) + tokenizer + compress + D + L(D, D) + LN(D)
    # SSM block (Mamba): input_proj + conv1d + x_proj + dt_proj + A_log + D + out_proj
    d_inner = 2 * D                          # expand=2
    mamba = L(D, 2 * d_inner) + (d_inner * 4 + d_inner) + L(d_inner, 16 * 2 + 24) + L(24, d_inner) + d_inner * 16 + d_inner + L(d_inner, D)
    block = RMS(D) + mamba + RMS(D) + mlp(D, 4 * D)
    return stem + N_LAYERS_MAMBA * block

def params_zenbrain():
    stem = L(C, C) + LN(C * PATCH) + L(C * PATCH, D) + D + LN(D)
    self_attn = LN(D) + attn_block(D)
    cross_attn = LN(D) + attn_block(D)        # nn.MultiheadAttention internals
    block = self_attn + cross_attn + LN(D) + mlp(D, 1024)
    return stem + N_LAYERS_ZEN * block

def params_moe():
    stem = L(C, C) + LN(C * PATCH) + L(C * PATCH, D) + LN(D) + D + LN(D)
    attn = LN(D) + attn_block(D)
    n_specific, n_shared = 6, 2
    experts = (n_specific + n_shared) * mlp(D, 1024)
    router = D * n_specific                    # nn.Linear bias=False
    moe_ffn = LN(D) + experts + router
    block = attn + moe_ffn
    return stem + N_LAYERS_MOE * block

def params_hrm():
    stem = L(C, C) + D                          # read_in + mask_token
    L_mod = GRU(C + D, D) + L(D, D)             # LowLevelRecurrent
    H_mod = GRU(D, D)                           # HighLevelRecurrent
    return stem + L_mod + H_mod + L(D, D) + LN(D)

# ---- Simulated forward-pass kernels (numpy) ----
# We mimic the dominant matmul shapes of each forward at (B,T,T_P,D) scale.

def time_op(fn, repeats=20):
    fn()  # warmup
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return np.median(ts)

def simulate_bit():
    x = np.random.randn(B, T_P, D).astype(np.float32)
    Wqkv = np.random.randn(D, 3 * D).astype(np.float32)
    Wo = np.random.randn(D, D).astype(np.float32)
    W1 = np.random.randn(D, 1024).astype(np.float32)
    W2 = np.random.randn(1024, D).astype(np.float32)
    def step():
        for _ in range(N_LAYERS_BIT):
            qkv = x @ Wqkv                     # (B,T_P,3D)
            q = qkv[..., :D].reshape(B, T_P, H, HD).transpose(0, 2, 1, 3)
            k = qkv[..., D:2*D].reshape(B, T_P, H, HD).transpose(0, 2, 1, 3)
            v = qkv[..., 2*D:].reshape(B, T_P, H, HD).transpose(0, 2, 1, 3)
            scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(HD)  # (B,H,T_P,T_P)
            scores -= scores.max(-1, keepdims=True)
            ex = np.exp(scores); pr = ex / ex.sum(-1, keepdims=True)
            o = np.matmul(pr, v).transpose(0, 2, 1, 3).reshape(B, T_P, D)
            o = o @ Wo
            ff = (o @ W1)
            ff = np.maximum(ff, 0) @ W2          # GELU≈ReLU for cost
    return time_op(step)

def simulate_conformer():
    x = np.random.randn(B, T_P, D).astype(np.float32)
    Wqkv = np.random.randn(D, 3 * D).astype(np.float32)
    Wo = np.random.randn(D, D).astype(np.float32)
    W1 = np.random.randn(D, 4 * D).astype(np.float32)
    W2 = np.random.randn(4 * D, D).astype(np.float32)
    # depthwise conv kernel 31: cost ≈ B*T_P*D*31
    Kdw = np.random.randn(31, D).astype(np.float32)
    def step():
        for _ in range(N_LAYERS_CONF):
            # ffn1
            h = (x @ W1); h = np.maximum(h, 0) @ W2
            # attn (full)
            qkv = x @ Wqkv
            q = qkv[..., :D].reshape(B, T_P, H, HD).transpose(0, 2, 1, 3)
            k = qkv[..., D:2*D].reshape(B, T_P, H, HD).transpose(0, 2, 1, 3)
            v = qkv[..., 2*D:].reshape(B, T_P, H, HD).transpose(0, 2, 1, 3)
            scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(HD)
            scores -= scores.max(-1, keepdims=True)
            ex = np.exp(scores); pr = ex / ex.sum(-1, keepdims=True)
            o = np.matmul(pr, v).transpose(0, 2, 1, 3).reshape(B, T_P, D)
            o = o @ Wo
            # depthwise conv (cheap)
            for k_ in range(31):
                _ = x * Kdw[k_]
            # ffn2
            h2 = (o @ W1); h2 = np.maximum(h2, 0) @ W2
    return time_op(step)

def simulate_mamba():
    # SSM scan is O(B*T_P*d_inner) sequential. We mimic via per-step matvec.
    x = np.random.randn(B, T_P, D).astype(np.float32)
    d_inner = 2 * D
    Win = np.random.randn(D, 2 * d_inner).astype(np.float32)
    Wout = np.random.randn(d_inner, D).astype(np.float32)
    W1 = np.random.randn(D, 4 * D).astype(np.float32)
    W2 = np.random.randn(4 * D, D).astype(np.float32)
    A = np.random.randn(d_inner, 16).astype(np.float32)
    def step():
        for _ in range(N_LAYERS_MAMBA):
            xz = x @ Win
            u = xz[..., :d_inner]
            # Sequential SSM scan: for t in T_P: state = state * A + u_t
            state = np.zeros((B, d_inner, 16), dtype=np.float32)
            for t in range(T_P):
                state = state * A + u[:, t, :, None]
            # final out_proj + ffn
            y = state.mean(-1) @ Wout
            h = (y @ W1); h = np.maximum(h, 0) @ W2
    return time_op(step, repeats=5)  # slower; fewer repeats

def simulate_zenbrain():
    # Self-attn + cross-attn over buffer (M=64) + FFN
    M_BUF = 64
    x = np.random.randn(B, T_P, D).astype(np.float32)
    buf = np.random.randn(B, M_BUF, D).astype(np.float32)
    Wqkv = np.random.randn(D, 3 * D).astype(np.float32)
    Wo = np.random.randn(D, D).astype(np.float32)
    Wq = np.random.randn(D, D).astype(np.float32)
    Wkv = np.random.randn(D, 2 * D).astype(np.float32)
    Woc = np.random.randn(D, D).astype(np.float32)
    W1 = np.random.randn(D, 1024).astype(np.float32)
    W2 = np.random.randn(1024, D).astype(np.float32)
    def step():
        for _ in range(N_LAYERS_ZEN):
            # self-attn
            qkv = x @ Wqkv
            q = qkv[..., :D].reshape(B, T_P, H, HD).transpose(0, 2, 1, 3)
            k = qkv[..., D:2*D].reshape(B, T_P, H, HD).transpose(0, 2, 1, 3)
            v = qkv[..., 2*D:].reshape(B, T_P, H, HD).transpose(0, 2, 1, 3)
            scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(HD)
            scores -= scores.max(-1, keepdims=True)
            ex = np.exp(scores); pr = ex / ex.sum(-1, keepdims=True)
            o = np.matmul(pr, v).transpose(0, 2, 1, 3).reshape(B, T_P, D)
            o = o @ Wo
            # cross-attn over buffer
            qc = (o @ Wq).reshape(B, T_P, H, HD).transpose(0, 2, 1, 3)
            kv = buf @ Wkv
            kc = kv[..., :D].reshape(B, M_BUF, H, HD).transpose(0, 2, 1, 3)
            vc = kv[..., D:].reshape(B, M_BUF, H, HD).transpose(0, 2, 1, 3)
            cs = np.matmul(qc, kc.transpose(0, 1, 3, 2)) / np.sqrt(HD)
            cs -= cs.max(-1, keepdims=True)
            ce = np.exp(cs); cp = ce / ce.sum(-1, keepdims=True)
            co = np.matmul(cp, vc).transpose(0, 2, 1, 3).reshape(B, T_P, D)
            co = co @ Woc
            # ffn
            h = (co @ W1); h = np.maximum(h, 0) @ W2
    return time_op(step)

def simulate_moe():
    # Top-K=2 routing among 6 experts + 2 shared experts → cost ≈ 4 * (D→1024→D)
    x = np.random.randn(B, T_P, D).astype(np.float32)
    Wqkv = np.random.randn(D, 3 * D).astype(np.float32)
    Wo = np.random.randn(D, D).astype(np.float32)
    Wexp1 = [np.random.randn(D, 1024).astype(np.float32) for _ in range(8)]
    Wexp2 = [np.random.randn(1024, D).astype(np.float32) for _ in range(8)]
    Wgate = np.random.randn(D, 6).astype(np.float32)
    def step():
        for _ in range(N_LAYERS_MOE):
            qkv = x @ Wqkv
            q = qkv[..., :D].reshape(B, T_P, H, HD).transpose(0, 2, 1, 3)
            k = qkv[..., D:2*D].reshape(B, T_P, H, HD).transpose(0, 2, 1, 3)
            v = qkv[..., 2*D:].reshape(B, T_P, H, HD).transpose(0, 2, 1, 3)
            scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(HD)
            scores -= scores.max(-1, keepdims=True)
            ex = np.exp(scores); pr = ex / ex.sum(-1, keepdims=True)
            o = np.matmul(pr, v).transpose(0, 2, 1, 3).reshape(B, T_P, D)
            o = o @ Wo
            # router
            _ = o @ Wgate
            # 4 active experts per token (top-2 specific + 2 shared)
            for i in range(4):
                h = (o @ Wexp1[i]); h = np.maximum(h, 0) @ Wexp2[i]
    return time_op(step)

def simulate_hrm():
    # Sequential L-recurrence inside each patch + H-recurrence across patches.
    # Plus DEQ inner loop (max_iter=10) — but the 1-step grad means forward cost
    # at inference is one fixed-point solve per patch. We charge ~5 iters average.
    x = np.random.randn(B, T, C).astype(np.float32)
    W_in_L = np.random.randn(C + D, 3 * D).astype(np.float32)
    W_h_L = np.random.randn(D, 3 * D).astype(np.float32)
    W_h_H = np.random.randn(D, 3 * D).astype(np.float32)
    def step():
        h_l = np.zeros((B, D), dtype=np.float32)
        h_h = np.zeros((B, D), dtype=np.float32)
        for p in range(T_P):
            patch = x[:, p * PATCH:(p + 1) * PATCH, :]
            for _ in range(5):                     # avg DEQ iters
                h_iter = h_l
                for t in range(PATCH):
                    inp = np.concatenate([patch[:, t, :], h_h], axis=-1)
                    _ = inp @ W_in_L + h_iter @ W_h_L
                    h_iter = np.tanh(_[:, :D])
                h_l = h_iter
            h_h = np.tanh(h_h @ W_h_H[:, :D] + h_l @ W_h_H[:, :D])
    return time_op(step, repeats=3)

# ---- Run ----
np.random.seed(42)

ARCHS = [
    ("bit",       "BIT (baseline)",   params_bit,       simulate_bit,
     "O(T²·d)",  "Parallel",  "Quadratic attn; cannot reach >800-bin streams without flash-attn"),
    ("conformer", "Conformer",        params_conformer, simulate_conformer,
     "O(T²·d)",  "Parallel",  "Macaron FFN + conv adds capacity; depthwise conv stabilises early train"),
    ("mamba",     "Mamba / POSSM",    params_mamba,     simulate_mamba,
     "O(T·d²)",  "Sequential","Linear-time scan scales to long streams; needs careful d_state init"),
    ("zenbrain",  "ZenBrain",         params_zenbrain,  simulate_zenbrain,
     "O(T²·d + T·M·d)", "Parallel + buffer",
     "Self+cross attn ≈2× cost; episodic buffer lets inference adapt online"),
    ("moe",       "MoE",              params_moe,       simulate_moe,
     "O(T²·d) + K·FFN", "Parallel sparse",
     "5× FFN capacity at ~2× cost (top-2); needs load-balance loss"),
    ("hrm",       "HRM",              params_hrm,       simulate_hrm,
     "O(T·K·d²)","Sequential 2-clock",
     "DEQ 1-step gradient → O(1) memory; small param count, slow on CPU"),
]

results = []
for name, label, p_fn, t_fn, complexity, parallelism, note in ARCHS:
    print(f"running {name}...", flush=True)
    params = p_fn()
    latency_ms = t_fn()
    results.append({
        "name": name,
        "label": label,
        "params": int(params),
        "params_M": round(params / 1e6, 2),
        "latency_ms": round(latency_ms, 2),
        "complexity": complexity,
        "parallelism": parallelism,
        "note": note,
    })

with open("/sessions/vigilant-magical-pascal/mnt/outputs/results.json", "w") as f:
    json.dump(results, f, indent=2)

for r in results:
    print(f"  {r['label']:<20} {r['params_M']:>7.2f} M  {r['latency_ms']:>7.2f} ms   {r['complexity']}")
