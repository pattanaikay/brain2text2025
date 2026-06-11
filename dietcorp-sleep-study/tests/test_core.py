"""
tests/test_core.py — CPU unit tests for the standalone study (no checkpoint / GPU needed).
Run:  py -3 -m pytest dietcorp-sleep-study/tests -q
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from core.bit_encoder import BIT_Transformer
from core.model import (CTCPhonemeModel, session_ids_from_checkpoint, patch_embed_params,
                        infer_dims_from_state)
from core.episodic_memory import EpisodicBuffer, episodic_consistency_loss
from core.consolidator import TTAConsolidator, TTAConfig, ctc_greedy_decode
from core.lm_refine import PhonemeNGramLM, ctc_beam_search
from core.replay import ReplayStore, ReplayItem
from core.drift_eval import levenshtein, per, parse_session


def tiny_model(V=6):
    enc = BIT_Transformer(input_dim=8, embed_dim=12, num_layers=1, num_heads=2, patch_size=2)
    return CTCPhonemeModel(enc, num_phonemes=V)


# ── model / checkpoint helpers ────────────────────────────────────────────────

def test_session_ids_from_checkpoint():
    sd = {"encoder.read_in.t15_2023_08_11.weight": 0, "encoder.read_in.t15_2023_08_11.bias": 0,
          "encoder.read_in.default.weight": 0, "head.weight": 0}
    assert session_ids_from_checkpoint(sd) == ["t15_2023_08_11"]


def test_model_forward_shape_and_param_selection():
    m = tiny_model(V=6)
    out = m(torch.randn(2, 16, 8))
    assert out.shape[0] == 2 and out.shape[-1] == 6
    tp = patch_embed_params(m)
    assert len(tp) > 0


def test_infer_patch_size_from_checkpoint_shape():
    """Regression: the trained checkpoint uses patch_size=5 (patch_embedding in = 512*5).
    build_ctc_model must infer this rather than assume the encoder default of 4, or the
    real best_model_per.pth fails to load with a size mismatch."""
    state = {
        "encoder.read_in.default.weight": torch.zeros(512, 512),      # Linear(512, 512)
        "encoder.patch_embedding.weight": torch.zeros(384, 512 * 5),  # Linear(512*5, 384)
    }
    input_dim, patch_size = infer_dims_from_state(state)
    assert input_dim == 512 and patch_size == 5
    # explicit override is reported but the checkpoint shape still wins
    assert infer_dims_from_state(state, patch_size=4) == (512, 5)


# ── lm_refine ─────────────────────────────────────────────────────────────────

def test_ngram_lm_fit_score_and_roundtrip(tmp_path):
    lm = PhonemeNGramLM(order=3).fit([[1, 2, 3, 1, 2, 3], [1, 2, 3]])
    assert lm.logprob(2, (1,)) > lm.logprob(5, (1,))      # seen bigram > unseen
    p = tmp_path / "lm.json"; lm.save(str(p))
    lm2 = PhonemeNGramLM.load(str(p))
    assert lm2.logprob(2, (1,)) == lm.logprob(2, (1,))


def test_ctc_beam_search_recovers_clean_sequence():
    V = 5
    seq = [1, 2, 3]                                       # blank=0
    frames = [0, 1, 1, 0, 2, 0, 3, 3]
    lp = torch.full((len(frames), V), -10.0)
    for t, tok in enumerate(frames):
        lp[t, tok] = 0.0
    lp = torch.log_softmax(lp, dim=-1)
    assert ctc_beam_search(lp, lm=None, beam_size=4) == seq


# ── replay ────────────────────────────────────────────────────────────────────

def test_replay_priority_and_sample():
    store = ReplayStore(capacity=10)
    lat = torch.randn(12)
    store.add(ReplayItem(neural=torch.randn(8, 8), latent=lat, confidence=0.9, surprise=0.1))
    store.add(ReplayItem(neural=torch.randn(8, 8), latent=torch.randn(12), confidence=0.2, surprise=5.0))
    top = store.sample(1)
    assert len(top) == 1 and top[0].surprise == 5.0       # high surprise wins


# ── drift_eval metrics ────────────────────────────────────────────────────────

def test_metrics_and_session_parse():
    assert levenshtein([1, 2, 3], [1, 4, 3]) == 1
    assert per(torch.tensor([1, 4, 3]), torch.tensor([1, 2, 3])) == 1 / 3
    assert parse_session("t15.2023.08.11_data_val.hdf5__trial_0007") == "t15.2023.08.11"


# ── consolidator mechanism ────────────────────────────────────────────────────

def test_consolidate_lowers_loss_and_touches_only_target():
    torch.manual_seed(0)
    m = tiny_model(V=6)
    tp = patch_embed_params(m)
    # freeze a non-target param to verify it stays put
    head_w0 = m.head.weight.detach().clone()
    cons = TTAConsolidator(m, tp, config=TTAConfig(n_aug=4, lr=1e-2, mask_frac=0.5, mask_span=1))
    neural = torch.randn(20, 8)
    labels, conf = cons.pseudo_label(neural, mode="self")
    if labels.numel() == 0:
        labels = torch.tensor([1, 2])
    res = cons.consolidate([{"neural": neural, "labels": labels, "session": None, "confidence": conf}],
                           n_steps=8)
    assert not res["skipped"]
    assert res["loss_after"] <= res["loss_before"] + 1e-4
    assert res["params_changed"] >= 1
    assert torch.equal(head_w0, m.head.weight)            # head (non-target) unchanged
    assert res["wake_latency_ms"] >= 0.0


def test_oracle_mode_uses_given_labels():
    m = tiny_model(V=6)
    cons = TTAConsolidator(m, patch_embed_params(m), config=TTAConfig(n_aug=2))
    labels, conf = cons.pseudo_label(torch.randn(10, 8), mode="oracle",
                                     true_labels=torch.tensor([1, 2, 3]))
    assert labels.tolist() == [1, 2, 3] and conf >= 0.0


def test_memory_anchor_backprops():
    m = tiny_model(V=6)
    mem = EpisodicBuffer(embed_dim=12, buffer_size=8, n_heads=2)
    cons = TTAConsolidator(m, patch_embed_params(m), memory=mem,
                           config=TTAConfig(n_aug=4, lr=1e-2))
    neural = torch.randn(18, 8)
    labels = torch.tensor([1, 2, 3])
    res = cons.consolidate([{"neural": neural, "labels": labels, "session": None, "confidence": 0.9}],
                           n_steps=3)
    assert not res["skipped"] and res["params_changed"] >= 1
    # consistency loss term is well-formed and backprops
    q = torch.randn(2, 5, 12, requires_grad=True)
    loss = episodic_consistency_loss(q, torch.randn(2, 5, 12))
    loss.backward()
    assert q.grad is not None and float(loss) >= 0.0


def test_paramfree_retrieval_recalls_and_backprops():
    mem = EpisodicBuffer(embed_dim=12, buffer_size=16, n_heads=2,
                         retrieval="paramfree", min_read=2, fuse=True)
    for _ in range(4):                                   # fill a few slots
        mem.write_policy.write(mem, torch.randn(1, 5, 12), confidence=1.0, session_id=0)
    x = torch.randn(1, 5, 12, requires_grad=True)
    out = mem(x, write=False)
    assert out.shape == x.shape
    assert not torch.equal(out, x)                       # recall + fuse changed the output
    assert mem.last_read["memory_retrieved"].shape == x.shape
    out.sum().backward()
    assert x.grad is not None                            # gradient flows through param-free recall


def test_cold_buffer_guard_is_noop():
    mem = EpisodicBuffer(embed_dim=12, buffer_size=16, n_heads=2, min_read=8, fuse=True)
    x = torch.randn(1, 5, 12)
    out = mem(x, write=False)                            # empty buffer -> read bypassed
    assert torch.equal(out, x)
    # anchor target == query, so the consistency MSE is exactly 0 when cold
    assert torch.equal(mem.last_read["memory_query"], mem.last_read["memory_retrieved"])


def test_wake_read_off_keeps_output_clean():
    mem = EpisodicBuffer(embed_dim=12, buffer_size=16, n_heads=2,
                         retrieval="paramfree", min_read=2, fuse=False)   # C3a: anchor only
    for _ in range(4):
        mem.write_policy.write(mem, torch.randn(1, 5, 12), confidence=1.0, session_id=0)
    x = torch.randn(1, 5, 12)
    out = mem(x, write=False)
    assert torch.equal(out, x)                           # wake path unchanged (clean)
    assert not torch.equal(mem.last_read["memory_retrieved"], x)  # but recall still available for anchor


# ── WER word decoding ─────────────────────────────────────────────────────────

def test_wer_decode_exact_lexicon_roundtrip():
    """Oracle-style: a phoneme stream that exactly concatenates known prons must decode
    back to those words (this is the self-check that validates the inventory)."""
    from core.wer_decode import Lexicon, decode_words, compute_wer
    # toy lexicon in head-id space (ids are arbitrary but consistent)
    lex = Lexicon({"the": [10, 11], "cat": [3, 4, 5], "sat": [6, 4, 5]},
                  unigram={"the": -1.0, "cat": -2.0, "sat": -2.0})
    stream = [10, 11, 3, 4, 5, 6, 4, 5]            # the cat sat
    words = decode_words(stream, lex, lm_weight=0.1)
    assert words == ["the", "cat", "sat"]
    assert compute_wer(words, "the cat sat") == 0.0
    # one phoneme corrupted -> still recovers via nearest pronunciation
    noisy = [10, 11, 3, 9, 5, 6, 4, 5]
    assert compute_wer(decode_words(noisy, lex, lm_weight=0.1), "the cat sat") <= 1 / 3


def test_word_lm_and_canonical_decode_roundtrip(tmp_path):
    from core.wer_decode import Lexicon, compute_wer
    from core.word_lm_decode import WordNGramLM, decode_words_lm
    lex = Lexicon({"the": [10, 11], "cat": [3, 4, 5], "sat": [6, 4, 5]},
                  unigram={"the": -1.0, "cat": -2.0, "sat": -2.0})
    wlm = WordNGramLM(order=2).fit([["the", "cat", "sat"], ["the", "cat"]])
    assert wlm.logprob("cat", ("the",)) > wlm.logprob("sat", ("the",))   # seen bigram preferred
    p = tmp_path / "wlm.json"; wlm.save(str(p))
    wlm2 = WordNGramLM.load(str(p))
    assert abs(wlm2.logprob("cat", ("the",)) - wlm.logprob("cat", ("the",))) < 1e-9
    stream = [10, 11, 3, 4, 5, 6, 4, 5]                                    # the cat sat
    words = decode_words_lm(stream, lex, wlm, lm_weight=0.1, word_lm_weight=0.5)
    assert compute_wer(words, "the cat sat") == 0.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
