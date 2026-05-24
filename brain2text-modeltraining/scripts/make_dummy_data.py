"""
Generate tiny synthetic HDF5 files for local smoke-testing all three training scripts.
Usage:  py scripts/make_dummy_data.py
Outputs: outputs/dummy/data_train.hdf5  and  outputs/dummy/data_val.hdf5
"""
import os
import numpy as np
import h5py

OUT_DIR = "outputs/dummy"
os.makedirs(OUT_DIR, exist_ok=True)

RNG = np.random.default_rng(42)

SENTENCES = [
    "the cat sat on the mat",
    "hello world this is a test",
    "neural signals are fascinating",
    "brain computer interfaces work",
    "the quick brown fox jumps",
    "speech decoding from neural data",
]

PHONEME_VOCAB = 41  # 1-41, 0 is CTC blank

def write_split(path, n_trials, session="t5.2019.11.25"):
    with h5py.File(path, "w") as f:
        for i in range(n_trials):
            T = RNG.integers(80, 200)      # random time length (bins)
            grp = f.create_group(f"trial_{i:04d}")
            grp.create_dataset("neural", data=RNG.standard_normal((T, 512)).astype(np.float32))

            sentence = SENTENCES[i % len(SENTENCES)]
            grp.create_dataset("text", data=np.frombuffer(sentence.encode("utf-8"), dtype=np.uint8))

            n_phonemes = RNG.integers(4, 12)
            phonemes = RNG.integers(1, PHONEME_VOCAB + 1, size=n_phonemes).astype(np.int64)
            grp.create_dataset("phonemes", data=phonemes)
            grp.attrs["session"] = session
            grp.attrs["seq_len"] = n_phonemes

    print(f"Wrote {n_trials} trials → {path}")

write_split(os.path.join(OUT_DIR, "data_train.hdf5"), n_trials=32)
write_split(os.path.join(OUT_DIR, "data_val.hdf5"),   n_trials=8)
print("Done.")
