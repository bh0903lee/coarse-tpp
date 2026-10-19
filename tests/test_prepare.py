"""Derived data sets: new splits, coarsening, quantization, JODIE conversion."""
import json
import pickle

import numpy as np

from coarse_tpp import data as D
from coarse_tpp.prepare import coarsen, jodie, quantize, resplit


def toy(tmp, name="toy", n=(20, 5, 5), seed=0):
    rng = np.random.default_rng(seed)
    for sp, k in zip(("train", "dev", "test"), n):
        seqs = []
        for _ in range(k):
            t = np.cumsum(rng.exponential(1.0, 12)); t -= t[0]
            g = np.diff(t, prepend=0.0)
            seqs.append([dict(time_since_start=float(a), time_since_last_event=float(max(b, 1e-6)), type_event=0) for a, b in zip(t, g)])
        (tmp / name).mkdir(parents=True, exist_ok=True)
        pickle.dump({sp: seqs, "dim_process": 1}, open(tmp / name / f"{sp}.pkl", "wb"))


def test_resplit_preserves_sizes_and_the_multiset_and_is_deterministic(tmp_path):
    toy(tmp_path)
    a = resplit.build(tmp_path, "toy")
    b1 = (tmp_path / "toy_resplit" / "train.pkl").read_bytes()
    resplit.build(tmp_path, "toy")
    assert a == {"train": 20, "dev": 5, "test": 5} and (tmp_path / "toy_resplit" / "train.pkl").read_bytes() == b1


def test_quantize_and_coarsen_write_constructed_resolutions(tmp_path):
    toy(tmp_path, "conttime")
    quantize.quantize(tmp_path / "conttime", tmp_path / "conttime_q10", 0.1)
    d = pickle.load(open(tmp_path / "conttime_q10" / "train.pkl", "rb"))
    t = np.array([e["time_since_start"] for e in d["train"][0]])
    assert np.allclose(np.round(t / 0.1) * 0.1, t) and json.loads((tmp_path / "conttime_q10" / "resolution.json").read_text())["delta"] == 0.1
    row = coarsen.coarsen(tmp_path, "conttime", 1.0)
    old = D.data_dir(); D.set_data_dir(tmp_path)
    try:
        s = D.load_fine_s(row["dataset"], "test")
        g = D.load(row["dataset"], "test").coarse.iet
        assert all(np.all(x[1:] >= y[1:] - 1e-9) and np.all(x[1:] < y[1:] + row["delta"] + 1e-9) for x, y in zip(s, g))
    finally:
        D.set_data_dir(old)


def test_jodie_conversion_builds_user_sequences_with_the_floor_convention(tmp_path):
    csv = tmp_path / "log.csv"
    rows = ["user_id,item_id,timestamp,state_label,f1"]
    rng = np.random.default_rng(0)
    for u in range(30):
        t = 0.0
        for _ in range(rng.integers(1, 6)):
            t += float(rng.integers(0, 3))
            rows.append(f"{u},{rng.integers(0, 4)},{t},0,0.1")
    csv.write_text("\n".join(rows) + "\n")
    out = jodie.convert(csv, "log", tmp_path)
    tot = sum(v["n_seq"] for v in out["splits"].values())
    assert tot == out["n_seq"] and out["n_items"] <= 4
    d = pickle.load(open(tmp_path / "log" / "train.pkl", "rb"))
    gaps = [e["time_since_last_event"] for s in d["train"] for e in s[1:]]
    assert min(gaps) >= 1e-6 and all(len(s) >= 2 for s in d["train"])
