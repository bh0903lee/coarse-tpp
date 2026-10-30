"""Shared helpers for the EasyTPP models of Section 5.2 and Supplementary Tables S3-S5: the two vendored EasyTPP copies,
the data sets and their single-file EasyTPP pickles, the run layout and the content hashes that bind every result to
its inputs.

Two copies of EasyTPP (S2P2 repository, commit c393324, Apache-2.0) are vendored under third_party/:
  easy_tpp        the unmodified models RMTPP, NHP, SAHP and AttNHP (coarse_tpp.rescore_intensity);
  easy_tpp_train  the complete package with the patches the retrained models were trained with
                  (third_party/easy_tpp_train/PATCHES.md; coarse_tpp.easytpp_train, coarse_tpp.rescore_retrained,
                  coarse_tpp.intensity_exceedance).
Both define the top-level package `easy_tpp`; `activate` puts the requested copy first on the path and, if the other
copy was imported earlier in the same process, removes it from sys.modules.

Data. EasyTPP trains on one pickle per data set, {"train": [...], "dev": [...], "test": [...], "dim_process": K}, a
sequence a list of {"time_since_start", "time_since_last_event", "type_event"} dicts with 0-based marks
(`prepare_file`). It is built from the public data folders (data/README.md): every sequence is kept, also single-event
ones, and values are copied as stored, except that the process logs, which store ties as exact zeros, get a positive
tie value for a zero gap after the first event (EasyTPP requires positive gaps): 1e-6 h for Sepsis and 1e-7 h for
BPIC12, whose log has genuine 1-3 ms gaps below 1e-6 h. A stored gap at most tie_threshold(ds) is read as a tie
(1.000001e-6, or the tie value times 1 + 1e-6 on the logs). Prepared from the data folders of data/README.md, the six
files are byte-identical to those the paper's models were trained on (TRAINED_SHA256).

Runs are found in the layout written by coarse_tpp.easytpp_train or in the layout of the trained-models archive
(`locate_run`).
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from . import data as D

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party"
COPIES = {"models": THIRD_PARTY / "easy_tpp", "train": THIRD_PARTY / "easy_tpp_train"}

MODELS = ("RMTPP", "NHP", "SAHP", "AttNHP", "THP", "S2P2", "ODETPP", "FullyNN", "IntensityFree")
INTENSITY = ("RMTPP", "NHP", "SAHP", "AttNHP", "THP", "S2P2", "ODETPP")
SEEDS = (0, 1, 2, 3, 4)

# data set folder (data/README.md) -> data set identifier inside the EasyTPP configurations (as in the trained-models
# archive); the identifier names the EasyTPP pickle and the experiment block of the configuration
EASYTPP_ID = {"email_eu": "data_email_eu", "retweet_sub": "data_retweet_sampled", "collegemsg": "data_collegemsg",
              "volcano": "data_volcano", "bpi_sepsis": "bpi_sepsis", "bpi_bpic12": "bpi_bpic12"}
FOLDER = {v: k for k, v in EASYTPP_ID.items()}
NAMES = {"email_eu": "Email-EU", "retweet_sub": "Retweet (subsample)", "collegemsg": "CollegeMsg",
         "volcano": "Volcano", "bpi_sepsis": "Sepsis", "bpi_bpic12": "BPIC12"}
TIE_VALUE = {"bpi_sepsis": 1e-6, "bpi_bpic12": 1e-7}       # written for a zero gap after the first event
SPLITS = ("train", "dev", "test")
# The files the paper's models were trained on: key order and pickle protocol of each file, and its SHA-256 (a file
# prepared from the data folders of data/README.md is byte-identical; `prepare_file` reports a difference)
FILE_LAYOUT = {"email_eu": ("splits_first", 5), "retweet_sub": ("splits_first", 5), "collegemsg": ("splits_first", 5),
               "volcano": ("dim_first", 4), "bpi_sepsis": ("dim_first", 4), "bpi_bpic12": ("dim_first", 4)}
TRAINED_SHA256 = {
    "email_eu": "9c46815451f879a6e59f62f222b105a101596cc13ad50a5011afef99ffba3a15",
    "retweet_sub": "1bcfb15ce90eec98e8478c163c725b5b4cd5d237ba8d31d67bd9c3b043202869",
    "collegemsg": "0d3d45c5d1c99fcc2720e9a772f438e5e38053644d98b4d132f0eb53f7cf75da",
    "volcano": "b8f180b6c63599f78dd5466e52005fcc5469721b5d7e2d99520910fe411d45ed",
    "bpi_sepsis": "22447f63a7295e7b06278e556f920f1af44dcc7086f4925b639d7dee579fa11e",
    "bpi_bpic12": "9639a27722d5357cff136009e633416f4a86f3c62f546efc8931e380c0f09348"}


# ------------------------------------------------------------------ vendored EasyTPP
def activate(copy: str = "train"):
    """Make `import easy_tpp` load the vendored copy `copy` ("train" or "models") and return the module."""
    root = COPIES[copy].resolve()
    mod = sys.modules.get("easy_tpp")
    if mod is not None and Path(mod.__file__).resolve().is_relative_to(root):
        return mod
    if mod is not None:
        for name in [n for n in sys.modules if n == "easy_tpp" or n.startswith("easy_tpp.")]:
            del sys.modules[name]
    others = {str(p.resolve()) for p in COPIES.values()} | {str(p) for p in COPIES.values()}
    sys.path[:] = [p for p in sys.path if p not in others]
    sys.path.insert(0, str(root))
    import easy_tpp
    assert Path(easy_tpp.__file__).resolve().is_relative_to(root), easy_tpp.__file__
    return easy_tpp


# ------------------------------------------------------------------ hashes
def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def content_sha256(obj) -> str:
    """SHA-256 of a canonical JSON dump (configurations compared by content, independent of key order)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# ------------------------------------------------------------------ data
def tie_threshold(ds: str) -> float:
    """Largest stored gap of the EasyTPP pickle of `ds` that counts as a tie."""
    v = TIE_VALUE.get(ds)
    return D.TIE_STORED if v is None else v * (1 + 1e-6)


def easytpp_split(ds: str, split: str):
    """The sequences of a split as EasyTPP sees them (all lengths, marks 0-based, ties as tie values)."""
    seqs, n_types = D._read_split(ds, split, 1)
    marks = D._check_marks([np.asarray([e["type_event"] for e in s], np.int64) for s in seqs], n_types, ds, split)
    tv = TIE_VALUE.get(ds)
    if tv is None and not D.stores_floor(ds):
        raise ValueError(f"{ds}: no tie value defined for a data set that stores exact zeros")
    gap = lambda j, g: float(tv) if (tv is not None and j > 0 and g == 0.0) else float(g)
    out = [[{"time_since_start": float(e["time_since_start"]), "time_since_last_event": gap(j, e["time_since_last_event"]),
             "type_event": int(k)} for j, (e, k) in enumerate(zip(s, mk))] for s, mk in zip(seqs, marks)]
    return out, n_types


def check_file(ds: str, d: dict) -> list:
    """Checks of a prepared pickle against the data folder (empty list = pass): sequences equal to the folder's (gaps
    after the tie value), positive gaps after the first event, marks in range, and the number of target gaps and ties
    equal to coarse_tpp.data.load."""
    bad, thr = [], tie_threshold(ds)
    for sp in SPLITS:
        ref, _ = easytpp_split(ds, sp)
        if d[sp] != ref:
            bad.append(f"{sp}: sequences differ from the data folder")
        for i, s in enumerate(d[sp]):
            g = [e["time_since_last_event"] for e in s[1:]]
            if g and min(g) <= 0:
                bad.append(f"{sp} sequence {i}: non-positive gap after the first event"); break
        mk = [e["type_event"] for s in d[sp] for e in s]
        if mk and not (0 <= min(mk) and max(mk) < d["dim_process"]):
            bad.append(f"{sp}: marks outside [0, {d['dim_process']})")
        v = D.load(ds, sp)
        g = np.concatenate([x[1:] for x in v.coarse.iet])
        mine = np.asarray([e["time_since_last_event"] for s in d[sp] if len(s) >= 2 for e in s[1:]])
        if (len(mine), int(np.sum(mine <= thr))) != (len(g), int(np.sum(g == 0))):
            bad.append(f"{sp}: target / tie counts differ from coarse_tpp.data.load")
    return bad


def prepare_file(ds: str, out_dir) -> Path:
    """Write <out_dir>/<EasyTPP identifier>.pkl for the data folder `ds` (layout above) and check it."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    d, dims = {}, set()
    for sp in SPLITS:
        d[sp], k = easytpp_split(ds, sp)
        dims.add(k)
    if len(dims) != 1:
        raise ValueError(f"{ds}: dim_process differs between splits: {dims}")
    order, protocol = FILE_LAYOUT.get(ds, ("dim_first", 4))
    k = int(dims.pop())
    d = {"dim_process": k, **d} if order == "dim_first" else {**d, "dim_process": k}
    bad = check_file(ds, d)
    if bad:
        raise ValueError(f"{ds}: prepared file fails its checks: {bad}")
    dst = out_dir / f"{EASYTPP_ID[ds]}.pkl"
    tmp = dst.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(d, f, protocol=protocol)
    tmp.replace(dst)
    if ds in TRAINED_SHA256 and sha256(dst) != TRAINED_SHA256[ds]:
        print(f"[coarse_tpp.easytpp] {dst.name}: SHA-256 differs from the file the paper's models were trained on "
              "(check the data folder against data/README.md)", file=sys.stderr)
    return dst


def read_file(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f, encoding="latin-1")


def load_split(pkl, split: str):
    """[(t, dt, k)] float64 / int64 arrays per sequence of length >= 2 of an EasyTPP pickle, and dim_process."""
    d = read_file(pkl)
    out = [(np.asarray([e["time_since_start"] for e in s], np.float64),
            np.asarray([e["time_since_last_event"] for e in s], np.float64),
            np.asarray([e["type_event"] for e in s], np.int64)) for s in d[split] if len(s) >= 2]
    return out, int(d["dim_process"])


# ------------------------------------------------------------------ runs
def run_dir(root, ds: str, model: str, seed: int) -> Path:
    """<root>/<data set>/<model>/s<seed>; the EasyTPP identifier is accepted as the data set directory too."""
    root = Path(root)
    for name in (ds, EASYTPP_ID.get(ds, ds)):
        p = root / name / model / f"s{seed}"
        if p.exists():
            return p
    return root / ds / model / f"s{seed}"


def train_dir(rdir: Path, ds: str, model: str):
    """The EasyTPP run directory <rdir>/<identifier>/<model>/train_<time> holding models/saved_model.pt (last one)."""
    c = sorted(p for p in (Path(rdir) / EASYTPP_ID.get(ds, ds) / model).glob("train_*")
               if (p / "models" / "saved_model.pt").exists())
    return c[-1] if c else None


def output_config(tdir: Path):
    c = sorted(Path(tdir).glob("*_output.yaml")) or sorted(Path(tdir).glob("config.yaml"))
    return c[0] if c else None


def _archive_row(root: Path, ds: str, model: str, seed: int):
    p = root / "scores.csv"
    if not p.exists():
        return None
    import csv
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("dataset") == ds and r.get("model") == model and str(r.get("seed")) == str(seed):
                return r
    return None


def locate_run(root, ds: str, model: str, seed: int) -> dict:
    """A trained run in either layout:
      training layout (coarse_tpp.easytpp_train): <root>/<data set>/<model>/s<seed>/result.json and EasyTPP's
        <identifier>/<model>/train_<time>/{models/saved_model.pt, *_output.yaml};
      archive layout (trained-models archive): <root>/<data set>/<model>/seed<seed>/{saved_model.pt, config.yaml, log}
        with an optional result.json, else the run's row of <root>/scores.csv (status, epochs, training record).
    Returns dict(dir, checkpoint, config, result_path, result) with None for what is missing; result None = no record."""
    root = Path(root)
    rdir = run_dir(root, ds, model, seed)
    if (rdir / "result.json").exists():
        tdir = train_dir(rdir, ds, model)
        return dict(dir=rdir, checkpoint=tdir / "models" / "saved_model.pt" if tdir else None,
                    config=output_config(tdir) if tdir else None, result_path=rdir / "result.json",
                    result=json.loads((rdir / "result.json").read_text()))
    adir = root / ds / model / f"seed{seed}"
    ck = adir / "saved_model.pt"
    if (adir / "result.json").exists():
        res, rp = json.loads((adir / "result.json").read_text()), adir / "result.json"
    else:
        row, rp = _archive_row(root, ds, model, seed), (root / "scores.csv")
        if row is None:
            return dict(dir=adir, checkpoint=None, config=None, result_path=None, result=None)
        num = lambda k: float(row[k]) if row.get(k) not in (None, "") else None
        res = dict(status="ok" if row.get("status") == "ok" else "error", error=row.get("failure") or None,
                   epochs=100, epochs_run=int(num("epochs_run")) if num("epochs_run") is not None else None,
                   microbatch=int(num("microbatch") or 1), mark_loss=False, source="archive scores.csv",
                   result=dict(best_valid_epoch=int(num("best_valid_epoch")) if num("best_valid_epoch") is not None else None,
                               best_metrics=dict(valid=dict(time_ll=num("training_record_valid_ll")),
                                                 test=dict(time_ll=num("training_record_test_ll")))))
    return dict(dir=adir, checkpoint=ck if ck.exists() else None, config=output_config(adir) if ck.exists() else None,
                result_path=rp, result=res)
