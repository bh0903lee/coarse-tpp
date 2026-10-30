"""Business process event logs (4TU.ResearchData) as temporal point process data sets (Section 6.2, S2).

    python -m coarse_tpp.prepare.bpi download [short ...]   # fetch the raw logs into RAW_DIR (default DATA_DIR/raw_bpi)
    python -m coarse_tpp.prepare.bpi convert  [short ...]   # build DATA_DIR/bpi_<short>/{train,dev,test}.pkl
    python -m coarse_tpp.prepare.bpi all      [short ...]

Logs (short names): sepsis, bpic12, bpic13i, helpdesk, bpic20dd, bpic20ptc, bpic20rfp; official deposits and DOIs
below. Licenses: BPIC 2020 logs CC BY-NC 4.0; the others the 4TU.ResearchData General Terms of Use.

Preprocessing (MoTPP, Yuan et al., Section 5.1, which follows Weytjens and De Weerdt 2022,
github.com/hansweytjens/predictive-process-monitoring-benchmarks, create_benchmarks.py):
  1. one case = one trace; mark = event `concept:name` (no lifecycle suffix); events sorted by timestamp, stable on
     ties (log order kept)
  2. exact duplicate events removed (same case, identical full event-attribute set)
  3. Weytjens' chronological-outlier date filters, only for logs where Weytjens defines them
  4. longest 5% of cases removed: duration <= q95 of case durations (days)
  5. end-of-window debiasing: keep cases with start <= (last timestamp after step 4) - q95 duration
     (skipped for BPIC13 incidents, as in MoTPP)
  6. strict temporal split: T = start time of the case at rank int(0.8 n) in start order; test = cases starting at
     or after T; train = cases ending strictly before T; cases that start before T and end at or after T are dropped
  7. dev = the most recent 20% (by start time) of the train cases
  8. vocabulary = activities of the final train split (frequency desc, then name); activities unseen in train map to
     one extra 'unknown' id = len(vocab); dim_process = len(vocab) + 1
  9. sequences shorter than 2 events dropped
Time unit: HOURS for every log. Absolute timestamps are parsed exactly into integer microseconds (UTC instants,
offsets applied); gaps are integer differences converted to float64 hours, so ties are exactly 0.0 (no 1e-6 floor is
stored). The first event of a case has time_since_start = time_since_last_event = 0.0.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from .. import data as D

OUT = D.data_dir()
RAW = OUT / "raw_bpi"
UNIT = "hour"
US_PER_UNIT = 3_600_000_000
TEST_SHARE = 0.2
DEV_SHARE = 0.2
TRUNC_Q = 0.95
UA = "Mozilla/5.0 (research data download)"
ND = "https://data.4tu.nl/ndownloader/files/"
# earlier official download URLs of the same 4TU deposits, tried if the current /file/ URL fails
LEGACY = {'sepsis': [24061976], 'bpic12': [24027287], 'bpic13i': [24033593], 'bpic20dd': [24031811, 24031814],
          'bpic20ptc': [24043835, 24043838], 'bpic20rfp': [24061151, 24061154]}

F4 = "https://data.4tu.nl/file/"
LOGS = {
    "sepsis": dict(name="Sepsis Cases - Event Log", doi="10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460",
                   page="https://data.4tu.nl/articles/_/12707639/1", fname="Sepsis Cases - Event Log.xes.gz",
                   url=F4 + "33632f3c-5c48-40cf-8d8f-2db57f5a6ce7/643dccf2-985a-459e-835c-a82bce1c0339",
                   motpp=dict(cases=1049, events=15214, acts=16)),
    "bpic12": dict(name="BPI Challenge 2012", doi="10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f",
                   page="https://data.4tu.nl/articles/_/12689204/1", fname="BPI_Challenge_2012.xes.gz",
                   url=F4 + "533f66a4-8911-4ac7-8612-1235d65d1f37/3276db7f-8bee-4f2b-88ee-92dbffb5a893",
                   weytjens=dict(start=None, end="2012-02", max_days=32.28),
                   motpp=dict(cases=13087, events=262200, acts=24)),
    "bpic13i": dict(name="BPI Challenge 2013, incidents", doi="10.4121/uuid:500573e6-accc-4b0c-9576-aa5468b10cee",
                    page="https://data.4tu.nl/articles/_/12693914/1", fname="BPI_Challenge_2013_incidents.xes.gz",
                    url=F4 + "0fc5c579-e544-4fab-9143-fab1f5192432/aa51ffbb-25fd-4b5a-b0b8-9aba659b7e8c",
                    no_end_debias=True, motpp=dict(cases=7554, events=65533, acts=4)),
    "helpdesk": dict(name="Helpdesk (Polato et al.)", doi="10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb",
                     page="https://data.4tu.nl/articles/_/12675977/1", fname="finale.csv",
                     url=F4 + "94ee26c8-78f6-4387-b32b-f028f2103a2c/697ff9af-fca0-4363-bd84-e81735d27e9f",
                     motpp=dict(cases=4580, events=21348, acts=14)),
    "bpic20dd": dict(name="BPI Challenge 2020: Domestic Declarations", doi="10.4121/uuid:3f422315-ed9d-4882-891f-e180b5b4feb5",
                     page="https://data.4tu.nl/articles/_/12692543/1", fname="DomesticDeclarations.xes.gz",
                     url=F4 + "6a0a26d2-82d0-4018-b1cd-89afb0e8627f/6eeb0328-f991-48c7-95f2-35033504036e",
                     motpp=dict(cases=10500, events=56437, acts=17)),
    "bpic20ptc": dict(name="BPI Challenge 2020: Prepaid Travel Cost", doi="10.4121/uuid:5d2fe5e1-f91f-4a3b-ad9b-9e4126870165",
                      page="https://data.4tu.nl/articles/_/12696722/1", fname="PrepaidTravelCost.xes.gz",
                      url=F4 + "fb84cf2d-166f-4de2-87be-62ee317077e5/612068f6-14d0-4a82-b118-1b51db52e73a",
                      weytjens=dict(start=None, end="2019-01", max_days=114.26),
                      motpp=dict(cases=2099, events=18246, acts=29)),
    "bpic20rfp": dict(name="BPI Challenge 2020: Request For Payment", doi="10.4121/uuid:895b26fb-6f25-46eb-9e48-0dca26fcd030",
                      page="https://data.4tu.nl/articles/_/12706886/1", fname="RequestForPayment.xes.gz",
                      url=F4 + "a6f651a7-5ce0-4bc6-8be1-a7747effa1cc/7b1f2e56-e4a8-43ee-9a09-6e64f45a1a98",
                      weytjens=dict(start=None, end="2018-12", max_days=28.86),
                      motpp=dict(cases=6886, events=36796, acts=19)),
}


# ----------------------------------------------------------------------------------------- download
def _get(url: str, timeout=60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _looks_valid(data: bytes, fname: str) -> str | None:
    """Return None if `data` is a complete raw file of the expected kind, else the reason."""
    head = data[:2048].lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        return "got an HTML page (repository maintenance page?)"
    try:
        if fname.endswith(".gz"):
            if data[:2] != b"\x1f\x8b":
                return "not gzip"
            data = gzip.decompress(data)
    except Exception as e:  # truncated gzip
        return f"gzip error: {e}"
    if fname.endswith((".xes", ".xes.gz")):
        tail = data[-4096:].rstrip()
        if not tail.endswith(b"</log>"):
            return "XES does not end with </log> (truncated)"
    elif fname.endswith(".csv"):
        if data.count(b"\n") < 100:
            return "CSV too short"
    return None


def download(short: str, log: dict) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    dst = RAW / log["fname"]
    urls = [log["url"]] + [ND + str(i) for i in LEGACY.get(short, [])]
    rec = dict(short=short, file=log["fname"], official_url=log["url"], alt_official_urls=urls[1:],
               doi=log["doi"], page=log["page"])
    if dst.exists() and _looks_valid(dst.read_bytes(), log["fname"]) is None:
        rec["status"] = "present"
        rec["sha256"] = hashlib.sha256(dst.read_bytes()).hexdigest()
        return rec
    attempts, data = [], None
    for u in urls:                                   # 1. live official URLs
        try:
            print(f"{log['fname']}: fetching {u}", flush=True)
            cand = _get(u)
            why = _looks_valid(cand, log["fname"])
            if why is None:
                data = cand
                rec.update(source="official 4TU.ResearchData", fetched_url=u)
                break
            attempts.append(f"{u}: {why}")
        except Exception as e:
            attempts.append(f"{u}: {type(e).__name__}: {e}")
    rec["attempts"] = attempts
    if data is None:
        rec["status"] = "failed"
        return rec
    dst.write_bytes(data)
    rec.update(status="downloaded", date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
               bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return rec



# ------------------------------------------------------------------------------------------- parsing
_TS = re.compile(r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2})(?:[.,](\d+))?)?\s*(Z|[+-]\d{2}:?\d{2})?$")
_EPOCH_ORD = date(1970, 1, 1).toordinal()


def parse_ts(s: str):
    """Exact parse -> (utc_us, local_us, n_frac_digits). local_us = wall clock as written."""
    m = _TS.match(s.strip())
    if not m:
        raise ValueError(f"unparsed timestamp {s!r}")
    y, mo, d, hh, mi = (int(m.group(i)) for i in range(1, 6))
    ss = int(m.group(6) or 0)
    frac = m.group(7) or ""
    us = int((frac + "000000")[:6]) if frac else 0
    days = date(y, mo, d).toordinal() - _EPOCH_ORD
    local = ((days * 24 + hh) * 60 + mi) * 60 * 1_000_000 + ss * 1_000_000 + us
    tz = m.group(8)
    off = 0
    if tz and tz != "Z":
        sign = -1 if tz[0] == "-" else 1
        t = tz[1:].replace(":", "")
        off = sign * (int(t[:2]) * 60 + int(t[2:])) * 60 * 1_000_000
    return local - off, local, len(frac)


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def read_xes(path: Path):
    """Yield (case_id, [(log_idx, activity, ts_string, attr_tuple), ...]) per trace."""
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as fh:
        ctx = ET.iterparse(fh, events=("start", "end"))
        depth_trace = False
        for ev, el in ctx:
            tag = _local(el.tag)
            if ev == "start":
                if tag == "trace":
                    depth_trace = True
                continue
            if tag == "trace":
                cid, events = None, []
                for ch in el:
                    t = _local(ch.tag)
                    if t == "event":
                        attrs = []
                        for a in ch:
                            k = a.get("key")
                            attrs.append((_local(a.tag), k, a.get("value"),
                                          ET.tostring(a) if len(a) else b""))
                        dct = {k: v for _, k, v, _ in attrs}
                        events.append((len(events), dct.get("concept:name"), dct.get("time:timestamp"),
                                       tuple(sorted(attrs, key=lambda x: (x[1] or "", x[2] or "")))))
                    elif ch.get("key") == "concept:name":
                        cid = ch.get("value")
                yield cid, events
                el.clear()
                depth_trace = False
            elif tag == "log":
                el.clear()


def read_helpdesk(path: Path):
    import csv
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    cols = list(rows[0].keys())
    ccol = next(c for c in cols if c.lower().replace(" ", "") in ("caseid", "case"))
    acol = next(c for c in cols if c.lower() in ("activity", "activity name"))
    tcol = next(c for c in cols if "timestamp" in c.lower())
    cases: dict = {}
    for i, r in enumerate(rows):
        ts = r[tcol].strip().replace("/", "-")
        ev = cases.setdefault(r[ccol], [])
        ev.append((len(ev), r[acol], ts, tuple(sorted(r.items()))))
    yield from cases.items()


# --------------------------------------------------------------------------------------- conversion
def _month_key(us):
    d = datetime.fromtimestamp(us / 1e6, tz=timezone.utc)
    return f"{d.year:04d}-{d.month:02d}"


GRID = [("1ms", 1_000), ("1s", 1_000_000), ("1min", 60_000_000), ("1h", 3_600_000_000),
        ("1day", 86_400_000_000)]


def resolution(local_us: np.ndarray, nfrac: np.ndarray) -> dict:
    out = {"n_timestamps": int(local_us.size),
           "frac_digits_counts": {str(k): int(v) for k, v in sorted(Counter(nfrac.tolist()).items())},
           "share_nonzero_subsecond": float(np.mean(local_us % 1_000_000 != 0)),
           "share_nonzero_ms": float(np.mean((local_us % 1_000_000) // 1_000 != 0)),
           "share_nonzero_us_below_ms": float(np.mean(local_us % 1_000 != 0)),
           "share_nonzero_seconds_field": float(np.mean((local_us // 1_000_000) % 60 != 0)),
           "share_nonzero_minutes_field": float(np.mean((local_us // 60_000_000) % 60 != 0)),
           "share_nonzero_time_of_day": float(np.mean(local_us % 86_400_000_000 != 0))}
    on = {name: float(np.mean(local_us % g == 0)) for name, g in GRID}
    out["share_on_grid"] = on
    delta = "1us"
    delta_us = 1
    for name, g in GRID:
        if on[name] >= 0.99:
            delta, delta_us = name, g
    out["delta_name"] = delta
    out["delta_in_unit"] = delta_us / US_PER_UNIT
    out["note"] = "measured on the wall-clock time as written in the log (offset not applied)"
    return out


def convert(short: str, log: dict) -> dict:
    src = RAW / log["fname"]
    reader = read_helpdesk if log["fname"].endswith(".csv") else read_xes
    cases = []           # (cid, [(utc, local, nfrac, log_idx, act, attrs)])
    n_unsorted = n_missing = 0
    for cid, evs in reader(src):
        rows = []
        for idx, act, ts, attrs in evs:
            if ts is None or act is None:
                n_missing += 1
                continue
            u, l, nf = parse_ts(ts)
            rows.append((u, l, nf, idx, act, attrs))
        if any(rows[i][0] > rows[i + 1][0] for i in range(len(rows) - 1)):
            n_unsorted += 1
        rows.sort(key=lambda r: (r[0], r[3]))  # stable on ties: log order
        cases.append((cid, rows))
    raw_cases, raw_events = len(cases), sum(len(r) for _, r in cases)
    raw_acts = len({r[4] for _, rr in cases for r in rr})
    all_local = np.array([r[1] for _, rr in cases for r in rr], dtype=np.int64)
    all_nf = np.array([r[2] for _, rr in cases for r in rr], dtype=np.int64)
    res = resolution(all_local, all_nf)
    # offsets: how many cases contain events with different UTC offsets (DST changes)
    n_offset_change = sum(1 for _, rr in cases if len({r[1] - r[0] for r in rr}) > 1)

    steps = {"raw_cases": raw_cases, "raw_events": raw_events, "raw_activities": raw_acts,
             "events_missing_name_or_time": n_missing, "cases_not_time_sorted_in_log": n_unsorted,
             "cases_with_utc_offset_change": n_offset_change}
    # 2. exact duplicates
    dedup, n_dup = [], 0
    for cid, rr in cases:
        seen, keep = set(), []
        for r in rr:
            if r[5] in seen:
                n_dup += 1
                continue
            seen.add(r[5])
            keep.append(r)
        if keep:
            dedup.append((cid, keep))
    cases = dedup
    steps["duplicate_events_removed"] = n_dup
    # informational only (not removed): events repeating (activity, timestamp) within a case
    steps["info_repeated_case_activity_timestamp"] = sum(
        len(rr) - len({(r[4], r[0]) for r in rr}) for _, rr in cases)
    # 3. Weytjens chronological outlier filters
    w = log.get("weytjens")
    if w and (w.get("start") or w.get("end")):
        n0 = len(cases)
        if w.get("start"):
            cases = [c for c in cases if _month_key(c[1][0][0]) >= w["start"]]
        if w.get("end"):
            cases = [c for c in cases if _month_key(c[1][-1][0]) <= w["end"]]
        steps["date_filter"] = {"start": w.get("start"), "end": w.get("end"), "cases_removed": n0 - len(cases)}
    # 4. longest 5% removed
    dur_days = np.array([(rr[-1][0] - rr[0][0]) / 86_400e6 for _, rr in cases])
    max_days = float(np.quantile(dur_days, TRUNC_Q))
    n0 = len(cases)
    cases = [c for c, d in zip(cases, dur_days) if d <= max_days * 1.00000000001]
    steps["max_days_q95"] = max_days
    steps["weytjens_max_days"] = (w or {}).get("max_days")
    steps["long_cases_removed"] = n0 - len(cases)
    # 5. end-of-window debiasing
    if not log.get("no_end_debias"):
        last = max(rr[-1][0] for _, rr in cases)
        latest_start = last - int(round(max_days * 86_400e6))
        n0 = len(cases)
        cases = [c for c in cases if c[1][0][0] <= latest_start]
        steps["late_start_cases_removed"] = n0 - len(cases)
    else:
        steps["late_start_cases_removed"] = "skipped (as in MoTPP)"
    steps["cases_after_filters"] = len(cases)
    steps["events_after_filters"] = sum(len(rr) for _, rr in cases)
    # 6. strict temporal split
    starts = np.array(sorted(rr[0][0] for _, rr in cases), dtype=np.int64)
    T = int(starts[int(len(starts) * (1 - TEST_SHARE))])
    test = [c for c in cases if c[1][0][0] >= T]
    train_all = [c for c in cases if c[1][-1][0] < T]
    steps["overlap_cases_dropped"] = len(cases) - len(test) - len(train_all)
    steps["separation_time_utc"] = datetime.fromtimestamp(T / 1e6, tz=timezone.utc).isoformat()
    # 7. dev = most recent 20% of train by start (stable: start, then case order)
    order = sorted(range(len(train_all)), key=lambda i: (train_all[i][1][0][0], i))
    n_dev = int(round(len(order) * DEV_SHARE))
    dev_idx = set(order[len(order) - n_dev:])
    train = [c for i, c in enumerate(train_all) if i not in dev_idx]
    dev = [c for i, c in enumerate(train_all) if i in dev_idx]
    # 8. vocabulary from train
    cnt = Counter(r[4] for _, rr in train for r in rr)
    vocab = [a for a, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))]
    vid = {a: i for i, a in enumerate(vocab)}
    unk = len(vocab)

    def to_seqs(cs):
        seqs, short_dropped = [], 0
        for _, rr in cs:
            if len(rr) < 2:
                short_dropped += 1
                continue
            t0 = rr[0][0]
            seq, prev = [], t0
            for r in rr:
                seq.append({"time_since_start": (r[0] - t0) / US_PER_UNIT,
                            "time_since_last_event": (r[0] - prev) / US_PER_UNIT,
                            "type_event": int(vid.get(r[4], unk))})
                prev = r[0]
            seqs.append(seq)
        return seqs, short_dropped

    outdir = OUT / f"bpi_{short}"
    outdir.mkdir(parents=True, exist_ok=True)
    split_stats, all_gaps = {}, []
    for name, cs in (("train", train), ("dev", dev), ("test", test)):
        seqs, nshort = to_seqs(cs)
        with open(outdir / f"{name}.pkl", "wb") as fh:
            pickle.dump({name: seqs, "dim_process": unk + 1}, fh, protocol=4)
        gaps = np.array([e["time_since_last_event"] for s in seqs for e in s[1:]], dtype=np.float64)
        all_gaps.append(gaps)
        n_unk = sum(1 for s in seqs for e in s if e["type_event"] == unk)
        split_stats[name] = {"cases": len(seqs), "events": int(sum(len(s) for s in seqs)),
                             "targets": int(gaps.size), "short_cases_dropped": nshort,
                             "tie_share": float(np.mean(gaps == 0)) if gaps.size else None,
                             "median_gap": float(np.median(gaps)) if gaps.size else None,
                             "max_len": int(max(len(s) for s in seqs)) if seqs else 0,
                             "unknown_activity_events": n_unk}
    g = np.concatenate(all_gaps)
    pos = g[g > 0]
    meta = {"short": short, "name": log["name"], "doi": log["doi"], "raw_file": log["fname"],
            "time_unit": UNIT, "n_types_train_vocab": len(vocab), "dim_process": unk + 1,
            "unknown_id": unk, "vocab": vocab, "steps": steps, "splits": split_stats,
            "tie_share_all": float(np.mean(g == 0)), "median_gap_all": float(np.median(g)),
            "median_positive_gap": float(np.median(pos)) if pos.size else None,
            "min_positive_gap": float(pos.min()) if pos.size else None,
            "max_seq_len": int(max(v["max_len"] for v in split_stats.values())),
            "resolution_raw_log": res, "motpp_table1": log.get("motpp")}
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    return meta


def cmd_download(shorts):
    """Fetch the raw logs; returns the short names whose download failed (recorded in RAW/downloads.json)."""
    man_p = RAW / "downloads.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    failed = []
    for s in shorts:
        rec = download(s, LOGS[s])
        if rec["status"] == "present" and s in man and man[s].get("sha256") == rec["sha256"]:
            print(f"{s}: present")
            continue
        man[s] = {**man.get(s, {}), **rec}
        print(f"{s}: {rec['status']} {rec.get('attempts', '')}")
        if rec["status"] == "failed":
            failed.append(s)
    man_p.write_text(json.dumps(man, indent=1))
    return failed


def cmd_convert(shorts):
    for s in shorts:
        if not (RAW / LOGS[s]["fname"]).exists():
            print(f"{s}: raw file missing, skipped")
            continue
        m = convert(s, LOGS[s])
        sp = m["splits"]
        print(f"{s}: train/dev/test cases {sp['train']['cases']}/{sp['dev']['cases']}/{sp['test']['cases']} "
              f"ties {m['tie_share_all']:.3f} resolution {m['resolution_raw_log']['delta_name']} "
              f"median gap {m['median_gap_all']:.4g} h")


def main(argv=None):
    global RAW, OUT
    import argparse
    ap = argparse.ArgumentParser(description="Download and convert the business process logs.")
    ap.add_argument("cmd", nargs="?", default="all", choices=["download", "convert", "all"])
    ap.add_argument("logs", nargs="*", help=f"short names (default: all of {', '.join(LOGS)})")
    ap.add_argument("--data", help="data directory (raw files in DATA/raw_bpi)")
    ap.add_argument("--raw", help="directory of the raw files")
    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    if a.data:
        OUT = Path(a.data); RAW = OUT / "raw_bpi"
    if a.raw:
        RAW = Path(a.raw)
    bad = [x for x in a.logs if x not in LOGS]
    if bad:
        ap.error(f"unknown log(s): {', '.join(bad)}")
    cmd, shorts = a.cmd, a.logs or list(LOGS)
    failed = []
    if cmd in ("download", "all"):
        failed = cmd_download(shorts)
    if cmd in ("convert", "all"):
        cmd_convert([s for s in shorts if s not in failed])
    if failed:
        print(f"download failed: {', '.join(failed)} (attempted URLs in {RAW / 'downloads.json'})", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
