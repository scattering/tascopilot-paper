#!/usr/bin/env python
"""Rebuild the adaptive-run trace that plot_legacy_vs_tasai.py reads from the
public archive files alone, so the figure is reproducible without the run host.

The live executor names every file from its own step, Q and E, e.g.
  tasai_live_20260911_202614007_H0p000_K1p180_L0p000_E10p000779.dat.bt7
(run stamp 202614, step 007, K = 1.180, E = 10.0; the trailing number is the
NICE scan number). Counts and monitor come from the file's data row. The policy
phase is reconstructed from the step index and the run's thresholds
(space filling until gp_min_observations, log-GP until physics_min_observations).

    python scripts/tasai_trace_from_archive.py --data-dir ~/bt7_from_stampede/bt7/202608/14/data/data \
        --run 20260911_202614 --out local_runs/lsmo_tasai_live/trace_from_archive.jsonl
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

NAME = re.compile(r"tasai_live_(?P<run>\d{8}_\d{6})(?P<step>\d{3})_H(?P<h>m?\dp\d+)_K(?P<k>m?\dp\d+)_L(?P<l>m?\dp\d+)_E(?P<e>m?\d+p\d{3})(?P<scan>\d+)\.dat\.bt7$")

def num(tag: str) -> float:
    return float(tag.replace("m", "-").replace("p", "."))

def read_row(path: Path) -> dict:
    names = None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("## "):
            names = [t.split("(")[0] for t in line[3:].split()]; continue
        if line.startswith("#") or names is None or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= len(names):
            return {n: parts[i] for i, n in enumerate(names)}
    return {}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--run", required=True, help="run stamp, e.g. 20260911_202614")
    ap.add_argument("--gp-min", type=int, default=10); ap.add_argument("--physics-min", type=int, default=25)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for p in sorted(a.data_dir.glob(f"tasai_live_{a.run}*.dat.bt7")):
        m = NAME.match(p.name)
        if not m:
            continue
        row = read_row(p)
        if not row:
            continue
        step = int(m["step"])
        surrogate = "space_filling" if step < a.gp_min else ("tasai_log_gp" if step < a.physics_min else "tasai_fm_andie")
        rows.append({"step": step, "nodes": {"Q.H": num(m["h"]), "Q.K": num(m["k"]), "Q.L": num(m["l"]), "et.deltaE": num(m["e"])},
                     "observed_counts": float(row.get("Detector", "nan")), "monitor": float(row.get("monitor", "nan")),
                     "count_time_actual": float(row.get("time", "nan")), "scan_number": int(m["scan"]), "source_file": p.name,
                     "policy_metadata": {"surrogate": surrogate, "reconstructed_from": "archive filename and step thresholds"}})
    rows.sort(key=lambda r: r["step"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} points -> {a.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
