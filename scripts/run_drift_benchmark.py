#!/usr/bin/env python3
"""Canonical drift benchmark runner.

Runs TimeQA + arXiv drift scripts with fixed defaults and aggregates outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    p = subprocess.run(cmd, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonical concept-drift benchmark")
    parser.add_argument("--timeqa-limit", type=int, default=2000)
    parser.add_argument("--arxiv-max-papers", type=int, default=120)
    parser.add_argument("--timeqa-workers", type=int, default=1)
    parser.add_argument("--arxiv-workers", type=int, default=1)
    parser.add_argument("--arxiv-slices", nargs="+", default=["2017-2018", "2020-2021", "2024-2025"])
    parser.add_argument("--arxiv-concepts", nargs="+", default=["transformer", "attention", "prompt", "alignment"])
    parser.add_argument("--timeqa-output", default=None)
    parser.add_argument("--arxiv-output", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    timeqa_out = args.timeqa_output or f"perf/benchmark_timeqa_drift_{ts}.json"
    arxiv_out = args.arxiv_output or f"perf/benchmark_arxiv_drift_{ts}.json"
    final_out = args.output or f"perf/benchmark_drift_summary_{ts}.json"

    timeqa_cmd = [
        sys.executable,
        "scripts/analyze_timeqa_drift.py",
        "--split", "test",
        "--limit", str(args.timeqa_limit),
        "--text-source", "both",
        "--top-k-neighbors", "8",
        "--min-years-per-entity", "2",
        "--min-entities-per-year", "2",
        "--min-neighbor-support", "1",
        "--workers", str(args.timeqa_workers),
        "--output", timeqa_out,
    ]
    arxiv_cmd = [
        sys.executable,
        "scripts/analyze_arxiv_concept_drift.py",
        "--category", "cs.CL",
        "--slices", *args.arxiv_slices,
        "--concepts", *args.arxiv_concepts,
        "--max-papers-per-slice", str(args.arxiv_max_papers),
        "--top-m-docs", "25",
        "--keyword-k", "20",
        "--min-doc-support", "10",
        "--workers", str(args.arxiv_workers),
        "--output", arxiv_out,
    ]

    run_cmd(timeqa_cmd)
    run_cmd(arxiv_cmd)

    timeqa_payload = json.loads(Path(timeqa_out).read_text(encoding="utf-8"))
    arxiv_payload = json.loads(Path(arxiv_out).read_text(encoding="utf-8"))

    config = {
        "timeqa_limit": args.timeqa_limit,
        "timeqa_cmd": timeqa_cmd,
        "arxiv_cmd": arxiv_cmd,
    }
    cfg_hash = hashlib.sha1(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    summary = {
        "schema_version": "drift.v1",
        "benchmark_name": "canonical_drift_benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": cfg_hash,
        "runs": {
            "timeqa": {
                "output": timeqa_out,
                "summary": timeqa_payload.get("summary", {}),
            },
            "arxiv": {
                "output": arxiv_out,
                "summary": arxiv_payload.get("summary", {}),
            },
        },
    }

    Path(final_out).parent.mkdir(parents=True, exist_ok=True)
    Path(final_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Canonical drift benchmark complete")
    print(f"Summary: {final_out}")
    print(f"TimeQA output: {timeqa_out}")
    print(f"arXiv output: {arxiv_out}")


if __name__ == "__main__":
    main()
