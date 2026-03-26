#!/usr/bin/env python3
"""bench-sweep: automated llama-bench profiling sweep driven by config.yaml."""

import argparse
import csv
import re
import subprocess
import threading
import time
from pathlib import Path

import yaml


# ── Resource Monitor ─────────────────────────────────────────
class ResourceMonitor:
    """Poll RSS, temperature, and RAPL energy in a background thread."""

    def __init__(self, pid, cfg):
        self.pid = pid
        self.poll_interval = cfg.get("poll_interval_s", 0.5)
        self.temp_path = cfg.get("temp_path")
        self.rapl_path = cfg.get("rapl_path")
        self._stop = threading.Event()
        self._thread = None
        # collected peaks
        self.rss_peak_kb = 0
        self.temp_max_c = 0.0
        self.energy_start_uj = None
        self.energy_end_uj = None

    def _read_int(self, path):
        try:
            return int(Path(path).read_text().strip())
        except (OSError, ValueError):
            return None

    def _poll(self):
        # snapshot RAPL start
        if self.rapl_path:
            self.energy_start_uj = self._read_int(self.rapl_path)
        while not self._stop.is_set():
            # RSS from /proc/pid/status VmHWM
            try:
                for line in Path(f"/proc/{self.pid}/status").read_text().splitlines():
                    if line.startswith("VmHWM:"):
                        kb = int(line.split()[1])
                        self.rss_peak_kb = max(self.rss_peak_kb, kb)
                        break
            except (OSError, ValueError):
                pass
            # Temperature
            if self.temp_path:
                val = self._read_int(self.temp_path)
                if val is not None:
                    self.temp_max_c = max(self.temp_max_c, val / 1000.0)
            self._stop.wait(self.poll_interval)
        # snapshot RAPL end
        if self.rapl_path:
            self.energy_end_uj = self._read_int(self.rapl_path)

    def start(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def energy_uj(self):
        if self.energy_start_uj is not None and self.energy_end_uj is not None:
            delta = self.energy_end_uj - self.energy_start_uj
            if delta < 0:  # counter wrapped
                delta += 2**32
            return delta
        return None


# ── Stderr parsers ───────────────────────────────────────────
# Pattern: "=== PAPI Coarse Metrics for Prefill ==="
RE_COARSE_HEADER = re.compile(r"=== PAPI Coarse Metrics for (\w+) ===")
# Pattern: "Cycles: 123 | Instructions: 456 | Cache Misses: 789 | Calls: 1740"
RE_COARSE_VALS = re.compile(
    r"Cycles:\s*(\d+)\s*\|\s*Instructions:\s*(\d+)\s*\|\s*Cache Misses:\s*(\d+)\s*\|\s*Calls:\s*(\d+)"
)
# Pattern: "IPC: 1.23 | MPKI: 4.56"
RE_COARSE_DERIVED = re.compile(r"IPC:\s*([\d.]+)\s*\|\s*MPKI:\s*([\d.]+)")
# Pattern: "  Prefill: State: 36.00 MB | KV: 36.00 MB (255 pos) | Time: 5513.15 ms | 46.43 tokens/s"
RE_COARSE_LINE = re.compile(
    r"^\s*(\w+):\s*State:\s*[\d.]+ MB\s*\|\s*KV:\s*([\d.]+) MB\s*\((\d+) pos\)\s*\|\s*Time:\s*([\d.]+) ms\s*\|\s*([\d.]+) tokens/s"
)
# Fine-grained table
# Pattern: "=== PAPI Fine Metrics for Prefill ==="
RE_FINE_HEADER = re.compile(r"=== PAPI Fine Metrics for (\w+) ===")
# Row: 40-char kernel name | Cycles | Instructions | Cache Misses | Calls | IPC | MPKI
# Kernel name can contain spaces and parentheses, e.g. "MUL_MAT cache_v_l28 (view) (permuted)"
RE_FINE_ROW = re.compile(
    r"^(.{40,40})\s*\|\s*(\d+)\s+\|\s*(\d+)\s+\|\s*(\d+)\s+\|\s*(\d+)\s+\|\s*([\d.]+)\s+\|\s*([\d.]+)"
)


def parse_stderr(text):
    """Parse all PAPI / coarse / fine data from stderr.

    Returns (coarse_records, fine_records) where each record is a dict.
    """
    coarse_records = []
    fine_records = []
    current_stage = None

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # Coarse PAPI header
        m = RE_COARSE_HEADER.search(line)
        if m:
            current_stage = m.group(1)
            rec = {"stage": current_stage}
            # next line: Cycles/Instructions/Cache Misses/Calls
            if i + 1 < len(lines):
                m2 = RE_COARSE_VALS.search(lines[i + 1])
                if m2:
                    rec["cyc"] = int(m2.group(1))
                    rec["ins"] = int(m2.group(2))
                    rec["miss"] = int(m2.group(3))
                    rec["calls"] = int(m2.group(4))
            # next next line: IPC/MPKI
            if i + 2 < len(lines):
                m3 = RE_COARSE_DERIVED.search(lines[i + 2])
                if m3:
                    rec["ipc"] = float(m3.group(1))
                    rec["mpki"] = float(m3.group(2))
            coarse_records.append(rec)
            i += 3
            continue

        # Coarse timing/KV line
        m = RE_COARSE_LINE.search(line)
        if m:
            stage = m.group(1)
            # find or create matching record
            for rec in coarse_records:
                if rec["stage"] == stage:
                    rec["kv_mb"] = float(m.group(2))
                    rec["kv_pos"] = int(m.group(3))
                    rec["latency_ms"] = float(m.group(4))
                    rec["ts"] = float(m.group(5))
                    break
            i += 1
            continue

        # Fine-grained header
        m = RE_FINE_HEADER.search(line)
        if m:
            fine_stage = m.group(1)
            i += 3  # skip header + separator
            while i < len(lines):
                m2 = RE_FINE_ROW.match(lines[i])
                if not m2:
                    break
                fine_records.append({
                    "stage": fine_stage,
                    "kernel": m2.group(1).strip(),
                    "cyc": int(m2.group(2)),
                    "ins": int(m2.group(3)),
                    "miss": int(m2.group(4)),
                    "calls": int(m2.group(5)),
                    "ipc": float(m2.group(6)),
                    "mpki": float(m2.group(7)),
                })
                i += 1
            continue

        i += 1

    return coarse_records, fine_records


# ── CSV writers ──────────────────────────────────────────────
COARSE_FIELDS = [
    "phase", "model", "n_prompt", "n_gen", "n_depth", "n_threads", "rep",
    "stage", "ts", "latency_ms", "kv_mb", "kv_pos",
    "calls", "cyc", "ins", "miss", "ipc", "mpki",
]
FINE_RAW_FIELDS = [
    "phase", "model", "n_prompt", "n_gen", "n_depth", "n_threads", "rep",
    "stage", "kernel", "calls", "cyc", "ins", "miss", "ipc", "mpki",
]
FINE_BY_OP_FIELDS = [
    "phase", "model", "n_prompt", "n_gen", "n_depth", "n_threads", "rep",
    "stage", "op", "calls", "cyc", "ins", "miss", "ipc", "mpki",
]
FINE_BY_WEIGHT_FIELDS = [
    "phase", "model", "n_prompt", "n_gen", "n_depth", "n_threads", "rep",
    "stage", "weight", "calls", "cyc", "ins", "miss", "ipc", "mpki",
]
RESOURCE_FIELDS = [
    "phase", "model", "n_prompt", "n_gen", "n_depth", "n_threads",
    "rss_peak_kb", "temp_max_c", "energy_uj",
]


def ensure_csv(path, fields):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def append_csv(path, fields, rows):
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


# ── Aggregation helpers ──────────────────────────────────────
# Extract op type: first word of kernel name (e.g. "MUL_MAT", "SOFT_MAX", "ROPE")
RE_OP_TYPE = re.compile(r"^(\S+)")

# Extract weight pattern from kernel name
RE_WEIGHT_PATTERN = re.compile(
    r"(attn_q|attn_k|attn_v|attn_output|ffn_down|ffn_up|ffn_gate|output\.weight|cache_k|cache_v)"
)


def _aggregate(records, key_fn):
    """Aggregate fine records by a key function.

    Returns list of dicts with summed cyc/ins/miss/calls and recomputed ipc/mpki.
    """
    buckets = {}
    for rec in records:
        key = (rec["stage"], key_fn(rec))
        if key not in buckets:
            buckets[key] = {"stage": rec["stage"], "cyc": 0, "ins": 0, "miss": 0, "calls": 0}
        b = buckets[key]
        b["cyc"] += rec["cyc"]
        b["ins"] += rec["ins"]
        b["miss"] += rec["miss"]
        b["calls"] += rec["calls"]

    result = []
    for (stage, group_key), b in buckets.items():
        b["ipc"] = round(b["ins"] / b["cyc"], 3) if b["cyc"] > 0 else 0.0
        b["mpki"] = round(b["miss"] / (b["ins"] / 1000.0), 3) if b["ins"] > 0 else 0.0
        b["_key"] = group_key
        result.append(b)
    return result


def aggregate_by_op(fine_records):
    """Aggregate fine-grained records by op type (MUL_MAT, ROPE, etc.)."""
    def key_fn(rec):
        m = RE_OP_TYPE.match(rec["kernel"])
        return m.group(1) if m else "UNKNOWN"
    agg = _aggregate(fine_records, key_fn)
    for rec in agg:
        rec["op"] = rec.pop("_key")
    return agg


def aggregate_by_weight(fine_records):
    """Aggregate fine-grained records by weight pattern (attn_q, ffn_down, etc.)."""
    def key_fn(rec):
        m = RE_WEIGHT_PATTERN.search(rec["kernel"])
        return m.group(1) if m else "other"
    agg = _aggregate(fine_records, key_fn)
    for rec in agg:
        rec["weight"] = rec.pop("_key")
    return agg


# ── Run one benchmark ────────────────────────────────────────
def run_bench(llama_bench, model_cfg, run_meta, rmon_cfg, results_dir, logs_dir, dry_run=False):
    """Run a single llama-bench invocation and collect all data."""
    args = [llama_bench]
    args += ["-m", model_cfg["path"]]
    args += ["-p", str(run_meta["n_prompt"])]
    args += ["-n", str(run_meta["n_gen"])]
    args += ["-t", str(run_meta["n_threads"])]
    args += ["-r", str(run_meta["reps"])]
    if run_meta.get("n_depth"):
        args += ["-d", str(run_meta["n_depth"])]
    args += run_meta.get("extra_args", [])

    label = (
        f"{run_meta['phase']}_{model_cfg['label']}"
        f"_p{run_meta['n_prompt']}_n{run_meta['n_gen']}"
        f"_d{run_meta.get('n_depth', 0)}_t{run_meta['n_threads']}"
    )

    if dry_run:
        print(f"  [DRY] {' '.join(args)}")
        return

    print(f"  >> {' '.join(args)}")
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Resource monitor
    mon = None
    if rmon_cfg.get("enabled"):
        mon = ResourceMonitor(proc.pid, rmon_cfg)
        mon.start()

    stdout_bytes, stderr_bytes = proc.communicate()
    if mon:
        mon.stop()

    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")

    # Save raw log
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / f"{label}.log").write_text(stderr_text + "\n---STDOUT---\n" + stdout_text)

    if proc.returncode != 0:
        print(f"  !! llama-bench exited {proc.returncode}, see {logs_dir / label}.log")
        return

    # Parse stderr
    coarse_records, fine_records = parse_stderr(stderr_text)

    common = {
        "phase": run_meta["phase"],
        "model": model_cfg["label"],
        "n_prompt": run_meta["n_prompt"],
        "n_gen": run_meta["n_gen"],
        "n_depth": run_meta.get("n_depth", ""),
        "n_threads": run_meta["n_threads"],
    }

    # Write coarse
    coarse_path = results_dir / "coarse.csv"
    ensure_csv(coarse_path, COARSE_FIELDS)
    for rep_idx, rec in enumerate(coarse_records):
        row = {**common, "rep": rep_idx // 2, **rec}  # 2 stages per rep
        append_csv(coarse_path, COARSE_FIELDS, [row])

    # Write fine raw (all per-kernel rows)
    fine_raw_path = results_dir / "fine_raw.csv"
    ensure_csv(fine_raw_path, FINE_RAW_FIELDS)
    for rec in fine_records:
        row = {**common, "rep": 0, **rec}
        append_csv(fine_raw_path, FINE_RAW_FIELDS, [row])

    # Write fine aggregated by op type
    fine_by_op_path = results_dir / "fine_by_op.csv"
    ensure_csv(fine_by_op_path, FINE_BY_OP_FIELDS)
    for rec in aggregate_by_op(fine_records):
        row = {**common, "rep": 0, **rec}
        append_csv(fine_by_op_path, FINE_BY_OP_FIELDS, [row])

    # Write fine aggregated by weight pattern
    fine_by_weight_path = results_dir / "fine_by_weight.csv"
    ensure_csv(fine_by_weight_path, FINE_BY_WEIGHT_FIELDS)
    for rec in aggregate_by_weight(fine_records):
        row = {**common, "rep": 0, **rec}
        append_csv(fine_by_weight_path, FINE_BY_WEIGHT_FIELDS, [row])

    # Write resource
    if mon:
        res_path = results_dir / "resource.csv"
        ensure_csv(res_path, RESOURCE_FIELDS)
        append_csv(res_path, RESOURCE_FIELDS, [{
            **common,
            "rss_peak_kb": mon.rss_peak_kb,
            "temp_max_c": f"{mon.temp_max_c:.1f}",
            "energy_uj": mon.energy_uj() or "",
        }])

    print(f"  OK ({len(coarse_records)} coarse, {len(fine_records)} fine-raw records)")

    # Cooldown between runs
    cooldown = run_meta.get("cooldown_s", 0)
    if cooldown > 0:
        print(f"  .. cooling down {cooldown}s")
        time.sleep(cooldown)


# ── Phase runners ────────────────────────────────────────────
def run_prompt_sweep(cfg, models, results_dir, logs_dir, dry_run):
    phase = cfg["phases"]["prompt_sweep"]
    if not phase.get("enabled"):
        return 0
    count = 0
    print("\n== Phase: prompt_sweep ==")
    for pval in phase["n_prompt"]:
        for mkey, mcfg in models.items():
            run_bench(
                cfg["llama_bench"], mcfg,
                {"phase": "prompt_sweep", "n_prompt": pval, "n_gen": phase["n_gen"],
                 "n_threads": mcfg["threads"], "reps": mcfg.get("reps", cfg["default_reps"]),
                 "cooldown_s": cfg.get("cooldown_s", 0),
                 "extra_args": phase.get("extra_args", [])},
                cfg.get("resource_monitor", {}), results_dir, logs_dir, dry_run,
            )
            count += 1
    return count


def run_depth_sweep(cfg, models, results_dir, logs_dir, dry_run):
    phase = cfg["phases"]["depth_sweep"]
    if not phase.get("enabled"):
        return 0
    count = 0
    print("\n== Phase: depth_sweep ==")
    for dval in phase["n_depth"]:
        for mkey, mcfg in models.items():
            run_bench(
                cfg["llama_bench"], mcfg,
                {"phase": "depth_sweep", "n_prompt": phase["n_prompt"], "n_gen": phase["n_gen"],
                 "n_depth": dval, "n_threads": mcfg["threads"],
                 "reps": mcfg.get("reps", cfg["default_reps"]),
                 "cooldown_s": cfg.get("cooldown_s", 0),
                 "extra_args": phase.get("extra_args", [])},
                cfg.get("resource_monitor", {}), results_dir, logs_dir, dry_run,
            )
            count += 1
    return count


def run_thread_sweep(cfg, models, results_dir, logs_dir, dry_run):
    phase = cfg["phases"]["thread_sweep"]
    if not phase.get("enabled"):
        return 0
    count = 0
    print("\n== Phase: thread_sweep ==")
    for tval in phase["threads"]:
        for mkey, mcfg in models.items():
            run_bench(
                cfg["llama_bench"], mcfg,
                {"phase": "thread_sweep", "n_prompt": phase["n_prompt"], "n_gen": phase["n_gen"],
                 "n_threads": tval, "reps": mcfg.get("reps", cfg["default_reps"]),
                 "cooldown_s": cfg.get("cooldown_s", 0),
                 "extra_args": phase.get("extra_args", [])},
                cfg.get("resource_monitor", {}), results_dir, logs_dir, dry_run,
            )
            count += 1
    return count


# ── Main ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="llama-bench profiling sweep")
    parser.add_argument("-c", "--config", default="bench-sweep/config.yaml",
                        help="Path to config YAML (default: bench-sweep/config.yaml)")
    parser.add_argument("--phase", choices=["prompt", "depth", "thread", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["results_dir"])
    logs_dir = Path(cfg["logs_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    models = cfg["models"]
    total = 0

    phase_map = {
        "prompt": [run_prompt_sweep],
        "depth":  [run_depth_sweep],
        "thread": [run_thread_sweep],
        "all":    [run_prompt_sweep, run_depth_sweep, run_thread_sweep],
    }

    for runner in phase_map[args.phase]:
        total += runner(cfg, models, results_dir, logs_dir, args.dry_run)

    print(f"\nDone. {total} runs {'(dry-run)' if args.dry_run else 'completed'}.")
    if not args.dry_run:
        print(f"Results: {results_dir}/")


if __name__ == "__main__":
    main()
