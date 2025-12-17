#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQ2 log extractor (refactored):

Parse from a run log folder:
- GPU monitor CSV (nvidia-smi query output)
- vmstat log
- full training log (*_full_*.log)

Outputs 2 CSV tables into:
  <output_root>/RQ2_summary_<timestamp>/
    - Cost_benefit_summary_<timestamp>.csv
    - System_Overhead_Breakdown_<timestamp>.csv

Usage:
  python rq2_extract.py \
    --input_dir /data/hyan671/yhproject/FakingRecipe/logs/run_nolenslang_fakesv_train_3343_20251128_221302 \
    --output_root /data/hyan671/yhproject/FakingRecipe/lenslang/outputs/data_analysis/BOTH/RQ2
"""

import os
import re
import ast
import glob
import math
import argparse
import datetime
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd


# ------------------------ helpers ------------------------

def now_tag() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def find_one(patterns: List[str], base_dir: str) -> Optional[str]:
    """Return the first matched file path by ordered patterns."""
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(base_dir, pat)))
        if hits:
            return hits[0]
    return None

def parse_duration_to_seconds(s: str) -> Optional[int]:
    """
    Parse durations like:
      "53m 57s"
      "1h 02m 03s"
      "02:15"
      "01:02:03"
    Return seconds or None.
    """
    s = s.strip()

    # hh:mm:ss or mm:ss
    if re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", s):
        h, m, sec = s.split(":")
        return int(h) * 3600 + int(m) * 60 + int(sec)
    if re.fullmatch(r"\d{1,3}:\d{2}", s):
        m, sec = s.split(":")
        return int(m) * 60 + int(sec)

    # 1h 2m 3s / 53m 57s / 116s
    h = m = sec = 0
    mh = re.search(r"(\d+)\s*h", s)
    mm = re.search(r"(\d+)\s*m", s)
    ms = re.search(r"(\d+)\s*s", s)
    if mh or mm or ms:
        if mh: h = int(mh.group(1))
        if mm: m = int(mm.group(1))
        if ms: sec = int(ms.group(1))
        return h * 3600 + m * 60 + sec

    # fallback: pure seconds
    if re.fullmatch(r"\d+", s):
        return int(s)

    return None

def seconds_to_hms(seconds: Optional[float]) -> str:
    if seconds is None or (isinstance(seconds, float) and (math.isnan(seconds) or math.isinf(seconds))):
        return ""
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def safe_percent_to_float(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("%", "").replace(" ", "")
    try:
        return float(s)
    except Exception:
        return np.nan

def safe_mib_to_float(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower().replace("mib", "").replace(" ", "")
    try:
        return float(s)
    except Exception:
        return np.nan

def p95(a: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan
    return float(np.percentile(a, 95))


# ------------------------ GPU CSV ------------------------

def parse_gpu_csv(gpu_csv: str) -> Dict[str, Any]:
    df = pd.read_csv(gpu_csv)

    # normalize columns (allow slight naming differences)
    col_map = {}
    for c in df.columns:
        col_map[c.strip().lower()] = c

    def get_col(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            if cand in col_map:
                return col_map[cand]
        return None

    c_util_gpu = get_col(["utilization.gpu [%]", "utilization.gpu"])
    c_mem_total = get_col(["memory.total [mib]", "memory.total"])
    c_mem_used = get_col(["memory.used [mib]", "memory.used"])

    util_gpu = df[c_util_gpu].apply(safe_percent_to_float).to_numpy() if c_util_gpu else np.array([])
    mem_total = df[c_mem_total].apply(safe_mib_to_float).to_numpy() if c_mem_total else np.array([])
    mem_used = df[c_mem_used].apply(safe_mib_to_float).to_numpy() if c_mem_used else np.array([])

    mem_total_peak_mib = float(np.nanmax(mem_total)) if mem_total.size else np.nan
    mem_used_peak_mib = float(np.nanmax(mem_used)) if mem_used.size else np.nan

    out = {
        "gpu_mem_total_mib": mem_total_peak_mib,
        "gpu_mem_used_peak_mib": mem_used_peak_mib,
        "gpu_util_mean_pct": float(np.nanmean(util_gpu)) if util_gpu.size else np.nan,
        "gpu_util_p95_pct": p95(util_gpu) if util_gpu.size else np.nan,
        "gpu_file": os.path.basename(gpu_csv),
    }
    out["gpu_mem_used_peak_gib"] = out["gpu_mem_used_peak_mib"] / 1024.0 if not np.isnan(out["gpu_mem_used_peak_mib"]) else np.nan
    out["gpu_mem_total_gib"] = out["gpu_mem_total_mib"] / 1024.0 if not np.isnan(out["gpu_mem_total_mib"]) else np.nan
    return out


# ------------------------ vmstat ------------------------

def parse_vmstat(vmstat_log: str) -> Dict[str, Any]:
    """
    Parse vmstat output:
    - identify header line that starts with 'r  b ...'
    - parse numeric rows
    Note: vmstat memory fields are typically in KB on Linux.  [oai_citation:1‡Engineering LibreTexts](https://eng.libretexts.org/Bookshelves/Computer_Science/Operating_Systems/Linux_-_The_Penguin_Marches_On_%28McClanahan%29/08%3A_How_to_Manage_System_Components/5.10%3A_CPU_and_Memory_Troubleshooting/5.10.02%3A_CPU_and_Memory_Troubleshooting_free_vmstat_Commands?utm_source=chatgpt.com)
    """
    rows = []
    cols = None

    with open(vmstat_log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if re.match(r"^r\s+b\s+swpd\s+free\s+buff\s+cache\s+si\s+so\s+bi\s+bo\s+in\s+cs\s+us\s+sy\s+id\s+wa\s+st$", line):
                cols = line.split()
                continue

            if line.startswith("procs") or line.startswith("-----"):
                continue

            if cols is None:
                continue

            parts = line.split()
            if len(parts) != len(cols):
                continue

            try:
                values = [int(x) for x in parts]
            except Exception:
                continue

            rows.append(values)

    if not rows or cols is None:
        return {
            "vmstat_rows": 0,
            "cpu_idle_mean_pct": np.nan,
            "cpu_idle_p95_pct": np.nan,
            "cpu_us_mean_pct": np.nan,
            "cpu_sy_mean_pct": np.nan,
            "cpu_wa_mean_pct": np.nan,
            "ram_free_mean_gib": np.nan,
            "ram_free_p95_gib": np.nan,
            "vmstat_file": os.path.basename(vmstat_log),
        }

    df = pd.DataFrame(rows, columns=cols)

    free_gib = df["free"].to_numpy(dtype=float) / 1024.0 / 1024.0
    cpu_id = df["id"].to_numpy(dtype=float)
    cpu_us = df["us"].to_numpy(dtype=float)
    cpu_sy = df["sy"].to_numpy(dtype=float)
    cpu_wa = df["wa"].to_numpy(dtype=float)

    return {
        "vmstat_rows": int(len(df)),
        "cpu_idle_mean_pct": float(np.mean(cpu_id)),
        "cpu_idle_p95_pct": p95(cpu_id),
        "cpu_us_mean_pct": float(np.mean(cpu_us)),
        "cpu_sy_mean_pct": float(np.mean(cpu_sy)),
        "cpu_wa_mean_pct": float(np.mean(cpu_wa)),
        "ram_free_mean_gib": float(np.mean(free_gib)),
        "ram_free_p95_gib": p95(free_gib),
        "vmstat_file": os.path.basename(vmstat_log),
    }


# ------------------------ full training log ------------------------

def parse_namespace_line(ns_text: str) -> Dict[str, Any]:
    """
    Parse Namespace(k=v, ...) into a dict (best-effort).
    """
    out = {}
    for m in re.finditer(r"(\w+)\s*=\s*([^,]+)", ns_text):
        k = m.group(1).strip()
        vraw = m.group(2).strip().rstrip(")")
        try:
            v = ast.literal_eval(vraw)
        except Exception:
            v = vraw.strip("'").strip('"')
        out[k] = v
    return out

def parse_metrics_dict(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not (line.startswith("{") and line.endswith("}")):
        return None
    try:
        d = ast.literal_eval(line)
        if isinstance(d, dict):
            return d
    except Exception:
        return None
    return None

def parse_full_log(full_log: str) -> Dict[str, Any]:
    """
    Extract:
    - Namespace args
    - data load time
    - per-epoch TRAIN/VAL elapsed from tqdm 100% lines
      (IMPORTANT: TRAIN/VAL only consume the first 100% line after the phase marker)
    - TEST elapsed from test tqdm 100% line (preferred)
      fallback to "Testing complete in ..." only if no test tqdm parsed
    - metrics: best val by f1, last test dict if exists
    """
    ns_args = {}
    data_load_sec = np.nan

    early_stop_epoch = None
    training_total_sec = np.nan
    testing_total_sec = np.nan

    epoch_records = []
    cur_epoch = None
    cur_phase = None  # TRAIN / VAL / TEST
    cur_train_elapsed = None
    cur_val_elapsed = None

    # phase guards (prevent overwrite by later 100% lines)
    expect_train_tqdm = False
    expect_val_tqdm = False

    val_metrics_by_epoch = {}
    test_metrics_last = None

    # test tqdm time (preferred)
    test_tqdm_elapsed_sec = np.nan

    # If test phase marker is missing, test tqdm often becomes "unassigned 100% line"
    unassigned_100_elapsed = []

    # tqdm elapsed extractor: ... [04:28<00:00, 24.41s/it]
    re_tqdm_elapsed = re.compile(r"\[\s*(\d{1,3}:\d{2}(?::\d{2})?)\s*<")

    with open(full_log, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()

            # Namespace
            if "Namespace(" in s:
                m = re.search(r"Namespace\((.*)\)", s)
                if m:
                    ns_args = parse_namespace_line(m.group(1))
                continue

            # data load time: "data load time: 11.02"
            if s.lower().startswith("data load time:"):
                m = re.search(r"data load time:\s*([0-9.]+)", s.lower())
                if m:
                    data_load_sec = float(m.group(1))
                continue

            # explicit early stop
            m = re.search(r"early\s*stop\s*at\s*epoch\s*(\d+)", s, flags=re.IGNORECASE)
            if m:
                early_stop_epoch = int(m.group(1))
                continue

            # Training complete (keep as informational; may be used as fallback)
            m = re.search(r"Training complete in\s+(.+)$", s, flags=re.IGNORECASE)
            if m:
                training_total_sec = parse_duration_to_seconds(m.group(1)) or training_total_sec
                continue

            # Testing complete (fallback only; prefer test tqdm time)
            m = re.search(r"Testing complete in\s+(.+)$", s, flags=re.IGNORECASE)
            if m:
                if np.isnan(testing_total_sec):
                    testing_total_sec = parse_duration_to_seconds(m.group(1)) or testing_total_sec
                continue

            # Epoch marker
            m = re.match(r"Epoch\s+(\d+)\s*/\s*\d+", s)
            if m:
                if cur_epoch is not None:
                    epoch_records.append({
                        "epoch": cur_epoch,
                        "train_elapsed_sec": cur_train_elapsed if cur_train_elapsed is not None else np.nan,
                        "val_elapsed_sec": cur_val_elapsed if cur_val_elapsed is not None else np.nan,
                    })
                cur_epoch = int(m.group(1))
                cur_phase = None
                cur_train_elapsed = None
                cur_val_elapsed = None
                expect_train_tqdm = False
                expect_val_tqdm = False
                continue

            # phase switch markers
            if s == "TRAIN":
                cur_phase = "TRAIN"
                expect_train_tqdm = True
                continue
            if s == "VAL":
                cur_phase = "VAL"
                expect_val_tqdm = True
                continue
            if s == "TEST":
                cur_phase = "TEST"
                continue

            # tqdm 100% lines
            if "100%|" in s:
                mt = re_tqdm_elapsed.search(s)
                if mt:
                    elapsed_str = mt.group(1)
                    sec = parse_duration_to_seconds(elapsed_str)
                    if sec is not None:
                        assigned = False

                        # TRAIN/VAL only consume the first 100% after their marker
                        if cur_phase == "TRAIN" and expect_train_tqdm:
                            cur_train_elapsed = sec
                            expect_train_tqdm = False
                            assigned = True
                        elif cur_phase == "VAL" and expect_val_tqdm:
                            cur_val_elapsed = sec
                            expect_val_tqdm = False
                            assigned = True
                        elif cur_phase == "TEST":
                            test_tqdm_elapsed_sec = sec  # keep last seen test tqdm elapsed
                            assigned = True

                        # If not assigned, keep it; useful to recover test time when TEST marker is absent
                        if not assigned:
                            unassigned_100_elapsed.append(sec)

                continue

            # metrics dict lines
            d = parse_metrics_dict(s)
            if d is not None:
                if cur_phase == "VAL" and cur_epoch is not None:
                    val_metrics_by_epoch[cur_epoch] = d
                if cur_phase == "TEST":
                    test_metrics_last = d

    # flush last epoch
    if cur_epoch is not None:
        epoch_records.append({
            "epoch": cur_epoch,
            "train_elapsed_sec": cur_train_elapsed if cur_train_elapsed is not None else np.nan,
            "val_elapsed_sec": cur_val_elapsed if cur_val_elapsed is not None else np.nan,
        })

    epochs_run = len({r["epoch"] for r in epoch_records}) if epoch_records else 0

    if early_stop_epoch is None and epochs_run > 0:
        early_stop_epoch = int(max(r["epoch"] for r in epoch_records))

    train_total_from_epochs = float(np.nansum([r["train_elapsed_sec"] for r in epoch_records])) if epoch_records else np.nan
    val_total_from_epochs = float(np.nansum([r["val_elapsed_sec"] for r in epoch_records])) if epoch_records else np.nan

    # prefer sum of epoch train times if "Training complete" missing
    if np.isnan(training_total_sec) and not np.isnan(train_total_from_epochs):
        training_total_sec = train_total_from_epochs

    # TEST time: prefer test tqdm; else try last unassigned 100%; else fallback to "Testing complete in"
    if not np.isnan(test_tqdm_elapsed_sec):
        test_total_sec_final = test_tqdm_elapsed_sec
    elif unassigned_100_elapsed:
        test_total_sec_final = float(unassigned_100_elapsed[-1])
    else:
        test_total_sec_final = testing_total_sec

    best_val_epoch = None
    best_val = None
    if val_metrics_by_epoch:
        def key_f1(item):
            epoch, dct = item
            return float(dct.get("f1", -1.0))
        best_val_epoch, best_val = max(val_metrics_by_epoch.items(), key=key_f1)

    return {
        "namespace": ns_args,
        "data_load_sec": data_load_sec,
        "epochs_run": epochs_run,
        "early_stop_epoch": early_stop_epoch,
        "training_total_sec": training_total_sec,
        "val_total_from_epochs_sec": val_total_from_epochs,
        "test_total_sec": test_total_sec_final,
        "epoch_records": epoch_records,
        "best_val_epoch": best_val_epoch,
        "best_val_metrics": best_val,
        "test_metrics_last": test_metrics_last,
        "full_log_file": os.path.basename(full_log),
        "unassigned_100_count": int(len(unassigned_100_elapsed)),
    }


# ------------------------ build tables ------------------------

def build_tables(input_dir: str, gpu_csv: str, vmstat_log: str, full_log: str, output_root: str) -> str:
    ts = now_tag()
    out_dir = os.path.join(output_root, f"RQ2_summary_{ts}")
    ensure_dir(out_dir)

    gpu_info = parse_gpu_csv(gpu_csv) if gpu_csv else {}
    vm_info = parse_vmstat(vmstat_log) if vmstat_log else {}
    log_info = parse_full_log(full_log) if full_log else {}

    # ---------------- Table 1: Cost_benefit_summary (GPU-only header) ----------------
    peak_mem_gib = gpu_info.get("gpu_mem_used_peak_gib", np.nan)
    total_mem_gib = gpu_info.get("gpu_mem_total_gib", np.nan)

    if (not np.isnan(peak_mem_gib)) and (not np.isnan(total_mem_gib)) and total_mem_gib > 0:
        peak_mem_pct = (peak_mem_gib / total_mem_gib) * 100.0
    else:
        peak_mem_pct = np.nan

    cost_benefit = pd.DataFrame([{
        "Peak Mem (GiB)": peak_mem_gib,
        "Peak Mem (%)": peak_mem_pct,
        "Util Mean (%)": gpu_info.get("gpu_util_mean_pct", np.nan),
        "Util P95 (%)": gpu_info.get("gpu_util_p95_pct", np.nan),
        "Source": gpu_csv,  # full path
    }])

    # ---------------- Table 2: System_Overhead_Breakdown ----------------
    dataset = ""
    model = ""
    ns = log_info.get("namespace", {}) or {}
    if isinstance(ns.get("dataset", None), str):
        dl = ns["dataset"].lower()
        if dl == "fakesv":
            dataset = "FakeSV"
        elif dl == "fakett":
            dataset = "FakeTT"

    # model is not required by your new spec; keep best-effort but non-blocking
    text_for_guess = (input_dir + " " + os.path.basename(full_log)).lower()
    if "nolenslang" in text_for_guess or "no_lenslang" in text_for_guess:
        model = "M1"
    elif ("lenslang_only" in text_for_guess) or ("onlylenslang" in text_for_guess) or ("lenslang-only" in text_for_guess):
        model = "M3"
    elif "lenslang" in text_for_guess:
        model = "M2"

    test_total_sec = log_info.get("test_total_sec", np.nan)

    epoch_df = pd.DataFrame(log_info.get("epoch_records", []))
    if epoch_df.empty:
        overhead = pd.DataFrame([{
            "dataset": dataset,
            "model": model,
            "epoch": "ALL",
            "train_elapsed_hms": "",
            "train_elapsed_sec": np.nan,
            "val_elapsed_hms": "",
            "val_elapsed_sec": np.nan,
            "test_elapsed_hms": seconds_to_hms(test_total_sec),
            "test_elapsed_sec": test_total_sec,
            "notes": "No per-epoch tqdm elapsed parsed; check full log format.",
        }])
    else:
        epoch_df["train_elapsed_hms"] = epoch_df["train_elapsed_sec"].apply(seconds_to_hms)
        epoch_df["val_elapsed_hms"] = epoch_df["val_elapsed_sec"].apply(seconds_to_hms)
        epoch_df.insert(0, "model", model)
        epoch_df.insert(0, "dataset", dataset)

        summary_row = {
            "dataset": dataset,
            "model": model,
            "epoch": "ALL",
            "train_elapsed_sec": float(np.nansum(epoch_df["train_elapsed_sec"].to_numpy(dtype=float))),
            "val_elapsed_sec": float(np.nansum(epoch_df["val_elapsed_sec"].to_numpy(dtype=float))),
        }
        summary_row["train_elapsed_hms"] = seconds_to_hms(summary_row["train_elapsed_sec"])
        summary_row["val_elapsed_hms"] = seconds_to_hms(summary_row["val_elapsed_sec"])
        summary_row["test_elapsed_hms"] = seconds_to_hms(test_total_sec)
        summary_row["test_elapsed_sec"] = test_total_sec
        summary_row["notes"] = (
            "TRAIN/VAL elapsed: first tqdm 100% after each phase marker; "
            "TEST elapsed: test tqdm 100% preferred (fallback to last unassigned 100% / 'Testing complete in')."
        )
        overhead = pd.concat([epoch_df, pd.DataFrame([summary_row])], ignore_index=True)

    # write CSVs
    cost_path = os.path.join(out_dir, f"Cost_benefit_summary_{ts}.csv")
    overhead_path = os.path.join(out_dir, f"System_Overhead_Breakdown_{ts}.csv")

    cost_benefit.to_csv(cost_path, index=False)
    overhead.to_csv(overhead_path, index=False)

    print(f"[OK] Output folder: {out_dir}")
    print(f"  - {os.path.basename(cost_path)}")
    print(f"  - {os.path.basename(overhead_path)}")

    return out_dir


# ------------------------ main ------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/data/hyan671/yhproject/FakingRecipe/logs/run_lenslang_m3_fakett_3461_20251205_213631_round7",
        help="Folder containing gpu csv, vmstat log, and full training log."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/data/hyan671/yhproject/FakingRecipe/lenslang/outputs/data_analysis/BOTH/RQ2",
        help="Root folder to create RQ2_summary_<timestamp>/"
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_root = args.output_root

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    ensure_dir(output_root)

    gpu_csv = find_one(["*gpu*.csv", "*GPU*.csv", "*.gpu*.csv"], input_dir)
    vmstat_log = find_one(["*vmstat*.log", "*vmstat*.txt"], input_dir)
    full_log = find_one(["*_full_*.log", "*train_full*.log", "*_full_*.txt"], input_dir)

    missing = []
    if not gpu_csv:
        missing.append("gpu csv (*gpu*.csv)")
    if not vmstat_log:
        missing.append("vmstat log (*vmstat*.log)")
    if not full_log:
        missing.append("full training log (*_full_*.log)")

    if missing:
        raise FileNotFoundError(
            "Missing required file(s) in input_dir:\n  - " + "\n  - ".join(missing)
        )

    build_tables(
        input_dir=input_dir,
        gpu_csv=gpu_csv,
        vmstat_log=vmstat_log,
        full_log=full_log,
        output_root=output_root
    )


if __name__ == "__main__":
    main()