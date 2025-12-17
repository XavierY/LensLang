#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LensLang distribution shift analysis: FakeSV vs FakeTT

Per-field outputs:
- CSV: per-field stats / proportions + PSI / KS / Wasserstein (when applicable)
- FIG:
  - numeric: histogram (density, shared bins) + ECDF
  - categorical: side-by-side proportions (top-K + OTHER), always include exclusive values

Folders:
<OUT_ROOT>/Dataset_PSI_<timestamp>/{csv,fig}

Default OUT_ROOT:
  /data/hyan671/yhproject/FakingRecipe/lenslang/outputs/data_analysis/BOTH/SVTT_PSI
Default FakeSV JSON DIR:
  /data/hyan671/yhproject/FakingRecipe/lenslang/outputs/lenslang_FakeSV_no_keyChts_3624only_Complete
Default FakeTT JSON DIR:
  /data/hyan671/yhproject/FakingRecipe/lenslang/outputs/lenslang_FakeTT_no_keyChts_1991only_Complete_ENG
"""

import os
import re
import glob
import json
import argparse
import datetime
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# ---- headless-safe backend (must be before pyplot import) ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, wasserstein_distance

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="matplotlib")

MISSING_TOKEN = "__MISSING__"
EMPTY_TOKEN = "__EMPTY__"

# ignore fields
IGNORE_LEAF_FIELDS = {"video_id", "video_name", "style_tags"}  # user required


# -------------------------
# IO helpers
# -------------------------
def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_json_files(json_dir: str):
    return sorted(glob.glob(os.path.join(json_dir, "*.json")))


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def sanitize_filename(s: str, maxlen: int = 180):
    s = s.replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "_", s)
    return s[:maxlen]


# -------------------------
# Plot helpers (always save a figure)
# -------------------------
def save_placeholder_fig(out_png: str, title: str, message: str):
    fig = plt.figure(figsize=(8, 4))
    plt.axis("off")
    plt.title(title)
    plt.text(0.01, 0.6, message, fontsize=11)
    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# -------------------------
# Flatten JSON fields
# -------------------------
def flatten_leaves(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_leaves(v, new_prefix))
    elif isinstance(obj, list):
        out[prefix] = obj
    else:
        out[prefix] = obj
    return out


def extract_group_fields(j, group_name: str):
    if group_name not in j or j[group_name] is None:
        return {}
    return flatten_leaves(j[group_name], prefix=group_name)


def should_ignore_field(field_path: str) -> bool:
    # field_path like "stats.xxx.yyy" or "analysis.xxx.yyy"
    leaf = field_path.split(".")[-1]
    if leaf in IGNORE_LEAF_FIELDS:
        return True
    # be robust: ignore any path containing ".style_tags"
    if ".style_tags" in field_path:
        return True
    return False


# -------------------------
# Type inference
# -------------------------
def infer_field_kind(values_sv, values_tt):
    sample = None
    for v in list(values_sv) + list(values_tt):
        if v is not None:
            sample = v
            break
    if sample is None:
        return "all_missing"
    if isinstance(sample, list):
        return "categorical_multilabel"

    s_all = pd.Series(list(values_sv) + list(values_tt))
    if any(isinstance(v, dict) for v in s_all.dropna()):
        return "mixed_skip"

    num = pd.to_numeric(s_all, errors="coerce")
    nonnull = s_all.notna().sum()
    numeric_ok = num.notna().sum()
    if nonnull > 0 and numeric_ok / nonnull >= 0.95:
        return "numeric_scalar"
    return "categorical_scalar"


# -------------------------
# PSI (probability-based)
# -------------------------
def psi_from_probs(p_ref, p_tgt, eps=1e-6):
    p_ref = np.asarray(p_ref, dtype=float)
    p_tgt = np.asarray(p_tgt, dtype=float)
    p_ref = np.clip(p_ref, eps, None)
    p_tgt = np.clip(p_tgt, eps, None)
    return float(np.sum((p_ref - p_tgt) * np.log(p_ref / p_tgt)))


def psi_numeric(ref_vals, tgt_vals, bins=10, winsor_lo=0.01, winsor_hi=0.99, eps=1e-6):
    ref = pd.to_numeric(pd.Series(ref_vals), errors="coerce").dropna().values
    tgt = pd.to_numeric(pd.Series(tgt_vals), errors="coerce").dropna().values

    if len(ref) == 0 or len(tgt) == 0:
        return np.nan, None, None, None, None, None, None, None

    ql = np.quantile(ref, winsor_lo)
    qh = np.quantile(ref, winsor_hi)
    if not np.isfinite(ql) or not np.isfinite(qh):
        ql, qh = np.nanmin(ref), np.nanmax(ref)

    if ql == qh:
        ql -= 0.5
        qh += 0.5

    ref_w = np.clip(ref, ql, qh)
    tgt_w = np.clip(tgt, ql, qh)

    qs = np.linspace(0, 1, bins + 1)
    edges_mid = np.unique(np.quantile(ref_w, qs))
    if len(edges_mid) < 2:
        edges_mid = np.array([ql, qh], dtype=float)
    if edges_mid[0] == edges_mid[-1]:
        edges_mid = np.array([edges_mid[0] - 0.5, edges_mid[-1] + 0.5], dtype=float)

    # finite edges for plotting / hist
    edges_plot = np.concatenate(([ql], edges_mid[1:-1], [qh]))

    ref_hist, _ = np.histogram(ref_w, bins=edges_plot)
    tgt_hist, _ = np.histogram(tgt_w, bins=edges_plot)

    ref_p = ref_hist / max(ref_hist.sum(), 1)
    tgt_p = tgt_hist / max(tgt_hist.sum(), 1)

    psi = psi_from_probs(ref_p, tgt_p, eps=eps)
    return psi, edges_plot, ref_hist, tgt_hist, ref_p, tgt_p, ql, qh


def psi_categorical_from_counters(c_sv: Counter, c_tt: Counter, eps=1e-6):
    cats = sorted(set(c_sv.keys()) | set(c_tt.keys()))
    total_sv = sum(c_sv.values()) or 1
    total_tt = sum(c_tt.values()) or 1
    ref_p = np.array([c_sv.get(c, 0) / total_sv for c in cats], dtype=float)
    tgt_p = np.array([c_tt.get(c, 0) / total_tt for c in cats], dtype=float)
    psi = psi_from_probs(ref_p, tgt_p, eps=eps)
    return psi, cats, ref_p, tgt_p, total_sv, total_tt


# -------------------------
# Plotting
# -------------------------
def plot_numeric(field_name, ref_vals, tgt_vals, edges_plot, out_png, title_extra=""):
    ref = pd.to_numeric(pd.Series(ref_vals), errors="coerce").dropna().values
    tgt = pd.to_numeric(pd.Series(tgt_vals), errors="coerce").dropna().values

    if edges_plot is None or len(edges_plot) < 2 or len(ref) == 0 or len(tgt) == 0:
        save_placeholder_fig(out_png, f"{field_name} (Numeric) {title_extra}".strip(), "No sufficient numeric data to plot.")
        return

    ql, qh = float(edges_plot[0]), float(edges_plot[-1])
    ref = np.clip(ref, ql, qh)
    tgt = np.clip(tgt, ql, qh)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # density histogram (NOT raw counts) to make different N comparable
    axes[0].hist(ref, bins=edges_plot, alpha=0.5, label="FakeSV", density=True)
    axes[0].hist(tgt, bins=edges_plot, alpha=0.5, label="FakeTT", density=True)
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].set_title(f"{field_name} (Numeric) {title_extra}".strip())

    def ecdf(x):
        x = np.sort(x)
        y = np.arange(1, len(x) + 1) / len(x)
        return x, y

    x1, y1 = ecdf(ref)
    x2, y2 = ecdf(tgt)
    axes[1].plot(x1, y1, label="FakeSV")
    axes[1].plot(x2, y2, label="FakeTT")
    axes[1].set_xlabel("Value (winsorized to FakeSV ref range)")
    axes[1].set_ylabel("ECDF")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_categorical_pct(field_name,
                         counts_sv: Counter,
                         counts_tt: Counter,
                         out_png: str,
                         topk: int = 30,
                         title_extra: str = "",
                         force_include=None,
                         max_show: int = 60):
    """
    Plot percentage bars (NOT raw counts).
    """
    if force_include is None:
        force_include = set()
    else:
        force_include = set(force_include)

    all_cats = list(set(counts_sv.keys()) | set(counts_tt.keys()))
    if len(all_cats) == 0:
        save_placeholder_fig(out_png, f"{field_name} (Categorical) {title_extra}".strip(), "No categorical labels to plot.")
        return

    all_cats_sorted = sorted(all_cats, key=lambda c: counts_sv.get(c, 0) + counts_tt.get(c, 0), reverse=True)
    show = list(all_cats_sorted[:topk])

    for c in force_include:
        if c not in show:
            show.append(c)

    if len(show) > max_show:
        show = sorted(show, key=lambda c: counts_sv.get(c, 0) + counts_tt.get(c, 0), reverse=True)[:max_show]

    other = [c for c in all_cats_sorted if c not in show]

    def compress(counter):
        new = Counter()
        for c in show:
            new[c] = counter.get(c, 0)
        if other:
            new["OTHER"] = sum(counter.get(c, 0) for c in other)
        return new

    c_sv = compress(counts_sv)
    c_tt = compress(counts_tt)

    cats = list(c_sv.keys())
    total_sv = sum(c_sv.values()) or 1
    total_tt = sum(c_tt.values()) or 1

    sv_pct = [c_sv[c] / total_sv for c in cats]
    tt_pct = [c_tt[c] / total_tt for c in cats]

    x = np.arange(len(cats), dtype=float)
    width = 0.4

    fig = plt.figure(figsize=(max(10, len(cats) * 0.35), 6))
    plt.bar(x - width / 2, sv_pct, width=width, label="FakeSV")
    plt.bar(x + width / 2, tt_pct, width=width, label="FakeTT")
    plt.xticks(x, cats, rotation=45, ha="right")
    plt.ylabel("Proportion")
    plt.title(f"{field_name} (Categorical) {title_extra}".strip())
    plt.legend()
    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# -------------------------
# Data loading (FIXED: strict per-file alignment)
# -------------------------
def build_field_store_strict(json_dir: str, group_names=("stats", "analysis")):
    """
    Returns:
      store[group][field] = list aligned by file index (len == n_files)
      n_files
    """
    files = list_json_files(json_dir)
    records = {g: [] for g in group_names}

    for p in files:
        try:
            j = read_json(p)
        except Exception:
            # skip unreadable
            continue

        for g in group_names:
            flat = extract_group_fields(j, g)
            # drop ignored fields here (saves memory and avoids later bugs)
            flat = {k: v for k, v in flat.items() if not should_ignore_field(k)}
            records[g].append(flat)

    store = {g: {} for g in group_names}
    for g in group_names:
        # union fields across all records in this dataset/group
        all_fields = set()
        for r in records[g]:
            all_fields.update(r.keys())

        all_fields = sorted(all_fields)
        for f in all_fields:
            store[g][f] = [r.get(f, None) for r in records[g]]  # None means missing in that file

    return store, len(records[group_names[0]])


# -------------------------
# Metrics
# -------------------------
def compute_numeric_metrics(ref_vals, tgt_vals):
    ref = pd.to_numeric(pd.Series(ref_vals), errors="coerce")
    tgt = pd.to_numeric(pd.Series(tgt_vals), errors="coerce")

    ref_non = ref.dropna().values
    tgt_non = tgt.dropna().values

    def summarize(x):
        if len(x) == 0:
            return {}
        q = np.quantile(x, [0, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1.0])
        return {
            "n": int(len(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
            "min": float(q[0]),
            "p01": float(q[1]),
            "p05": float(q[2]),
            "p25": float(q[3]),
            "p50": float(q[4]),
            "p75": float(q[5]),
            "p95": float(q[6]),
            "p99": float(q[7]),
            "max": float(q[8]),
        }

    out = {
        "sv_missing_rate": float(ref.isna().mean()),
        "tt_missing_rate": float(tgt.isna().mean()),
    }
    out.update({f"sv_{k}": v for k, v in summarize(ref_non).items()})
    out.update({f"tt_{k}": v for k, v in summarize(tgt_non).items()})

    if len(ref_non) >= 2 and len(tgt_non) >= 2:
        out["ks_stat"] = float(ks_2samp(ref_non, tgt_non).statistic)
        out["wasserstein"] = float(wasserstein_distance(ref_non, tgt_non))
    else:
        out["ks_stat"] = np.nan
        out["wasserstein"] = np.nan
    return out


# -------------------------
# Categorical summarizers (NO "unknown"; use tokens)
# -------------------------
def summarize_categorical_scalar(vals):
    """
    Each sample contributes exactly one value or MISSING_TOKEN.
    """
    s = pd.Series(vals)
    missing_count = int(s.isna().sum())
    observed = s.dropna().astype(str).tolist()

    c = Counter(observed)
    if missing_count > 0:
        c[MISSING_TOKEN] += missing_count

    # option set excludes missing token
    opt_set = set(observed)
    total = sum(c.values()) or 1
    return c, opt_set, total


def summarize_categorical_multilabel(vals):
    """
    Each sample contributes:
      - MISSING_TOKEN if None
      - EMPTY_TOKEN if [] 
      - otherwise: unique labels in that list (dedup within a sample)
    Counter sums to "total presences" (video-label pairs + missing + empty).
    """
    c = Counter()
    opt_set = set()

    missing_samples = 0
    empty_samples = 0

    for v in vals:
        if v is None:
            missing_samples += 1
            continue
        if isinstance(v, list):
            if len(v) == 0:
                empty_samples += 1
                continue
            uniq = set(str(x) for x in v)
            for lab in uniq:
                c[lab] += 1
                opt_set.add(lab)
        else:
            lab = str(v)
            c[lab] += 1
            opt_set.add(lab)

    if missing_samples > 0:
        c[MISSING_TOKEN] += missing_samples
    if empty_samples > 0:
        c[EMPTY_TOKEN] += empty_samples

    total = sum(c.values()) or 1
    return c, opt_set, total


# -------------------------
# Main analysis
# -------------------------
def analyze_and_export(
    sv_dir,
    tt_dir,
    out_root,
    bins,
    winsor_lo,
    winsor_hi,
    topk_cats,
    min_samples
):
    store_sv, n_sv = build_field_store_strict(sv_dir)
    store_tt, n_tt = build_field_store_strict(tt_dir)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, f"SVTT_PSI_percent{ts}")
    csv_dir = os.path.join(run_dir, "csv")
    fig_dir = os.path.join(run_dir, "fig")
    ensure_dir(csv_dir)
    ensure_dir(fig_dir)

    summary_rows = []

    # NEW: enum summaries (pct)
    enum_rows = []       # dataset | group | field | kind | value | pct
    mismatch_rows = []   # dataset | group | field | kind | value | mismatch_pct

    for group in ("stats", "analysis"):
        fields = sorted(set(store_sv[group].keys()) | set(store_tt[group].keys()))
        for field in fields:
            if should_ignore_field(field):
                continue

            vals_sv = store_sv[group].get(field, [None] * n_sv)
            vals_tt = store_tt[group].get(field, [None] * n_tt)

            kind = infer_field_kind(vals_sv, vals_tt)

            safe_field = sanitize_filename(field)
            base_name = f"{group}__{safe_field}"
            out_csv = os.path.join(csv_dir, f"{base_name}.csv")
            out_png = os.path.join(fig_dir, f"{base_name}.png")

            # --- numeric scalar ---
            if kind == "numeric_scalar":
                sv_num = pd.to_numeric(pd.Series(vals_sv), errors="coerce")
                tt_num = pd.to_numeric(pd.Series(vals_tt), errors="coerce")
                sv_non = sv_num.dropna().values
                tt_non = tt_num.dropna().values

                if len(sv_non) < min_samples or len(tt_non) < min_samples:
                    pd.DataFrame([{
                        "group": group,
                        "field": field,
                        "kind": kind,
                        "note": f"insufficient samples (sv={len(sv_non)}, tt={len(tt_non)})"
                    }]).to_csv(out_csv, index=False)
                    save_placeholder_fig(out_png, f"{field} (Numeric)", f"Insufficient samples.\nFakeSV n={len(sv_non)}, FakeTT n={len(tt_non)}")
                    continue

                psi, edges_plot, ref_hist, tgt_hist, ref_p, tgt_p, ql, qh = psi_numeric(
                    sv_non, tt_non, bins=bins, winsor_lo=winsor_lo, winsor_hi=winsor_hi
                )

                met = compute_numeric_metrics(sv_non, tt_non)
                met.update({
                    "group": group,
                    "field": field,
                    "kind": kind,
                    "psi": float(psi),
                    "bins": int(bins),
                    "winsor_lo": float(winsor_lo),
                    "winsor_hi": float(winsor_hi),
                    "plot_range_lo": float(ql),
                    "plot_range_hi": float(qh),
                })

                # per-bin output: ONLY proportions (no absolute counts)
                bin_rows = []
                if edges_plot is not None and ref_p is not None and tgt_p is not None:
                    for i in range(len(ref_p)):
                        bin_rows.append({
                            "section": "bin",
                            "bin_left": float(edges_plot[i]),
                            "bin_right": float(edges_plot[i + 1]),
                            "sv_pct": float(ref_p[i]),
                            "tt_pct": float(tgt_p[i]),
                            "pct_diff": float(tgt_p[i] - ref_p[i]),
                        })

                df_meta = pd.DataFrame([{"section": "summary", **met}])
                df_bins = pd.DataFrame(bin_rows) if bin_rows else pd.DataFrame([{"section": "bin", "note": "no bins"}])
                pd.concat([df_meta, df_bins], ignore_index=True).to_csv(out_csv, index=False)

                title_extra = f"| PSI={psi:.4f} KS={met.get('ks_stat', np.nan):.4f} W={met.get('wasserstein', np.nan):.4f}"
                plot_numeric(field, sv_non, tt_non, edges_plot, out_png, title_extra)

                summary_rows.append({
                    "group": group,
                    "field": field,
                    "kind": kind,
                    "psi": float(psi),
                    "ks_stat": met.get("ks_stat", np.nan),
                    "wasserstein": met.get("wasserstein", np.nan),
                    "sv_missing_rate": met.get("sv_missing_rate", np.nan),
                    "tt_missing_rate": met.get("tt_missing_rate", np.nan),
                    "sv_n": int(len(sv_non)),
                    "tt_n": int(len(tt_non)),
                })

            # --- categorical scalar ---
            elif kind == "categorical_scalar":
                c_sv, set_sv, total_sv = summarize_categorical_scalar(vals_sv)
                c_tt, set_tt, total_tt = summarize_categorical_scalar(vals_tt)

                psi, cats, ref_p, tgt_p, _, _ = psi_categorical_from_counters(c_sv, c_tt)

                # per-field csv: ONLY proportions
                rows = []
                for c in cats:
                    sv_pct = c_sv.get(c, 0) / total_sv
                    tt_pct = c_tt.get(c, 0) / total_tt
                    rows.append({
                        "group": group,
                        "field": field,
                        "kind": kind,
                        "value": c,
                        "sv_pct": float(sv_pct),
                        "tt_pct": float(tt_pct),
                        "pct_diff": float(tt_pct - sv_pct),
                        "psi_total": float(psi),
                    })
                pd.DataFrame(rows).to_csv(out_csv, index=False)

                # enum summary (pct within dataset/field)
                for v, cnt in c_sv.items():
                    enum_rows.append({
                        "dataset": "FakeSV",
                        "group": group,
                        "field": field,
                        "kind": kind,
                        "value": v,
                        "pct": float(cnt / total_sv),
                    })
                for v, cnt in c_tt.items():
                    enum_rows.append({
                        "dataset": "FakeTT",
                        "group": group,
                        "field": field,
                        "kind": kind,
                        "value": v,
                        "pct": float(cnt / total_tt),
                    })

                # mismatch (exclude missing token from option-set comparison)
                excl_sv = sorted(list(set_sv - set_tt))
                excl_tt = sorted(list(set_tt - set_sv))
                for v in excl_sv:
                    mismatch_rows.append({
                        "dataset": "FakeSV",
                        "group": group,
                        "field": field,
                        "kind": kind,
                        "value": v,
                        "mismatch_pct": float(c_sv.get(v, 0) / total_sv),
                    })
                for v in excl_tt:
                    mismatch_rows.append({
                        "dataset": "FakeTT",
                        "group": group,
                        "field": field,
                        "kind": kind,
                        "value": v,
                        "mismatch_pct": float(c_tt.get(v, 0) / total_tt),
                    })

                force_include = set(excl_sv) | set(excl_tt)
                title_extra = f"| PSI={psi:.4f}"
                plot_categorical_pct(field, c_sv, c_tt, out_png, topk=topk_cats, title_extra=title_extra, force_include=force_include)

                summary_rows.append({
                    "group": group,
                    "field": field,
                    "kind": kind,
                    "psi": float(psi),
                    "ks_stat": np.nan,
                    "wasserstein": np.nan,
                    "sv_missing_rate": float(pd.Series(vals_sv).isna().mean()),
                    "tt_missing_rate": float(pd.Series(vals_tt).isna().mean()),
                    "sv_n": int(total_sv),
                    "tt_n": int(total_tt),
                })

            # --- categorical multilabel ---
            elif kind == "categorical_multilabel":
                c_sv, set_sv, total_sv = summarize_categorical_multilabel(vals_sv)
                c_tt, set_tt, total_tt = summarize_categorical_multilabel(vals_tt)

                psi, cats, ref_p, tgt_p, _, _ = psi_categorical_from_counters(c_sv, c_tt)

                rows = []
                for c in cats:
                    sv_pct = c_sv.get(c, 0) / total_sv
                    tt_pct = c_tt.get(c, 0) / total_tt
                    rows.append({
                        "group": group,
                        "field": field,
                        "kind": kind,
                        "value": c,
                        "sv_pct": float(sv_pct),
                        "tt_pct": float(tt_pct),
                        "pct_diff": float(tt_pct - sv_pct),
                        "psi_total": float(psi),
                    })
                pd.DataFrame(rows).sort_values(by=["sv_pct", "tt_pct"], ascending=False).to_csv(out_csv, index=False)

                # enum summary (pct)
                for v, cnt in c_sv.items():
                    enum_rows.append({
                        "dataset": "FakeSV",
                        "group": group,
                        "field": field,
                        "kind": kind,
                        "value": v,
                        "pct": float(cnt / total_sv),
                    })
                for v, cnt in c_tt.items():
                    enum_rows.append({
                        "dataset": "FakeTT",
                        "group": group,
                        "field": field,
                        "kind": kind,
                        "value": v,
                        "pct": float(cnt / total_tt),
                    })

                # mismatch (exclude missing/empty tokens)
                excl_sv = sorted(list(set_sv - set_tt))
                excl_tt = sorted(list(set_tt - set_sv))
                for v in excl_sv:
                    mismatch_rows.append({
                        "dataset": "FakeSV",
                        "group": group,
                        "field": field,
                        "kind": kind,
                        "value": v,
                        "mismatch_pct": float(c_sv.get(v, 0) / total_sv),
                    })
                for v in excl_tt:
                    mismatch_rows.append({
                        "dataset": "FakeTT",
                        "group": group,
                        "field": field,
                        "kind": kind,
                        "value": v,
                        "mismatch_pct": float(c_tt.get(v, 0) / total_tt),
                    })

                force_include = set(excl_sv) | set(excl_tt)
                title_extra = f"| PSI={psi:.4f}"
                plot_categorical_pct(field, c_sv, c_tt, out_png, topk=topk_cats, title_extra=title_extra, force_include=force_include)

                summary_rows.append({
                    "group": group,
                    "field": field,
                    "kind": kind,
                    "psi": float(psi),
                    "ks_stat": np.nan,
                    "wasserstein": np.nan,
                    "sv_missing_rate": float(pd.Series(vals_sv).isna().mean()),
                    "tt_missing_rate": float(pd.Series(vals_tt).isna().mean()),
                    "sv_n": int(total_sv),
                    "tt_n": int(total_tt),
                })

            else:
                pd.DataFrame([{
                    "group": group,
                    "field": field,
                    "kind": kind,
                    "note": "skip (all missing or mixed/unsupported leaf type)"
                }]).to_csv(out_csv, index=False)
                save_placeholder_fig(out_png, f"{field} ({kind})", "Skipped field: all missing or unsupported type.")

    # group-level summaries
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows).sort_values(by="psi", ascending=False)
        df_sum.to_csv(os.path.join(csv_dir, "summary__field_shift_rank.csv"), index=False)

        top = df_sum.head(30).iloc[::-1]
        fig = plt.figure(figsize=(10, max(6, 0.35 * len(top))))
        plt.barh(top["field"], top["psi"])
        plt.xlabel("PSI (FakeSV as reference)")
        plt.title("Top shifted LensLang fields (FakeSV vs FakeTT)")
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, "summary__top_shift_fields.png"), dpi=200)
        plt.close(fig)

    # enum options (pct only)
    if enum_rows:
        df_enum = pd.DataFrame(enum_rows)
        df_enum = df_enum.sort_values(by=["group", "field", "dataset", "pct"], ascending=[True, True, True, False])
        df_enum.to_csv(os.path.join(csv_dir, "summary__enum_options_by_dataset.csv"), index=False)
    else:
        pd.DataFrame(columns=["dataset", "group", "field", "kind", "value", "pct"]).to_csv(
            os.path.join(csv_dir, "summary__enum_options_by_dataset.csv"), index=False
        )

    if mismatch_rows:
        df_mis = pd.DataFrame(mismatch_rows)
        df_mis = df_mis.sort_values(by=["group", "field", "dataset", "mismatch_pct"],
                                    ascending=[True, True, True, False])
        df_mis.to_csv(os.path.join(csv_dir, "summary__enum_mismatch_values.csv"), index=False)
    else:
        pd.DataFrame(columns=["dataset", "group", "field", "kind", "value", "mismatch_pct"]).to_csv(
            os.path.join(csv_dir, "summary__enum_mismatch_values.csv"), index=False
        )

    print(f"[OK] Output written to: {run_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sv-dir",
        default="/data/hyan671/yhproject/FakingRecipe/lenslang/outputs/lenslang_FakeSV_no_keyChts_3624only_Complete",
        help="FakeSV LensLang JSON directory"
    )
    ap.add_argument(
        "--tt-dir",
        default="/data/hyan671/yhproject/FakingRecipe/lenslang/outputs/lenslang_FakeTT_no_keyChts_1991only_Complete_ENG",
        help="FakeTT LensLang JSON directory"
    )
    ap.add_argument(
        "--out-root",
        default="/data/hyan671/yhproject/FakingRecipe/lenslang/outputs/data_analysis/BOTH/SVTT_PSI",
        help="Root output directory"
    )
    ap.add_argument("--bins", type=int, default=10, help="Bins for numeric PSI/hist")
    ap.add_argument("--winsor-lo", type=float, default=0.01, help="Winsorization lower quantile (ref=FakeSV)")
    ap.add_argument("--winsor-hi", type=float, default=0.99, help="Winsorization upper quantile (ref=FakeSV)")
    ap.add_argument("--topk-cats", type=int, default=30, help="Top-K categories to show (others -> OTHER), exclusives are forced in")
    ap.add_argument("--min-samples", type=int, default=50, help="Minimum non-null samples for numeric shift")
    args = ap.parse_args()

    if not os.path.isdir(args.sv_dir):
        raise FileNotFoundError(f"FakeSV dir not found: {args.sv_dir}")
    if not os.path.isdir(args.tt_dir):
        raise FileNotFoundError(f"FakeTT dir not found: {args.tt_dir}")

    ensure_dir(args.out_root)

    analyze_and_export(
        sv_dir=args.sv_dir,
        tt_dir=args.tt_dir,
        out_root=args.out_root,
        bins=args.bins,
        winsor_lo=args.winsor_lo,
        winsor_hi=args.winsor_hi,
        topk_cats=args.topk_cats,
        min_samples=args.min_samples
    )


if __name__ == "__main__":
    main()