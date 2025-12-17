#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FakeSV_field_statics.py

对 FakeSV 的 LensLang JSON + 原始标签做字段诊断，主要包含：

1. 低方差 / 取值统计
   1.1 数值字段：计算均值、标准差、最小值、最大值，输出 CSV 和按 std 排序的条形图。
   1.2 类别字段：各取值频数统计，输出 CSV 和频数条形图。

2. 与真假标签（annotation）的关系
   2.1 数值字段：与标签的点双列（Pearson）相关、Spearman 相关，输出 CSV + 按 |相关系数| 排序的条形图。
   2.2 类别字段：对每个字段做 χ² 检验（真/假×类别），输出 CSV。
       同时绘制每个字段的“类别分布 × 真/假”条形图。

3. 多重共线性 / 强重复
   3.1 所有数值字段之间的相关系数矩阵（Pearson），输出 CSV + 热力图。
   3.2 所有 one-hot / multi-hot 特征（按字段展开）之间的 Pearson 相关，
       输出“高相关 pairs”列表 CSV + 热力图（选取部分特征）。

注意：
- 本脚本假定 lenslang JSON 已经是英文字段与取值。
- 标签文件 FakeSV data.json 为 JSON Lines：一行一个 JSON 对象。
"""

import os
import json
import glob
import math
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 尝试使用 SciPy 做统计检验，如不可用则回退到“仅相关系数不做 p 值”
try:
    from scipy.stats import chi2_contingency
    SCIPY_AVAILABLE = True
except Exception:
    chi2_contingency = None  # type: ignore
    SCIPY_AVAILABLE = False

# ========= 路径配置 =========
JSON_DIR = "/data/hyan671/yhproject/FakingRecipe/lenslang/outputs/lenslang_FakeSV_no_keyChts_3624only_Complete"
LABEL_JSON = "/data/hyan671/yhproject/FakingRecipe/dataio/FakeSV/data_complete.json"

OUT_ROOT = "/data/hyan671/yhproject/FakingRecipe/lenslang/outputs/data_analysis/FakeSV_DS/FakeSV_field_statics"

# ========= 字段配置 =========

NUMERIC_FIELDS = [
    "stats.num_shots",
    "stats.avg_shot_len_sec",
    "stats.cuts_per_30s",
    "stats.lighting.s_mean",
    "stats.lighting.v_mean",
    "stats.lighting.v_std",
    "stats.movement_counts.static_shot",
    "stats.movement_counts.pan_horizontal",
    "stats.movement_counts.tilt_vertical",
    "stats.movement_counts.dolly_in",
    "stats.movement_counts.dolly_out",
    "analysis.video_analysis.confidence_score",
]

# 单值类别字段（一个样本一个取值）
CAT_SINGLE_FIELDS = [
    "stats.editing_rhythm_guess",
    "analysis.video_analysis.editing_rhythm",
    "stats.lighting.quality",
    "stats.lighting.condition",
    "stats.lighting.color_temperature",
    "analysis.video_analysis.color_tone",
    "analysis.video_analysis.lighting_style",
]

# 多值类别字段（list 或 “|” 拼接）
CAT_MULTI_FIELDS = [
    "analysis.video_analysis.shot_types",
    "analysis.video_analysis.camera_movements",
    "analysis.video_analysis.composition_styles",
    "analysis.video_analysis.overall_style",
    "analysis.style_tags",
]

# 3.2 中需要展开为 0/1 特征的字段
ONEHOT_FIELDS = [
    "stats.editing_rhythm_guess",
    "analysis.video_analysis.editing_rhythm",
    "stats.lighting.quality",
    "stats.lighting.condition",
    "stats.lighting.color_temperature",
]
MULTIHOT_FIELDS_FOR_BINARY = [
    "analysis.video_analysis.color_tone",
    "analysis.video_analysis.lighting_style",
    "analysis.video_analysis.shot_types",
    "analysis.video_analysis.camera_movements",
    "analysis.video_analysis.composition_styles",
    "analysis.video_analysis.overall_style",
    "analysis.style_tags",
]

# ========= 工具函数 =========

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_get(d, path, default=None):
    """从嵌套 dict 中按 'a.b.c' 取值。"""
    cur = d
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def split_multi(val):
    """把 list 或 'a|b|c' 统一拆成列表。"""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    s = str(val).strip()
    if not s:
        return []
    return [t.strip() for t in s.split("|") if t.strip()]


def to_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan


# ========= 1) 读取 lenslang JSON =========

def load_lenslang_jsons():
    rows = []
    json_files = sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))
    if not json_files:
        raise RuntimeError(f"No JSON found under {JSON_DIR}")
    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[WARN] failed to parse {fp}: {e}")
            continue

        rec = {"video_id": obj.get("video_id")}
        # 数值字段
        for col in NUMERIC_FIELDS:
            rec[col] = to_float(safe_get(obj, col, np.nan))
        # 单值类别字段
        for col in CAT_SINGLE_FIELDS:
            v = safe_get(obj, col, None)
            rec[col] = None if v is None else str(v).strip()
        # 多值字段：先以“|”连接存储，后面再拆
        for col in CAT_MULTI_FIELDS:
            vals = split_multi(safe_get(obj, col, []))
            rec[col] = "|".join(vals)
        rows.append(rec)
    df = pd.DataFrame(rows)
    return df


# ========= 2) 读取标签 JSON Lines =========

def parse_annotation(val):
    """把 annotation 映射为 0/1：
       fake -> 0, real/true -> 1，其余返回 None。
    """
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in {"假", "辟谣", "0"}:
        return 0
    if s in {"真", "1"}:
        return 1
    return None


def load_labels():
    """
    读取 FakeSV/data.json（JSON Lines 格式，一行一个 JSON 对象）。
    每行形如：
    {"video_id":"...","description":"...","annotation":"fake", ...}
    """
    rows = []
    with open(LABEL_JSON, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] failed to parse label JSON line {line_no}: {e}")
                continue
            vid = obj.get("video_id")
            ann_raw = obj.get("annotation")
            ann = parse_annotation(ann_raw)
            if vid is None or ann is None:
                continue
            rows.append(
                {
                    "video_id": vid,
                    "label": ann,
                    "label_str": str(ann_raw).strip(),
                }
            )
    if not rows:
        raise RuntimeError(f"No valid labels parsed from {LABEL_JSON}")
    df = pd.DataFrame(rows)
    # 去重：同一个 video_id 多条的情况，保留第一条
    df = df.drop_duplicates(subset=["video_id"], keep="first")
    return df


# ========= 3) 合并 DF =========

def build_merged_df():
    print("[INFO] Building merged dataframe ...")
    df_ll = load_lenslang_jsons()
    df_lbl = load_labels()
    df = pd.merge(df_ll, df_lbl, on="video_id", how="inner")
    print(f"[INFO] merged rows: {len(df)}")
    return df


# ========= 绘图辅助 =========

def save_fig(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ========= 1.x 数值与类别字段基本统计 =========

def analyze_numeric_basic(df, csv_dir, fig_dir):
    rows = []
    for col in NUMERIC_FIELDS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.dropna()
        if s.empty:
            continue
        mean = float(s.mean())
        std = float(s.std())
        vmin = float(s.min())
        vmax = float(s.max())
        rows.append(
            {
                "field": col,
                "mean": mean,
                "std": std,
                "min": vmin,
                "max": vmax,
                "n": int(len(s)),
            }
        )
    if not rows:
        return
    stat_df = pd.DataFrame(rows)
    stat_df = stat_df.sort_values("std", ascending=True)
    out_csv = os.path.join(csv_dir, "numeric_variance_summary.csv")
    stat_df.to_csv(out_csv, index=False)

    # 按 std 画 bar
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(stat_df))))
    ax.barh(stat_df["field"], stat_df["std"])
    ax.set_xlabel("Std (standard deviation)")
    ax.set_title("Numeric fields std (sorted ascending)")
    save_fig(fig, os.path.join(fig_dir, "numeric_std_bar.png"))


def analyze_categorical_basic(df, csv_dir, fig_dir):
    # 单值字段 + 多值字段
    for col in CAT_SINGLE_FIELDS + CAT_MULTI_FIELDS:
        if col not in df.columns:
            continue
        # 多值字段：拆成单值后统计
        if col in CAT_MULTI_FIELDS:
            vals = []
            for s in df[col].fillna(""):
                vals.extend(split_multi(s))
        else:
            vals = [str(x).strip() for x in df[col].fillna("") if str(x).strip()]
        counter = Counter(vals)
        if not counter:
            continue
        vc_df = (
            pd.DataFrame({"value": list(counter.keys()), "count": list(counter.values())})
            .sort_values("count", ascending=False)
        )
        out_csv = os.path.join(
            csv_dir, f"categorical_counts__{col.replace('.', '_')}.csv"
        )
        vc_df.to_csv(out_csv, index=False)

        # 画条形图（只画前 20 个，避免太长）
        top_df = vc_df.head(20)
        fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(top_df))))
        ax.barh(top_df["value"], top_df["count"])
        ax.set_xlabel("count")
        ax.set_title(f"{col} value counts (top 20)")
        save_fig(
            fig,
            os.path.join(fig_dir, f"categorical_counts__{col.replace('.', '_')}_top20.png"),
        )


# ========= 2.1 数值字段与标签的相关 =========

def analyze_numeric_vs_label(df, csv_dir, fig_dir):
    rows = []
    for col in NUMERIC_FIELDS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        lbl = df["label"]
        mask = s.notna() & lbl.notna()
        if mask.sum() < 10:
            continue
        x = s[mask].astype(float).values
        y = lbl[mask].astype(float).values  # 0 / 1

        # 点双列相关 = Pearson(x, y)
        if np.std(x) <= 1e-8 or np.std(y) <= 1e-8:
            r_pb = np.nan
        else:
            r_pb = float(np.corrcoef(x, y)[0, 1])

        # Spearman
        xr = pd.Series(x).rank(method="average").values
        yr = pd.Series(y).rank(method="average").values
        if np.std(xr) <= 1e-8 or np.std(yr) <= 1e-8:
            r_sp = np.nan
        else:
            r_sp = float(np.corrcoef(xr, yr)[0, 1])

        rows.append(
            {
                "field": col,
                "r_point_biserial": r_pb,
                "r_spearman": r_sp,
                "n": int(mask.sum()),
            }
        )
    if not rows:
        return
    out_df = pd.DataFrame(rows)
    out_df["abs_r_pb"] = out_df["r_point_biserial"].abs()
    out_df = out_df.sort_values("abs_r_pb", ascending=False)
    out_csv = os.path.join(csv_dir, "numeric_vs_label_corr.csv")
    out_df.to_csv(out_csv, index=False)

    # 画条形图（按 |r_pb| 排序）
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(out_df))))
    ax.barh(out_df["field"], out_df["abs_r_pb"])
    ax.set_xlabel("|r (point-biserial)|")
    ax.set_title("Numeric fields vs label (sorted by |r|)")
    save_fig(fig, os.path.join(fig_dir, "numeric_vs_label_corr_bar.png"))


# ========= 2.2 类别字段与标签的 χ² =========

def chi_square_for_field(df, col, is_multi=False):
    """
    返回该字段的 χ² 统计以及每个取值的 contingency table（类别 × label）。
    对多值字段：按“标签是否出现在该视频中”展开。
    """
    label = df["label"]
    # 构建：行 = 类别值，列 = label(0/1)
    counter = defaultdict(lambda: [0, 0])  # value -> [count_label0, count_label1]
    if is_multi:
        for v_str, y in zip(df[col].fillna(""), label):
            ys = int(y)
            for t in split_multi(v_str):
                counter[t][ys] += 1
    else:
        for v, y in zip(df[col].fillna(""), label):
            v = str(v).strip()
            if not v:
                continue
            ys = int(y)
            counter[v][ys] += 1

    if not counter:
        return None, None, None

    values = sorted(counter.keys())
    table = np.array([counter[v] for v in values], dtype=float)  # shape (k,2)

    if SCIPY_AVAILABLE:
        chi2, p, dof, _ = chi2_contingency(table)
    else:
        # 自行计算 χ²，但不算 p 值
        total = table.sum()
        row_sums = table.sum(axis=1, keepdims=True)
        col_sums = table.sum(axis=0, keepdims=True)
        expected = row_sums @ col_sums / total
        with np.errstate(divide="ignore", invalid="ignore"):
            chi2 = np.nansum((table - expected) ** 2 / expected)
        dof = (table.shape[0] - 1) * (table.shape[1] - 1)
        p = np.nan

    return values, table, (chi2, p, dof)


def analyze_categorical_vs_label(df, csv_dir, fig_dir):
    rows = []
    for col in CAT_SINGLE_FIELDS + CAT_MULTI_FIELDS:
        if col not in df.columns:
            continue
        is_multi = col in CAT_MULTI_FIELDS
        values, table, stats = chi_square_for_field(df, col, is_multi=is_multi)
        if values is None:
            continue
        chi2, p, dof = stats
        rows.append(
            {
                "field": col,
                "chi2": chi2,
                "p_value": p,
                "dof": dof,
                "n_categories": len(values),
            }
        )

        # 输出该字段的 contingency table
        ct_df = pd.DataFrame(
            table, columns=["label=0", "label=1"], index=values
        )
        ct_df.to_csv(
            os.path.join(
                csv_dir,
                f"categorical_vs_label_contingency__{col.replace('.', '_')}.csv",
            ),
            index_label="value",
        )

        # 画 label 分布条形图（top20）
        # 先根据总频数排序
        total_counts = table.sum(axis=1)
        order_idx = np.argsort(-total_counts)
        top_k = min(20, len(values))
        sel = order_idx[:top_k]
        v_top = [values[i] for i in sel]
        tbl_top = table[sel, :]

        x = np.arange(top_k)
        fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * top_k)))
        ax.barh(x - 0.15, tbl_top[:, 0], height=0.3, label="label=0 (fake)")
        ax.barh(x + 0.15, tbl_top[:, 1], height=0.3, label="label=1 (real)")
        ax.set_yticks(x)
        ax.set_yticklabels(v_top)
        ax.set_xlabel("count")
        ax.set_title(f"{col} × label (top {top_k} values)")
        ax.legend()
        save_fig(
            fig,
            os.path.join(
                fig_dir,
                f"categorical_vs_label_bar__{col.replace('.', '_')}_top{top_k}.png",
            ),
        )

    if rows:
        out_df = pd.DataFrame(rows)
        out_df = out_df.sort_values("chi2", ascending=False)
        out_df.to_csv(
            os.path.join(csv_dir, "categorical_vs_label_chi2_summary.csv"),
            index=False,
        )


# ========= 3.1 数值字段相关矩阵 =========

def analyze_numeric_correlation(df, csv_dir, fig_dir):
    cols = [c for c in NUMERIC_FIELDS if c in df.columns]
    if not cols:
        return
    num_df = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = num_df.corr(method="pearson")
    corr.to_csv(os.path.join(csv_dir, "numeric_corr_matrix.csv"))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticklabels(cols)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Numeric fields correlation matrix")
    save_fig(fig, os.path.join(fig_dir, "numeric_corr_heatmap.png"))


# ========= 3.2 one-hot / multi-hot 展开后的相关 =========

def build_binary_design(df):
    """
    将指定的 one-hot / multi-hot 字段展开为 0/1 矩阵。
    返回：
      - bin_df: DataFrame，列名为 "<field>==<value>"
    """
    cols = {}
    # one-hot：每样本一个取值
    for field in ONEHOT_FIELDS:
        if field not in df.columns:
            continue
        s = df[field].fillna("").astype(str).str.strip()
        values = sorted(v for v in s.unique() if v)
        for v in values:
            col_name = f"{field}=={v}"
            cols[col_name] = (s == v).astype(int)

    # multi-hot：list / "a|b|c"
    for field in MULTIHOT_FIELDS_FOR_BINARY:
        if field not in df.columns:
            continue
        # 收集所有取值
        all_vals = set()
        for s in df[field].fillna(""):
            for t in split_multi(s):
                all_vals.add(t)
        for v in sorted(all_vals):
            col_name = f"{field}=={v}"
            col = []
            for s in df[field].fillna(""):
                tags = split_multi(s)
                col.append(1 if v in tags else 0)
            cols[col_name] = pd.Series(col, index=df.index, dtype=int)

    if not cols:
        return pd.DataFrame(index=df.index)
    bin_df = pd.DataFrame(cols)
    return bin_df


def analyze_binary_correlation(df, csv_dir, fig_dir, top_k_pairs=100, max_features_for_heatmap=40):
    bin_df = build_binary_design(df)
    if bin_df.empty:
        return

    # 计算 Pearson 相关
    corr = bin_df.corr(method="pearson")
    # 找到高相关 pairs
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iat[i, j]
            if np.isnan(r):
                continue
            pairs.append((cols[i], cols[j], float(r), abs(float(r))))
    if not pairs:
        return
    pairs.sort(key=lambda x: x[3], reverse=True)
    top_pairs = pairs[:top_k_pairs]
    out_df = pd.DataFrame(
        top_pairs, columns=["feature_a", "feature_b", "r", "abs_r"]
    )
    out_df.to_csv(
        os.path.join(csv_dir, "binary_feature_corr_top_pairs.csv"), index=False
    )

    # 为 heatmap 选取若干特征（按方差或出现频率）
    # 这里简单按每列的标准差从大到小取前 max_features_for_heatmap 个
    stds = bin_df.std().sort_values(ascending=False)
    feat_sel = stds.head(max_features_for_heatmap).index.tolist()
    sub_corr = bin_df[feat_sel].corr(method="pearson")

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(feat_sel)), 8))
    im = ax.imshow(sub_corr.values, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(feat_sel)))
    ax.set_yticks(range(len(feat_sel)))
    ax.set_xticklabels(feat_sel, rotation=90, fontsize=6)
    ax.set_yticklabels(feat_sel, fontsize=6)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Binary (one-hot/multi-hot) feature correlation (subset)")
    save_fig(fig, os.path.join(fig_dir, "binary_feature_corr_heatmap_subset.png"))


# ========= 主函数 =========

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(OUT_ROOT, f"FakeSV_field_statics_{ts}")
    csv_dir = os.path.join(out_dir, "csv")
    fig_dir = os.path.join(out_dir, "fig")
    ensure_dir(csv_dir)
    ensure_dir(fig_dir)

    df = build_merged_df()
    # 保存一份合并后的原始明细
    df.to_csv(os.path.join(csv_dir, "merged_dataset.csv"), index=False)

    # 1. 基本统计
    analyze_numeric_basic(df, csv_dir, fig_dir)
    analyze_categorical_basic(df, csv_dir, fig_dir)

    # 2. 与标签关系
    analyze_numeric_vs_label(df, csv_dir, fig_dir)
    analyze_categorical_vs_label(df, csv_dir, fig_dir)

    # 3. 多重共线性
    analyze_numeric_correlation(df, csv_dir, fig_dir)
    analyze_binary_correlation(df, csv_dir, fig_dir)

    print("[DONE] All analyses finished.")
    print(f"  OUT_DIR: {out_dir}")
    print(f"  CSV_DIR: {csv_dir}")
    print(f"  FIG_DIR: {fig_dir}")


if __name__ == "__main__":
    main()