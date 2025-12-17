#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Error slice 全量分析脚本（FakeSV: baseline × baseline+LensLang）

在原有 Table1–8 / Fig1–6 的基础上，新增：

1）对 JSON→CSV 中「所有 LensLang 字段」逐列做切片（slice）：
    - 对类别字段：每个取值一行
    - 对数值字段：先按分位数做分箱（qcut，失败则 cut），每个 bin 一行
2）对每个 slice 同时统计：
    support, base_error_rate, lens_error_rate, delta_error(lens-base)
3）导出：
    - 每个字段一张 CSV：csv/compare_err_by_<field>.csv
    - 若类别数不多（≤ MAX_CATEGORIES_FOR_PLOT），画柱状图：
      fig/compare_err_by_<field>.png
    - 汇总总表：csv/compare_err_all_slices.csv
    - 拆开的两张表：
        csv/error_rate_baseline.csv
        csv/error_rate_lenslang.csv

这样就能逐个摊开、比较 baseline vs baseline+LensLang
在「每一个 JSON 字段×取值/分箱」上的 error_rate 表现，
来看 LensLang 是否弥补了 baseline 的短板。
"""

import os
import datetime
import warnings
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, recall_score, precision_score

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ====================== 路径配置区 ====================== #

LENS_JSON_CSV_PATH = "/data/hyan671/yhproject/FakingRecipe/lenslang/outputs/data_analysis/FakeTT_DS/FakeTT_lenslang_json_dataset.csv"

BASELINE_CSV_PATH = "/data/hyan671/yhproject/FakingRecipe/lenslang/outputs/data_analysis/FakeTT_DS/error_slice_analysis/label_FakeTT_train_20251129_000126_BASELINE.csv"

LENSLANG_CSV_PATH = "/data/hyan671/yhproject/FakingRecipe/lenslang/outputs/data_analysis/FakeTT_DS/error_slice_analysis/label_FakeTT_train_20251201_224209_round3.csv"
# ↑ 这里改成你 baseline+LensLang 的真实路径

OUTPUT_ROOT = "/data/hyan671/yhproject/FakingRecipe/lenslang/outputs/data_analysis/FakeTT_DS/error_slice_analysis"

# 图表里最多画多少个柱（类别太多就不画，只导出 csv）
MAX_CATEGORIES_FOR_PLOT = 10

# ====================== 初始化输出目录 ====================== #

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = os.path.join(OUTPUT_ROOT, f"error_slice_fullv1_{TIMESTAMP}")
CSV_DIR = os.path.join(RUN_DIR, "csv")
FIG_DIR = os.path.join(RUN_DIR, "fig")

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ====================== 工具函数：加载 & 合并 ====================== #

def load_pred_file(path: str, model_name: str) -> pd.DataFrame:
    """
    读取一个预测成绩单：要求至少有列 vid,label,pred,wrong。
    为了区分模型，在列名上加 prefix。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Prediction file not found: {path}")

    df = pd.read_csv(path)
    required_cols = {"vid", "label", "pred", "wrong"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"{path} must have columns: {required_cols}, found {df.columns}")

    df = df.rename(columns={
        "pred": f"pred_{model_name}",
        "wrong": f"wrong_{model_name}"
    })
    return df[["vid", "label", f"pred_{model_name}", f"wrong_{model_name}"]]


def load_and_merge_all() -> Tuple[pd.DataFrame, List[str]]:
    """
    读取：
    - LensLang JSON CSV（video_id → vid）
    - baseline 成绩单
    - baseline+LensLang 成绩单

    返回：
    df_all：
        vid, label,
        pred_base, wrong_base,
        pred_lens, wrong_lens,
        is_error_base, is_error_lens,
        + LensLang 所有字段

    lens_feature_cols：
        JSON→CSV 中的所有特征列名（不含 vid）
    """
    print("[INFO] Loading LensLang CSV:", LENS_JSON_CSV_PATH)
    df_lens = pd.read_csv(LENS_JSON_CSV_PATH)

    if "video_id" in df_lens.columns:
        df_lens = df_lens.rename(columns={"video_id": "vid"})
    elif "vid" not in df_lens.columns:
        raise ValueError("LensLang CSV must have 'video_id' or 'vid' column.")

    # 记录 JSON 的特征列（之后做全量 slice 对比就用它）
    lens_feature_cols = [c for c in df_lens.columns if c != "vid"]

    print("[INFO] Loading baseline prediction:", BASELINE_CSV_PATH)
    df_base = load_pred_file(BASELINE_CSV_PATH, "base")

    print("[INFO] Loading baseline+LensLang prediction:", LENSLANG_CSV_PATH)
    df_lenspred = load_pred_file(LENSLANG_CSV_PATH, "lens")

    # 对齐 label（用 baseline 的 label 为主）
    if not (df_base["label"].values == df_lenspred["label"].values).all():
        # 保险起见：以 vid 为 key merge
        df_lenspred = df_lenspred.drop(columns=["label"])
        df_pred = df_base.merge(df_lenspred, on="vid", how="inner")
    else:
        df_pred = df_base.copy()
        df_pred["pred_lens"] = df_lenspred["pred_lens"]
        df_pred["wrong_lens"] = df_lenspred["wrong_lens"]

    # merge LensLang 特征
    df_all = df_pred.merge(df_lens, on="vid", how="inner")

    # 标准化错误标记
    df_all["is_error_base"] = df_all["wrong_base"].astype(int)
    df_all["is_error_lens"] = df_all["wrong_lens"].astype(int)

    print(f"[INFO] Merged samples: {len(df_all)}")
    print(f"[INFO] Baseline error rate: {df_all['is_error_base'].mean():.4f}")
    print(f"[INFO] LensLang error rate: {df_all['is_error_lens'].mean():.4f}")

    return df_all, lens_feature_cols


# ====================== 工具函数：整体 metrics（Table 1） ====================== #

def compute_overall_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 baseline / baseline+LensLang 的整体指标：
    auc, f1, recall, precision, acc
    """
    y_true = df["label"].values

    rows = []
    for model_name, pred_col in [("baseline", "pred_base"), ("baseline+lenslang", "pred_lens")]:
        y_pred = df[pred_col].values

        try:
            auc = roc_auc_score(y_true, y_pred)
        except Exception:
            auc = np.nan

        f1 = f1_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)

        rows.append({
            "model": model_name,
            "auc": auc,
            "f1": f1,
            "recall": rec,
            "precision": prec,
            "acc": acc
        })

    df_metrics = pd.DataFrame(rows)
    out_path = os.path.join(CSV_DIR, "table1_overall_metrics.csv")
    df_metrics.to_csv(out_path, index=False)
    print(f"[INFO] Saved Table 1 overall metrics to {out_path}")
    return df_metrics


# ====================== Table 2：baseline FP/FN 分解 ====================== #

def compute_confusion_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Table 2：baseline 一侧的 FP/FN 分解 + confusion matrix 风格统计。
    """
    tmp = df.groupby(["label", "pred_base"]).size().reset_index(name="count")

    def cell_type(row):
        if row["label"] == 0 and row["pred_base"] == 0:
            return "TN"
        if row["label"] == 1 and row["pred_base"] == 1:
            return "TP"
        if row["label"] == 0 and row["pred_base"] == 1:
            return "FP"
        if row["label"] == 1 and row["pred_base"] == 0:
            return "FN"
        return "UNK"

    tmp["cell_type"] = tmp.apply(cell_type, axis=1)
    tmp["proportion"] = tmp["count"] / len(df)

    out_path = os.path.join(CSV_DIR, "table2_confusion_baseline.csv")
    tmp.to_csv(out_path, index=False)
    print(f"[INFO] Saved Table 2 confusion baseline to {out_path}")
    return tmp


# ====================== 通用切片统计函数 ====================== #

def slice_error_table(
    df: pd.DataFrame,
    slice_col: str,
    slice_name: Optional[str] = None
) -> pd.DataFrame:
    """
    通用切片表：
    对 slice_col 分组，统计：
      support,
      base_error_rate,
      lens_error_rate,
      delta_error (lens - base)
    """
    if slice_name is None:
        slice_name = slice_col

    tmp = df[[slice_col, "is_error_base", "is_error_lens"]].copy()
    grp = tmp.groupby(slice_col)

    res = grp.agg(
        support=("is_error_base", "size"),
        base_error_rate=("is_error_base", "mean"),
        lens_error_rate=("is_error_lens", "mean")
    ).reset_index()

    res["delta_error"] = res["lens_error_rate"] - res["base_error_rate"]
    res = res.rename(columns={slice_col: slice_name})
    return res


# ====================== Table 3–6：具体 slice 表 ====================== #

def make_slice_tables(df: pd.DataFrame):
    """
    生成：
    - Table 3：按 editing_rhythm（analysis.video_analysis.editing_rhythm）
    - Table 4：按 cuts_per_30s 分箱（低/中/高）
    - Table 5：按 color_tone（analysis.video_analysis.color_tone）
    - Table 6：按 label × editing_rhythm 组合的错误率
    """
    # ---------- Table 3：editing_rhythm slice ---------- #
    if "analysis.video_analysis.editing_rhythm" in df.columns:
        col_edit = "analysis.video_analysis.editing_rhythm"
    else:
        col_edit = "editing_rhythm"

    df3 = slice_error_table(df, col_edit, slice_name="editing_rhythm")
    path3 = os.path.join(CSV_DIR, "table3_slice_editing_rhythm.csv")
    df3.to_csv(path3, index=False)
    print(f"[INFO] Saved Table 3 to {path3}")

    # ---------- Table 4：cuts_per_30s 分箱 ---------- #
    if "stats.cuts_per_30s" not in df.columns:
        print("[WARN] stats.cuts_per_30s not found, skip Table 4.")
    else:
        cuts = df["stats.cuts_per_30s"]
        try:
            df["_cuts_bin"] = pd.qcut(cuts, q=3, labels=["low", "mid", "high"])
        except ValueError:
            df["_cuts_bin"] = pd.cut(cuts, bins=3, labels=["low", "mid", "high"])

        df4 = slice_error_table(df, "_cuts_bin", slice_name="cuts_bin")
        path4 = os.path.join(CSV_DIR, "table4_slice_cuts_bins.csv")
        df4.to_csv(path4, index=False)
        print(f"[INFO] Saved Table 4 to {path4}")

    # ---------- Table 5：color_tone slice ---------- #
    if "analysis.video_analysis.color_tone" in df.columns:
        col_tone = "analysis.video_analysis.color_tone"
    elif "color_tone" in df.columns:
        col_tone = "color_tone"
    else:
        col_tone = None

    if col_tone is not None:
        df5 = slice_error_table(df, col_tone, slice_name="color_tone")
        path5 = os.path.join(CSV_DIR, "table5_slice_color_tone.csv")
        df5.to_csv(path5, index=False)
        print(f"[INFO] Saved Table 5 to {path5}")
    else:
        print("[WARN] color_tone not found, skip Table 5.")

    # ---------- Table 6：label × editing_rhythm ---------- #
    df["_label_edit_bin"] = df["label"].astype(str) + "×" + df[col_edit].astype(str)
    df6 = slice_error_table(df, "_label_edit_bin", slice_name="label×editing_rhythm")
    path6 = os.path.join(CSV_DIR, "table6_slice_label_by_editing.csv")
    df6.to_csv(path6, index=False)
    print(f"[INFO] Saved Table 6 to {path6}")


# ====================== Table 7–8：Top-K 改善/退步 slices ====================== #

def make_topK_improve_regress_tables(df: pd.DataFrame, top_k: int = 10):
    """
    基于几个核心 slice 维度（editing_rhythm, cuts_bin, color_tone, label×editing），
    输出：
    - Table 7：Top-K 改善 slice
    - Table 8：Top-K 退步 slice
    """
    slice_tables = []

    if "analysis.video_analysis.editing_rhythm" in df.columns:
        col_edit = "analysis.video_analysis.editing_rhythm"
    else:
        col_edit = "editing_rhythm"

    # editing_rhythm
    st_edit = slice_error_table(df, col_edit, slice_name="slice_value")
    st_edit["slice_type"] = "editing_rhythm"

    # cuts_bin
    if "_cuts_bin" not in df.columns and "stats.cuts_per_30s" in df.columns:
        cuts = df["stats.cuts_per_30s"]
        try:
            df["_cuts_bin"] = pd.qcut(cuts, q=3, labels=["low", "mid", "high"])
        except ValueError:
            df["_cuts_bin"] = pd.cut(cuts, bins=3, labels=["low", "mid", "high"])

    if "_cuts_bin" in df.columns:
        st_cuts = slice_error_table(df, "_cuts_bin", slice_name="slice_value")
        st_cuts["slice_type"] = "cuts_bin"
        slice_tables.append(st_cuts)

    # color_tone
    if "analysis.video_analysis.color_tone" in df.columns:
        col_tone = "analysis.video_analysis.color_tone"
    elif "color_tone" in df.columns:
        col_tone = "color_tone"
    else:
        col_tone = None

    if col_tone is not None:
        st_tone = slice_error_table(df, col_tone, slice_name="slice_value")
        st_tone["slice_type"] = "color_tone"
        slice_tables.append(st_tone)

    # label×editing
    df["_label_edit_bin"] = df["label"].astype(str) + "×" + df[col_edit].astype(str)
    st_label_edit = slice_error_table(df, "_label_edit_bin", slice_name="slice_value")
    st_label_edit["slice_type"] = "label×editing"
    slice_tables.append(st_edit)
    slice_tables.append(st_label_edit)

    all_slices = pd.concat(slice_tables, ignore_index=True)
    all_slices = all_slices[all_slices["support"] >= 20].copy()

    improved = all_slices.sort_values(by="delta_error").head(top_k)
    regressed = all_slices.sort_values(by="delta_error", ascending=False).head(top_k)

    path7 = os.path.join(CSV_DIR, "table7_topK_improved_slices.csv")
    improved.to_csv(path7, index=False)
    print(f"[INFO] Saved Table 7 (improved slices) to {path7}")

    path8 = os.path.join(CSV_DIR, "table8_topK_regressed_slices.csv")
    regressed.to_csv(path8, index=False)
    print(f"[INFO] Saved Table 8 (regressed slices) to {path8}")

    return improved, regressed


# ====================== Figures：若干核心对比图 ====================== #

def plot_bar_compare(
    df_slice: pd.DataFrame,
    slice_col: str,
    title: str,
    filename: str
):
    """
    简单的柱状对比图：
    X: slice 值
    Y: base_error_rate、lens_error_rate
    """
    labels = df_slice[slice_col].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))

    ax.bar(x - width / 2, df_slice["base_error_rate"], width, label="baseline")
    ax.bar(x + width / 2, df_slice["lens_error_rate"], width, label="baseline+LensLang")

    ax.set_ylabel("error_rate")
    ax.set_xlabel(slice_col)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(FIG_DIR, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved figure to {out_path}")


def make_figures(df: pd.DataFrame, improved: pd.DataFrame, regressed: pd.DataFrame):
    """
    生成：
    - Figure 1：按 editing_rhythm 的错误率前后对比
    - Figure 2：按 cuts_bin 的错误率前后对比
    - Figure 3：按 color_tone 的错误率前后对比
    - Figure 4：Top-K 改善 slices 的 delta_error 条形图
    - Figure 5：Top-K 退步 slices 的 delta_error 条形图
    - Figure 6：一个“low-information slice”的前后比较
    """
    if "analysis.video_analysis.editing_rhythm" in df.columns:
        col_edit = "analysis.video_analysis.editing_rhythm"
    else:
        col_edit = "editing_rhythm"

    # ---- Fig 1: editing_rhythm ---- #
    f1_df = slice_error_table(df, col_edit, slice_name="editing_rhythm")
    plot_bar_compare(
        f1_df,
        slice_col="editing_rhythm",
        title="Error rate by editing_rhythm",
        filename="fig1_err_by_editing_rhythm.png"
    )

    # ---- Fig 2: cuts_bin ---- #
    if "_cuts_bin" in df.columns:
        f2_df = slice_error_table(df, "_cuts_bin", slice_name="cuts_bin")
        plot_bar_compare(
            f2_df,
            slice_col="cuts_bin",
            title="Error rate by cuts_per_30s bins",
            filename="fig2_err_by_cuts_bins.png"
        )

    # ---- Fig 3: color_tone ---- #
    if "analysis.video_analysis.color_tone" in df.columns:
        col_tone = "analysis.video_analysis.color_tone"
    elif "color_tone" in df.columns:
        col_tone = "color_tone"
    else:
        col_tone = None

    if col_tone is not None:
        f3_df = slice_error_table(df, col_tone, slice_name="color_tone")
        plot_bar_compare(
            f3_df,
            slice_col="color_tone",
            title="Error rate by color_tone",
            filename="fig3_err_by_color_tone.png"
        )

    # ---- Fig 4: Top-K improved slices ---- #
    if not improved.empty:
        fig, ax = plt.subplots(figsize=(max(6, len(improved) * 1.4), 4))
        labels = (improved["slice_type"] + ":" + improved["slice_value"].astype(str)).tolist()
        x = np.arange(len(labels))
        ax.bar(x, -improved["delta_error"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Improvement in error_rate (base - lens)")
        ax.set_title("Top-K improved slices")
        fig.tight_layout()
        out_path = os.path.join(FIG_DIR, "fig4_top_improved_slices_bar.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[INFO] Saved figure 4 to {out_path}")

    # ---- Fig 5: Top-K regressed slices ---- #
    if not regressed.empty:
        fig, ax = plt.subplots(figsize=(max(6, len(regressed) * 1.4), 4))
        labels = (regressed["slice_type"] + ":" + regressed["slice_value"].astype(str)).tolist()
        x = np.arange(len(labels))
        ax.bar(x, regressed["delta_error"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Degradation in error_rate (lens - base)")
        ax.set_title("Top-K regressed slices")
        fig.tight_layout()
        out_path = os.path.join(FIG_DIR, "fig5_top_regressed_slices_bar.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[INFO] Saved figure 5 to {out_path}")

    # ---- Fig 6: low-information slice 对比示例 ---- #
    candidate = None
    all_for_low = []

    if col_tone is not None:
        t_df = slice_error_table(df, col_tone, slice_name="slice_value")
        t_df["slice_type"] = "color_tone"
        all_for_low.append(t_df)
    if col_edit is not None:
        e_df = slice_error_table(df, col_edit, slice_name="slice_value")
        e_df["slice_type"] = "editing_rhythm"
        all_for_low.append(e_df)

    if all_for_low:
        merged = pd.concat(all_for_low, ignore_index=True)
        merged = merged[merged["support"] >= 50].copy()
        merged["abs_delta"] = merged["delta_error"].abs()
        merged = merged.sort_values("abs_delta")
        if not merged.empty:
            candidate = merged.iloc[0]

    if candidate is not None:
        fig, ax = plt.subplots(figsize=(4, 4))
        x = np.arange(2)
        base_err = candidate["base_error_rate"]
        lens_err = candidate["lens_error_rate"]
        ax.bar(x, [base_err, lens_err])
        ax.set_xticks(x)
        ax.set_xticklabels(["baseline", "baseline+LensLang"])
        ax.set_ylabel("error_rate")
        ax.set_title(f"Low-info slice ({candidate['slice_type']}={candidate['slice_value']})")
        fig.tight_layout()
        out_path = os.path.join(FIG_DIR, "fig6_low_info_slice_compare.png")
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"[INFO] Saved figure 6 to {out_path}")
    else:
        print("[INFO] No suitable low-information slice found for Figure 6.")


# ====================== 新增部分：JSON 全量字段对比 ====================== #

def sanitize_col_name(col: str) -> str:
    """把列名变成适合做文件名的形式。"""
    return (
        col.replace(".", "_")
           .replace("/", "_")
           .replace(" ", "_")
           .replace(":", "_")
    )


def analyze_all_json_slices(df_all: pd.DataFrame, lens_feature_cols: List[str]):
    """
    对 JSON→CSV 中的每一个字段（lens_feature_cols）：

    - 若为数值字段：做分箱（qcut→cut），每个 bin 当作一个 slice
    - 若为类别字段：每个取值一个 slice
    - 统计：
        support,
        base_error_rate,
        lens_error_rate,
        delta_error(lens - base)

    输出：
        - 每字段一张 CSV：compare_err_by_<field>.csv
        - 若类别数不多（<= MAX_CATEGORIES_FOR_PLOT），画柱状图：
          fig/compare_err_by_<field>.png
        - 总汇总：compare_err_all_slices.csv
        - baseline-only：error_rate_baseline.csv
        - lens-only：error_rate_lenslang.csv
    """
    all_records = []

    for col in lens_feature_cols:
        # 可以按需排除明显不是特征的列
        if col in ["stats.video_name"]:
            continue

        if col not in df_all.columns:
            continue

        series = df_all[col]

        # 全空或常数列直接跳过
        if series.notnull().sum() == 0 or series.nunique(dropna=True) <= 1:
            continue

        df_tmp = df_all.copy()

        # 数值字段：分箱
        if pd.api.types.is_numeric_dtype(series):
            try:
                binned = pd.qcut(series, q=4, duplicates="drop")
            except Exception:
                try:
                    binned = pd.cut(series, bins=4)
                except Exception:
                    print(f"[WARN] Numeric feature {col}: binning failed, skip.")
                    continue

            if binned.isnull().all() or binned.nunique(dropna=True) <= 1:
                print(f"[INFO] Numeric feature {col}: bins single-valued, skip.")
                continue

            slice_col = f"{col}__bin"
            df_tmp[slice_col] = binned
        else:
            slice_col = col

        # 真正做 slice 统计
        try:
            res = slice_error_table(
                df_tmp.dropna(subset=[slice_col]),
                slice_col,
                slice_name="slice_value"
            )
        except Exception as e:
            print(f"[WARN] slice_error_table failed for {col}: {e}")
            continue

        if res.empty:
            continue

        res["feature"] = col
        res = res[["feature", "slice_value", "support",
                   "base_error_rate", "lens_error_rate", "delta_error"]]

        all_records.append(res)

        # 保存 per-feature CSV
        safe_name = sanitize_col_name(col)
        out_path = os.path.join(CSV_DIR, f"compare_err_by_{safe_name}.csv")
        res.to_csv(out_path, index=False)
        print(f"[INFO] Saved slice compare CSV for {col} to {out_path}")

        # 如果类别数不多，画一个 baseline vs lens 的柱状图
        if 1 < res.shape[0] <= MAX_CATEGORIES_FOR_PLOT:
            labels = res["slice_value"].astype(str).tolist()
            x = np.arange(len(labels))
            width = 0.35

            fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
            ax.bar(x - width / 2, res["base_error_rate"], width, label="baseline")
            ax.bar(x + width / 2, res["lens_error_rate"], width, label="baseline+LensLang")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("error_rate")
            ax.set_xlabel("slice_value")
            ax.set_title(f"compare_err_by_{col}")
            ax.legend()
            fig.tight_layout()

            fig_path = os.path.join(FIG_DIR, f"compare_err_by_{safe_name}.png")
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)
            print(f"[INFO] Saved slice compare FIG for {col} to {fig_path}")

    if not all_records:
        print("[INFO] No valid JSON features for full-slice comparison.")
        return

    all_slices = pd.concat(all_records, ignore_index=True)

    # 总汇总
    all_path = os.path.join(CSV_DIR, "compare_err_all_slices.csv")
    all_slices.to_csv(all_path, index=False)
    print(f"[INFO] Saved all-slice compare table to {all_path}")

    # baseline-only & lens-only 两张表
    base_only = all_slices[["feature", "slice_value", "support", "base_error_rate"]]
    lens_only = all_slices[["feature", "slice_value", "support", "lens_error_rate"]]

    base_path = os.path.join(CSV_DIR, "error_rate_baseline.csv")
    lens_path = os.path.join(CSV_DIR, "error_rate_lenslang.csv")
    base_only.to_csv(base_path, index=False)
    lens_only.to_csv(lens_path, index=False)
    print(f"[INFO] Saved baseline-only slice table to {base_path}")
    print(f"[INFO] Saved lens-only slice table to {lens_path}")


# ====================== main ====================== #

def main():
    print(f"[INFO] Output directory: {RUN_DIR}")
    df_all, lens_feature_cols = load_and_merge_all()

    # Table 1：整体指标
    compute_overall_metrics(df_all)

    # Table 2：baseline confusion（FP/FN）
    compute_confusion_baseline(df_all)

    # Table 3–6：几个核心结构化 slice 表
    make_slice_tables(df_all)

    # Table 7–8：Top-K 改善 / 退步 slice 排行
    improved, regressed = make_topK_improve_regress_tables(df_all, top_k=10)

    # Figures 1–6：图像对比
    make_figures(df_all, improved, regressed)

    # 新增：对 JSON 全字段做 baseline vs LensLang 的 slice 对比
    analyze_all_json_slices(df_all, lens_feature_cols)

    print(f"[INFO] All CSVs and figures are under: {RUN_DIR}")


if __name__ == "__main__":
    main()