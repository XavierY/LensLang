# -*- coding: utf-8 -*-
"""
plot_lenslang_from_json.py

读取 LensLang 全量 JSON -> 解析 -> 直接计算统计 + 出图（A–E,F），
无需依赖任何已导出的 CSV。

- JSON 来源:
    /data/hyan671/yhproject/FakingRecipe/lenslang/outputs/lenslang_FakeSV_no_keyChts_3624only_ENG

- 输出根目录（脚本每跑一次新建一个时间戳子目录）:
    /data/hyan671/yhproject/FakingRecipe/lenslang/outputs/data_analysis/FakeSV_DS/FakeSV_analysis_output/<TIMESTAMP>/

  其中：
    <TIMESTAMP>/csv   存所有 csv
    <TIMESTAMP>/fig   存所有 png 图

字段名称严格按照英文版 JSON，例如：
- stats.movement_counts.static_shot
- stats.movement_counts.pan_horizontal
- stats.movement_counts.tilt_vertical
- stats.movement_counts.dolly_in
- stats.movement_counts.dolly_out

说明：
- style_tags 只用于 E 图（多层共现视图），
  不进入 counts / top_outliers 等统计，避免重复夸大效应。
"""

import os, json, glob, math, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# ====== 1) 路径设置（按需修改） ======
JSON_DIR = "/data/hyan671/yhproject/FakingRecipe/lenslang/outputs/lenslang_FakeSV_no_keyChts_3624only_Complete"
OUT_ROOT = "/data/hyan671/yhproject/FakingRecipe/lenslang/outputs/data_analysis/FakeSV_DS/FakeSV_analysis_output"

# 自动生成时间戳子目录
RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(OUT_ROOT, RUN_ID)
CSV_DIR = os.path.join(OUT_DIR, "csv")
FIG_DIR = os.path.join(OUT_DIR, "fig")

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ====== 1.1 style_tags 家族定义（来自你给的脚本） ======
FAMILIES = {
    "Editing": [
        "very_fast", "fast", "moderate", "slow"
    ],
    "Lighting / Color": [
        "bright_sunny", "overcast_neutral", "dim_low_light",
        "cool_tone", "neutral_tone", "warm_tone",
        "high_contrast", "medium_contrast", "soft_low_contrast",
        "backlight", "side_light", "front_light", "natural_light", "artificial_light"
    ],
    "Cinematography": [
        "long_shot", "medium_shot", "close_shot", "close_up", "extreme_close_up",
        "static_shot", "pan_horizontal", "tilt_vertical", "dolly_in", "dolly_out"
    ],
    "Composition / Mise-en-scène": [
        "symmetrical_composition", "rule_of_thirds", "centered_composition", "negative_space",
        "low_angle_perspective", "high_angle_perspective", "deep_focus_perspective", "shallow_depth_of_field"
    ],
    "Narrative / Mood / Overall style": [
        "documentary_style", "narrative_style", "news_style", "vlog_style", "commercial_advertising",
        "cinematic_style", "tense_mood", "warm_mood", "solemn_mood", "energetic_mood"
    ]
}

TAG2FAMILY = {t: fam for fam, tags in FAMILIES.items() for t in tags}


# ====== 2) 工具函数 ======
def safe_get(d, path, default=None):
    """按 'a.b.c' 路径取值；不存在返回 default"""
    cur = d
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def parse_one_json(fp):
    """解析单个 JSON 为一条记录（扁平列 + 多标签字符串）"""
    with open(fp, "r", encoding="utf-8") as f:
        obj = json.load(f)

    rec = {}
    rec["video_id"] = safe_get(obj, "video_id")
    rec["stats.video_name"] = safe_get(obj, "stats.video_name")

    # 数值（节奏/亮度/运动计数）
    rec["stats.num_shots"] = safe_get(obj, "stats.num_shots")
    rec["stats.avg_shot_len_sec"] = safe_get(obj, "stats.avg_shot_len_sec")
    rec["stats.cuts_per_30s"] = safe_get(obj, "stats.cuts_per_30s")
    rec["stats.lighting.s_mean"] = safe_get(obj, "stats.lighting.s_mean")
    rec["stats.lighting.v_mean"] = safe_get(obj, "stats.lighting.v_mean")
    rec["stats.lighting.v_std"]  = safe_get(obj, "stats.lighting.v_std")

    # movement_counts（英文键名）
    mv = safe_get(obj, "stats.movement_counts", {}) or {}
    for k in ["static_shot", "pan_horizontal", "tilt_vertical", "dolly_in", "dolly_out"]:
        rec[f"stats.movement_counts.{k}"] = mv.get(k, np.nan)

    # 单选类别
    rec["stats.editing_rhythm_guess"] = safe_get(obj, "stats.editing_rhythm_guess")
    rec["stats.lighting.quality"] = safe_get(obj, "stats.lighting.quality")
    rec["stats.lighting.condition"] = safe_get(obj, "stats.lighting.condition")
    rec["stats.lighting.color_temperature"] = safe_get(obj, "stats.lighting.color_temperature")

    rec["analysis.video_analysis.editing_rhythm"] = safe_get(obj, "analysis.video_analysis.editing_rhythm")
    rec["analysis.video_analysis.color_tone"] = safe_get(obj, "analysis.video_analysis.color_tone")
    rec["analysis.video_analysis.lighting_style"] = safe_get(obj, "analysis.video_analysis.lighting_style")
    rec["analysis.video_analysis.confidence_score"] = safe_get(obj, "analysis.video_analysis.confidence_score")

    # 多标签：转为竖线拼接字符串（原始家族字段）
    def join_multi(x):
        if not isinstance(x, list):
            return ""
        return "|".join([str(t).strip() for t in x if str(t).strip() != ""])

    rec["analysis.video_analysis.shot_types"]         = join_multi(safe_get(obj, "analysis.video_analysis.shot_types", []))
    rec["analysis.video_analysis.camera_movements"]   = join_multi(safe_get(obj, "analysis.video_analysis.camera_movements", []))
    rec["analysis.video_analysis.composition_styles"] = join_multi(safe_get(obj, "analysis.video_analysis.composition_styles", []))
    rec["analysis.video_analysis.overall_style"]      = join_multi(safe_get(obj, "analysis.video_analysis.overall_style", []))

    # 汇总 style_tags：仅用于 E 图（多层视图）
    rec["analysis.style_tags"] = join_multi(safe_get(obj, "analysis.style_tags", []))

    return rec


def to_float(s):
    try:
        return float(s)
    except Exception:
        return np.nan


def split_multi_cell(s):
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [t.strip() for t in str(s).split("|") if t.strip()]


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=200)
    plt.close()


# ====== 2.1 style_tags 共现度量工具（来自你给的 E 脚本，略作改造） ======
def build_style_records_from_df():
    """
    从 df['analysis.style_tags'] 构造 records:
      records = List[List[tag]]，tag 已小写 + 空格转下划线 + 过滤到 FAMILIES 内。
    """
    records = []
    if "analysis.style_tags" not in df.columns:
        return records
    for s in df["analysis.style_tags"]:
        tags = split_multi_cell(s)
        norm = [t.strip().lower().replace(" ", "_") for t in tags]
        norm = [t for t in norm if t in TAG2FAMILY]
        records.append(norm)
    return records


def index_tags(records):
    counter = Counter()
    for tags in records:
        counter.update(set(tags))  # 按“是否出现”计数（视频层）
    tags_sorted = [t for t, _ in counter.most_common()]
    return tags_sorted, counter


def make_global_order(tags_sorted):
    fam_to_tags = defaultdict(list)
    for t in tags_sorted:
        fam_to_tags[TAG2FAMILY.get(t, "Other")].append(t)
    ordered, family_slices, start = [], {}, 0
    for fam in FAMILIES.keys():
        sub = fam_to_tags.get(fam, [])
        ordered.extend(sub)
        family_slices[fam] = (start, start + len(sub))
        start += len(sub)
    return ordered, family_slices


def compute_pair_stats(records, tags):
    idx = {t: i for i, t in enumerate(tags)}
    n = len(tags)
    pres = np.zeros(n, dtype=np.int64)
    co   = np.zeros((n, n), dtype=np.int64)
    for tag_list in records:
        unique = sorted(set([t for t in tag_list if t in idx]), key=lambda x: idx[x])
        for t in unique:
            pres[idx[t]] += 1
        for i in range(len(unique)):
            for j in range(len(unique)):
                if i <= j:
                    co[idx[unique[i]], idx[unique[j]]] += 1
    co = co + np.triu(co, 1).T
    return pres, co


def to_metric(co, pres, metric):
    n = co.shape[0]
    out = np.zeros_like(co, dtype=float)
    if metric == "count":
        return co.astype(float)
    elif metric == "jaccard":
        for i in range(n):
            for j in range(n):
                denom = pres[i] + pres[j] - co[i, j]
                out[i, j] = (co[i, j] / denom) if denom > 0 else 0.0
    elif metric == "pmi":
        # 以 max(pres, co) 作为 N 的近似，不影响排名与对比
        N = max(int(pres.max()), int(co.max()), 1)
        for i in range(n):
            for j in range(n):
                pij = co[i, j] / N
                pi  = pres[i] / N
                pj  = pres[j] / N
                out[i, j] = math.log(pij / (pi * pj)) if (pij > 0 and pi > 0 and pj > 0) else 0.0
    else:
        raise ValueError("Unknown metric")
    return out


def draw_heatmap(mat, labels_x, labels_y, title, out_path, family_grid=None, annotate_threshold=None):
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    im = ax.imshow(mat, aspect="auto")  # 使用默认颜色映射
    plt.title(title)
    plt.xticks(range(len(labels_x)), labels_x, rotation=60, ha="right", fontsize=9)
    plt.yticks(range(len(labels_y)), labels_y, fontsize=9)
    plt.colorbar(im)

    if family_grid is not None:
        for cut in family_grid:
            ax.axhline(cut - 0.5)
            ax.axvline(cut - 0.5)

    if annotate_threshold is not None:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j] >= annotate_threshold:
                    ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ====== 3) 批量读取 JSON -> DataFrame ======
all_files = sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))
rows, bad = [], []
for fp in all_files:
    try:
        rows.append(parse_one_json(fp))
    except Exception as e:
        bad.append((fp, str(e)))

df = pd.DataFrame(rows)

if bad:
    pd.DataFrame(bad, columns=["file", "error"]).to_csv(
        os.path.join(CSV_DIR, "bad_json_files.csv"), index=False
    )

# 数值列
num_cols = [
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
for c in num_cols:
    if c in df.columns:
        df[c] = df[c].apply(to_float)

# ====== 4) 生成 CSV 统计（不含 style_tags） ======

# 4.1 全量明细
df.to_csv(os.path.join(CSV_DIR, "dataset.csv"), index=False)

# 4.2 每列缺失率
# 4.2 每列缺失率 + 导出空值明细
def is_missing_value(v):
    """
    按你的口径判断“缺失”：
    - NaN / None
    - 字符串：去掉空白后为空
    - list/tuple/set/ndarray：长度为 0
    - dict：长度为 0
    其余情况一律视为“有值”。
    """
    # 1) NaN / None
    if pd.isna(v):
        return True

    # 2) 字符串类型
    if isinstance(v, str):
        return v.strip() == ""

    # 3) 各种“容器”类型
    if isinstance(v, (list, tuple, set, np.ndarray)):
        return len(v) == 0

    if isinstance(v, dict):
        return len(v) == 0

    # 其它类型（数字、bool 等）默认认为是“有值”
    return False


missing_rows = []      # 按字段统计
missing_cells = []     # 逐样本记录：哪条视频哪个字段是空

total = len(df)

for col in df.columns:
    s = df[col]
    miss_mask = s.apply(is_missing_value)
    miss = int(miss_mask.sum())

    # 字段级别统计
    missing_rows.append({
        "field": col,
        "total": total,
        "missing": miss,
        "missing_rate": float(miss) / total if total > 0 else np.nan,
    })

    # 逐样本明细：导出“文件名 + 字段”
    if miss > 0:
        idxs = df.index[miss_mask]
        for idx in idxs:
            missing_cells.append({
                "video_id": df.at[idx, "video_id"] if "video_id" in df.columns else None,
                "stats.video_name": df.at[idx, "stats.video_name"] if "stats.video_name" in df.columns else None,
                "field": col,
                "value": df.at[idx, col],  # 原始值，方便排查
            })

# 按字段汇总的缺失率
pd.DataFrame(missing_rows).to_csv(
    os.path.join(CSV_DIR, "missing_by_field.csv"), index=False
)

# 按“视频 + 字段”的缺失明细（只有存在缺失时才写）
if missing_cells:
    pd.DataFrame(missing_cells).to_csv(
        os.path.join(CSV_DIR, "missing_cells_detail.csv"), index=False
    )

# 4.3 数值汇总 numeric_summary
num_df = df[[c for c in num_cols if c in df.columns]].apply(
    pd.to_numeric, errors="coerce"
)
desc = num_df.describe().T
desc.insert(0, "field", desc.index)
desc.to_csv(os.path.join(CSV_DIR, "numeric_summary.csv"), index=False)

# 4.4 数值相关矩阵
corr = num_df.corr(method="pearson")
corr.to_csv(os.path.join(CSV_DIR, "numeric_correlation_pearson.csv"))

# 4.5 单标签类别统计 categorical_counts__
def write_categorical_counts(col, fname):
    if col not in df.columns:
        return
    s = df[col].astype(str).str.strip()
    s = s[(~s.isna()) & (s != "")]
    vc = s.value_counts().reset_index()
    vc.columns = [col, "count"]
    vc.to_csv(os.path.join(CSV_DIR, fname), index=False)

write_categorical_counts(
    "analysis.video_analysis.color_tone",
    "categorical_counts__analysis_video_analysis_color_tone.csv",
)
write_categorical_counts(
    "analysis.video_analysis.editing_rhythm",
    "categorical_counts__analysis_video_analysis_editing_rhythm.csv",
)
write_categorical_counts(
    "analysis.video_analysis.lighting_style",
    "categorical_counts__analysis_video_analysis_lighting_style.csv",
)
write_categorical_counts(
    "stats.editing_rhythm_guess",
    "categorical_counts__stats_editing_rhythm_guess.csv",
)
write_categorical_counts(
    "stats.lighting.color_temperature",
    "categorical_counts__stats_lighting_color_temperature.csv",
)
write_categorical_counts(
    "stats.lighting.condition",
    "categorical_counts__stats_lighting_condition.csv",
)
write_categorical_counts(
    "stats.lighting.quality",
    "categorical_counts__stats_lighting_quality.csv",
)

# 4.6 多标签统计 multilabel_counts__（不含 style_tags）
def write_multilabel_counts(col, fname):
    if col not in df.columns:
        return
    lists = df[col].apply(split_multi_cell).tolist()
    counter = Counter()
    for ts in lists:
        for t in ts:
            counter[t] += 1
    rows_ml = [{"label": k, "count": v} for k, v in counter.most_common()]
    pd.DataFrame(rows_ml).to_csv(os.path.join(CSV_DIR, fname), index=False)

write_multilabel_counts(
    "analysis.video_analysis.camera_movements",
    "multilabel_counts__analysis_video_analysis_camera_movements.csv",
)
write_multilabel_counts(
    "analysis.video_analysis.composition_styles",
    "multilabel_counts__analysis_video_analysis_composition_styles.csv",
)
write_multilabel_counts(
    "analysis.video_analysis.overall_style",
    "multilabel_counts__analysis_video_analysis_overall_style.csv",
)
write_multilabel_counts(
    "analysis.video_analysis.shot_types",
    "multilabel_counts__analysis_video_analysis_shot_types.csv",
)

# 4.7 数值列 top outliers__
def write_top_outliers(col, fname, topn=50):
    if col not in df.columns:
        return
    s = pd.to_numeric(df[col], errors="coerce")
    mean = s.mean()
    std = s.std()
    if std == 0 or np.isnan(std):
        return
    z = (s - mean) / std
    out = pd.DataFrame({
        "video_id": df.get("video_id"),
        "video_name": df.get("stats.video_name"),
        col: s,
        "z_score": z,
    })
    out = out.dropna(subset=[col, "z_score"])
    out = out.reindex(out["z_score"].abs().sort_values(ascending=False).index)
    out.head(topn).to_csv(os.path.join(CSV_DIR, fname), index=False)

write_top_outliers(
    "analysis.video_analysis.confidence_score",
    "top_outliers__analysis_video_analysis_confidence_score.csv",
)
write_top_outliers(
    "stats.avg_shot_len_sec",
    "top_outliers__stats_avg_shot_len_sec.csv",
)
write_top_outliers(
    "stats.cuts_per_30s",
    "top_outliers__stats_cuts_per_30s.csv",
)
write_top_outliers(
    "stats.lighting.s_mean",
    "top_outliers__stats_lighting_s_mean.csv",
)
write_top_outliers(
    "stats.lighting.v_mean",
    "top_outliers__stats_lighting_v_mean.csv",
)
write_top_outliers(
    "stats.lighting.v_std",
    "top_outliers__stats_lighting_v_std.csv",
)
write_top_outliers(
    "stats.movement_counts.static_shot",
    "top_outliers__stats_movement_counts_static_shot.csv",
)
write_top_outliers(
    "stats.movement_counts.pan_horizontal",
    "top_outliers__stats_movement_counts_pan_horizontal.csv",
)
write_top_outliers(
    "stats.movement_counts.tilt_vertical",
    "top_outliers__stats_movement_counts_tilt_vertical.csv",
)
write_top_outliers(
    "stats.movement_counts.dolly_in",
    "top_outliers__stats_movement_counts_dolly_in.csv",
)
write_top_outliers(
    "stats.movement_counts.dolly_out",
    "top_outliers__stats_movement_counts_dolly_out.csv",
)
write_top_outliers(
    "stats.num_shots",
    "top_outliers__stats_num_shots.csv",
)

# ====== 5) 画图（A–E,F），图片都进 fig/ ======

# A) 节奏一致性：cuts_per_30s vs avg_shot_len_sec
def plot_A_scatter_pacing():
    x = df["stats.cuts_per_30s"].astype(float)
    y = df["stats.avg_shot_len_sec"].astype(float)
    msk = x.notna() & y.notna()
    x, y = x[msk], y[msk]
    if len(x) < 5:
        return
    plt.figure()
    plt.scatter(x, y, s=6, alpha=0.35)
    try:
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 200)
        plt.plot(xx, m * xx + b)
        r = np.corrcoef(x, y)[0, 1]
        plt.title(f"cuts_per_30s vs avg_shot_len_sec (Pearson r={r:.2f})")
    except Exception:
        plt.title("cuts_per_30s vs avg_shot_len_sec")
    plt.xlabel("cuts_per_30s")
    plt.ylabel("avg_shot_len_sec")
    savefig("A_scatter_pacing.png")


# B) editing_rhythm_guess 分组箱线图
def boxplot_by_cat(cat_col, num_col, fname, order=None):
    if cat_col not in df.columns or num_col not in df.columns:
        return
    d = df[[cat_col, num_col]].dropna()
    if len(d) == 0:
        return
    if order is None:
        order = ["very_slow", "slow", "moderate", "fast", "very_fast"]
        order = [c for c in order if c in set(d[cat_col])]
        if not order:
            order = sorted(d[cat_col].unique().tolist())
    data = [d[d[cat_col] == c][num_col].astype(float).values for c in order]
    if not any(len(x) > 0 for x in data):
        return
    plt.figure()
    plt.boxplot(data, labels=order, showfliers=False)
    plt.xlabel(cat_col)
    plt.ylabel(num_col)
    plt.title(f"{num_col} by {cat_col}")
    savefig(fname)

def plot_B_boxes():
    boxplot_by_cat(
        "stats.editing_rhythm_guess",
        "stats.avg_shot_len_sec",
        "B_box_avglen_by_rhythm.png",
    )
    boxplot_by_cat(
        "stats.editing_rhythm_guess",
        "stats.cuts_per_30s",
        "B_box_cuts_by_rhythm.png",
    )


# C) lighting_style × v_std（箱线图）
def plot_C_vstd_by_lighting_style():
    col_cat = "analysis.video_analysis.lighting_style"
    col_val = "stats.lighting.v_std"
    if col_cat not in df.columns or col_val not in df.columns:
        return
    d = df[[col_cat, col_val]].dropna()
    if len(d) == 0:
        return
    groups = d.groupby(col_cat)[col_val].apply(list)
    if len(groups) == 0:
        return
    plt.figure()
    plt.boxplot(groups.values, labels=groups.index, showfliers=False)
    plt.ylabel("v_std")
    plt.xlabel("lighting_style")
    plt.title("v_std by lighting_style")
    savefig("C_box_vstd_by_lighting_style.png")


# D) color_tone × color_temperature（热力图 + Cramér's V）
def cramers_v_from_crosstab(crosstab):
    total = crosstab.values.sum()
    if total == 0:
        return np.nan
    expected = np.outer(
        crosstab.sum(axis=1).values, crosstab.sum(axis=0).values
    ) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = ((crosstab.values - expected) ** 2 / expected).sum()
    r, k = crosstab.shape
    return np.sqrt(chi2 / (total * (min(r, k) - 1))) if min(r, k) > 1 else np.nan

def plot_D_heatmap_color():
    a = "analysis.video_analysis.color_tone"
    b = "stats.lighting.color_temperature"
    if a not in df.columns or b not in df.columns:
        return
    tab = pd.crosstab(df[a], df[b])
    if tab.size == 0:
        return
    plt.figure(figsize=(6, 5))
    plt.imshow(tab.values, aspect="auto")
    plt.xticks(range(tab.shape[1]), tab.columns, rotation=45)
    plt.yticks(range(tab.shape[0]), tab.index)
    plt.colorbar()
    plt.title("Color tone × Color temperature (counts)")
    savefig("D_heatmap_color_tone_vs_temp.png")
    with open(os.path.join(CSV_DIR, "D_cramersV.txt"), "w", encoding="utf-8") as f:
        f.write(f"Cramér's V = {cramers_v_from_crosstab(tab):.3f}\n")


# E) style_tags 三层视图（集成你的脚本逻辑）
# ====== E) Family-level and global co-occurrence views (no style_tags) ======

# 5 个 Family 与对应字段
FAMILY_FIELDS = {
    "Editing": [
        ("analysis.video_analysis.editing_rhythm", "single"),
    ],
    "Lighting / Color": [
        ("stats.lighting.condition", "single"),
        ("analysis.video_analysis.color_tone", "single"),
        ("analysis.video_analysis.lighting_style", "single"),  # 实际也是单选
    ],
    "Cinematography": [
        ("analysis.video_analysis.shot_types", "multi"),
        ("analysis.video_analysis.camera_movements", "multi"),
    ],
    "Composition / Mise-en-scène": [
        ("analysis.video_analysis.composition_styles", "multi"),
    ],
    "Narrative / Mood / Overall style": [
        ("analysis.video_analysis.overall_style", "multi"),
    ],
}

FAMILY_LIST = list(FAMILY_FIELDS.keys())


def get_family_tags_per_video(df):
    """
    对每个视频，按 Family 汇总标签：
    返回 list，长度 = n_videos，每个元素是 {family_name: set(tags)}。
    tags 来自上面的字段值（multi 列拆分，single 列当成单标签）。
    """
    records = []
    for _, row in df.iterrows():
        fam_tags = {fam: set() for fam in FAMILY_FIELDS.keys()}
        for fam, cols in FAMILY_FIELDS.items():
            for col, kind in cols:
                if col not in df.columns:
                    continue
                val = row[col]
                if pd.isna(val) or str(val).strip() == "":
                    continue
                if kind == "multi":
                    tags = split_multi_cell(val)
                else:  # "single"
                    tags = [str(val).strip()]
                for t in tags:
                    if t != "":
                        fam_tags[fam].add(t)
        records.append(fam_tags)
    return records


def compute_pair_stats(tag_lists, tags):
    """
    tag_lists: List[List[str]]，每个元素是一个视频里的标签集合（已去重）
    tags:     全局标签列表（固定顺序）

    返回:
      pres[i]  : 标签 i 在多少视频中出现
      co[i,j]  : 标签 i 与标签 j 在多少视频中共现（对称矩阵）
    """
    idx = {t: i for i, t in enumerate(tags)}
    n = len(tags)
    pres = np.zeros(n, dtype=np.int64)
    co = np.zeros((n, n), dtype=np.int64)

    for tag_list in tag_lists:
        unique = sorted(set([t for t in tag_list if t in idx]), key=lambda x: idx[x])
        if not unique:
            continue
        ids = [idx[t] for t in unique]
        for i in ids:
            pres[i] += 1
        for i in range(len(ids)):
            for j in range(len(ids)):
                if i <= j:
                    co[ids[i], ids[j]] += 1
    co = co + np.triu(co, 1).T
    return pres, co


def to_metric(co, pres, metric="pmi"):
    """
    将共现矩阵 co 转换为指定度量:
      - "count": 原始共现次数
      - "jaccard": Jaccard 相似度
      - "pmi": 点互信息（未归一）
    """
    n = co.shape[0]
    out = np.zeros_like(co, dtype=float)
    if metric == "count":
        return co.astype(float)
    elif metric == "jaccard":
        for i in range(n):
            for j in range(n):
                denom = pres[i] + pres[j] - co[i, j]
                out[i, j] = (co[i, j] / denom) if denom > 0 else 0.0
    elif metric == "pmi":
        N = max(int(pres.max()), int(co.max()), 1)
        for i in range(n):
            for j in range(n):
                pij = co[i, j] / N
                pi = pres[i] / N
                pj = pres[j] / N
                out[i, j] = math.log(pij / (pi * pj)) if (pij > 0 and pi > 0 and pj > 0) else 0.0
    else:
        raise ValueError("Unknown metric")
    return out


def draw_heatmap(mat, labels_x, labels_y, title, filename, family_grid=None, annotate_threshold=None):
    """
    通用热力图绘制，输出到 FIG_DIR/filename
    """
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    im = ax.imshow(mat, aspect="auto")  # 默认 colormap
    plt.title(title)
    plt.xticks(range(len(labels_x)), labels_x, rotation=60, ha="right", fontsize=9)
    plt.yticks(range(len(labels_y)), labels_y, fontsize=9)
    plt.colorbar(im)

    # 可选：画 family 的网格线
    if family_grid is not None:
        for cut in family_grid:
            ax.axhline(cut - 0.5)
            ax.axvline(cut - 0.5)

    # 可选：在高值处标数字
    if annotate_threshold is not None:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j] >= annotate_threshold:
                    ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=200)
    plt.close()


def plot_E_family_and_global(metric="pmi", top_k_global=40):
    """
    E 图：三层视图（不再使用 style_tags）：
      E1: 5x5 Family ↔ Family 共现图
      E2: 各家族内部 Tag ↔ Tag 共现图
      E3: 全局 Top-K 标签共现图
    """
    # 先按 Family 抽出每个视频的标签
    fam_records = get_family_tags_per_video(df)  # List[dict(fam -> set(tags))]

    # ===== E1: Family ↔ Family 概览（5x5） =====
    n_fam = len(FAMILY_LIST)
    fam_idx = {fam: i for i, fam in enumerate(FAMILY_LIST)}
    mat_fam = np.zeros((n_fam, n_fam), dtype=int)

    for ft in fam_records:
        present_fams = [fam for fam, tags in ft.items() if len(tags) > 0]
        if not present_fams:
            continue
        # 每个视频内：所有出现过的 family 两两共现（含对角线）
        for i in range(len(present_fams)):
            for j in range(len(present_fams)):
                fi = fam_idx[present_fams[i]]
                fj = fam_idx[present_fams[j]]
                mat_fam[fi, fj] += 1

    draw_heatmap(
        mat_fam,
        FAMILY_LIST,
        FAMILY_LIST,
        "E1: Family-level co-occurrence (video-level counts)",
        "E1_family_overview_counts.png",
        family_grid=None,
        annotate_threshold=None,
    )

    # ===== E2: 家族内 Tag ↔ Tag 共现（4 个主要家族） =====
    # 只对这些家族画内部共现（Editing 单字段就不画家族内共现）
    for fam in FAMILY_LIST:
        if fam == "Editing":
            continue  # Editing 已在 B 图里和节奏/数值打通，这里就略过

        # 收集此 family 在所有视频中的标签列表
        tag_lists = []
        for ft in fam_records:
            tags = sorted(ft[fam])
            tag_lists.append(tags)

        # 汇总所有出现过的标签
        all_tags = sorted({t for tl in tag_lists for t in tl})
        if len(all_tags) < 2:
            continue

        pres, co = compute_pair_stats(tag_lists, all_tags)
        mat = to_metric(co, pres, metric=metric)

        fname = f"E2_within_{fam.replace(' ', '_').replace('/', '_')}_{metric}.png"
        draw_heatmap(
            mat,
            all_tags,
            all_tags,
            f"E2: Within-family co-occurrence ({fam}, metric={metric})",
            fname,
        )

    # ===== E3: 全域视图：所有标签一起看（Global） =====
    # 对每个视频，把 5 个 family 的标签全部 union 起来
    global_records = []
    for ft in fam_records:
        tags = sorted({t for tags in ft.values() for t in tags})
        global_records.append(tags)

    # 汇总频次并取 top_k_global
    counter = Counter()
    for tl in global_records:
        counter.update(set(tl))
    all_tags_sorted = [t for t, _ in counter.most_common()]
    if top_k_global and top_k_global > 0:
        all_tags_sorted = all_tags_sorted[:top_k_global]

    if len(all_tags_sorted) >= 2:
        pres_g, co_g = compute_pair_stats(global_records, all_tags_sorted)
        mat_g = to_metric(co_g, pres_g, metric=metric)
        draw_heatmap(
            mat_g,
            all_tags_sorted,
            all_tags_sorted,
            f"E3: Global tag co-occurrence (Top-{top_k_global}, metric={metric})",
            f"E3_global_top{top_k_global}_{metric}.png",
        )


# ===== 调用 E 图（替换原来的 E 调用） =====
plot_E_family_and_global(metric="pmi", top_k_global=40)


# F) 数值相关矩阵可视化
def plot_F_numeric_corr():
    cols = [c for c in num_cols if c in df.columns]
    if not cols:
        return
    corr = df[cols].astype(float).corr(method="pearson")
    plt.figure(figsize=(7, 6))
    plt.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
    plt.xticks(range(corr.shape[1]), corr.columns, rotation=90)
    plt.yticks(range(corr.shape[0]), corr.index)
    plt.colorbar()
    plt.title("Pearson correlation (numeric)")
    savefig("F_corr_numeric.png")


# ====== 6) 调用全部绘图 ======
plot_A_scatter_pacing()
plot_B_boxes()
plot_C_vstd_by_lighting_style()
plot_D_heatmap_color()
plot_E_family_and_global(metric="pmi", top_k_global=40)
plot_F_numeric_corr()

# ====== 7) 简要统计说明 ======
summary = {
    "total_json": len(all_files),
    "parsed_ok": len(df),
    "bad_files": len(bad),
    "missing_rate_numeric": {
        c: float(df[c].isna().mean()) for c in num_cols if c in df.columns
    },
}
with open(os.path.join(OUT_DIR, "README_from_json.txt"), "w", encoding="utf-8") as f:
    f.write("# LensLang EDA (from JSON, no CSV as input)\n")
    f.write(json.dumps(summary, ensure_ascii=False, indent=2))

print("Done.")
print("Run dir:", OUT_DIR)
print("CSV dir:", CSV_DIR)
print("FIG dir:", FIG_DIR)