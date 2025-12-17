#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
json2pkl_lenslang.py  —  LensLang JSON → FakingRecipe 可用 PKL（精简字段 + flag 版）

保留字段 & 处理逻辑：

- 数值（分位数截断 → log1p → z-score）：
    stats.avg_shot_len_sec   -> avg_shot_len_sec
    stats.cuts_per_30s       -> cuts_per_30s

- 二元 / 连续 flag（均做 z-score）：
    flag.is_fast_rhythm              # fast | very_fast = 1, 其余 = 0
    flag.is_warm_tone                # color_tone == warm_tone
    flag.is_high_contrast            # lighting_style == high_contrast
    flag.is_pan_horizontal           # camera_movements 中包含 pan_horizontal
    flag.style_high_risk             # style_tags 命中 core_high
    flag.style_low_risk              # style_tags 命中 core_low
    rate_per_shot.pan_horizontal
    rate_per_shot.static_shot

说明：
- 不再保留任何 one-hot / multi-hot 类特征；
- 仅利用 analysis.* / stats.* 字段派生上述少量连续特征；
- 读写路径、输出目录结构、meta / report 命名保持不变。
"""

import os, json, glob, math, pickle, csv
from collections import defaultdict, Counter
from datetime import datetime
import numpy as np

# ========== 固定路径（不变） ==========
JSON_DIR  = "/data/hyan671/yhproject/FakingRecipe/lenslang/outputs/lenslang_FakeSV_no_keyChts_3624only_Complete"
OUT_ROOT  = "/data/hyan671/yhproject/FakingRecipe/lenslang/outputs/FakeSV_pkl_3624"

# ========== 轻量环境检查 ==========
ENV_EXPECTED = "lenslang"
if os.environ.get("CONDA_DEFAULT_ENV", "") != ENV_EXPECTED:
    print(f"[WARN] 当前 conda 环境不是 '{ENV_EXPECTED}'（CONDA_DEFAULT_ENV={os.environ.get('CONDA_DEFAULT_ENV','')}). "
          f"建议在该环境中运行，或使用：conda run -n {ENV_EXPECTED} python json2pkl_lenslang.py")

# ========== 规范化（已是英文，不再做中英映射） ==========
def norm_token(x):
    if x is None:
        return None
    if not isinstance(x, str):
        x = str(x)
    x = x.strip()
    x = x.replace("（", "(").replace("）", ")").replace("/", "_").replace("-", "_").replace(" ", "_")
    x = x.replace("__", "_")
    return x.lower()

def canon_label(x):
    """
    仅做简单规范化（大小写/空格/下划线），不再做中英映射。
    """
    if x is None:
        return None
    x = str(x).strip()
    x = x.replace("very fast", "very_fast")
    return norm_token(x)

# ========== 字段集合：只保留指定字段 ==========
# 数值：分位数截断 + log1p + z
NUM_LOGZ_FIELDS = [
    "stats.avg_shot_len_sec",   # avg_shot_len_sec
    "stats.cuts_per_30s",       # cuts_per_30s
]

# 二元 / 连续 z-score 特征
NUM_Z_FIELDS = [
    "flag.is_fast_rhythm",
    "flag.is_warm_tone",
    "flag.is_high_contrast",
    "flag.is_pan_horizontal",
    "flag.style_high_risk",
    "flag.style_low_risk",
    "rate_per_shot.pan_horizontal",
    "rate_per_shot.static_shot",
]

# movement rate 字段子集（用于 meta 中说明）
RATE_FIELDS = [
    "rate_per_shot.pan_horizontal",
    "rate_per_shot.static_shot",
]

# 为了兼容原 unknown_report.csv，保留 single 字段列表（但不再编码进特征）
CAT_SINGLE_FIELDS = [
    "analysis.video_analysis.editing_rhythm",
    "stats.lighting.condition",
    "analysis.video_analysis.color_tone",
    "analysis.video_analysis.lighting_style",
]

# 为了兼容原 categorical_vocab__*.csv，仅做统计用
CAT_MULTI_FIELDS = [
    "analysis.video_analysis.camera_movements",
    "analysis.video_analysis.overall_style",
    "analysis.style_tags",
]

# style_tags 高/低风险组合
CORE_HIGH = {
    "warm_tone",
    "warm_mood",
    "documentary_style",
    "narrative_style",
    "slow",
    "medium_contrast",
}

CORE_LOW = {
    "cool_tone",
    "documentary_style",
    "narrative_style",
    "slow",
    "medium_contrast",
}

# ========== 小工具 ==========
def safe_get(d, path, default=None):
    cur = d
    for k in path.split('.'):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [t.strip() for t in x.split("|") if t.strip()]
    return [x]

def compute_z_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 1.0
    mean = float(np.mean(arr))
    std  = float(np.std(arr))
    if std < 1e-8:
        std = 1.0
    return mean, std

def zscore(x, mean, std):
    return (x - mean) / std

def compute_flags_and_rates(jd):
    """
    从单条 JSON 中派生：
    - flag.* 系列二元特征
    - rate_per_shot.* 两个连续特征
    返回 dict[path -> raw_value]
    """
    feats = {}

    # --- editing_rhythm → is_fast_rhythm ---
    rh = safe_get(jd, "analysis.video_analysis.editing_rhythm", None)
    if rh is None:
        rh = safe_get(jd, "stats.editing_rhythm_guess", None)
    rh_tok = canon_label(rh) if rh is not None else None
    is_fast = 1.0 if rh_tok in ("fast", "very_fast") else 0.0
    feats["flag.is_fast_rhythm"] = is_fast

    # --- color_tone → is_warm_tone ---
    ct = canon_label(safe_get(jd, "analysis.video_analysis.color_tone", None))
    feats["flag.is_warm_tone"] = 1.0 if ct == "warm_tone" else 0.0

    # --- lighting_style → is_high_contrast ---
    ls = canon_label(safe_get(jd, "analysis.video_analysis.lighting_style", None))
    feats["flag.is_high_contrast"] = 1.0 if ls == "high_contrast" else 0.0

    # --- camera_movements → is_pan_horizontal ---
    cms = ensure_list(safe_get(jd, "analysis.video_analysis.camera_movements", []))
    cms_tok = {canon_label(v) for v in cms if v is not None}
    feats["flag.is_pan_horizontal"] = 1.0 if "pan_horizontal" in cms_tok else 0.0

    # --- style_tags → 高/低风险组合 flag ---
    sts = ensure_list(safe_get(jd, "analysis.style_tags", []))
    sts_tok = {canon_label(v) for v in sts if v is not None}
    feats["flag.style_high_risk"] = 1.0 if any(t in CORE_HIGH for t in sts_tok) else 0.0
    feats["flag.style_low_risk"]  = 1.0 if any(t in CORE_LOW  for t in sts_tok) else 0.0

    # --- movement_counts → rate_per_shot.* ---
    num_shots = safe_get(jd, "stats.num_shots", 0)
    avg_len   = safe_get(jd, "stats.avg_shot_len_sec", 0.0)
    try:
        ns = float(num_shots) if num_shots is not None else 0.0
    except:
        ns = 0.0
    try:
        al = float(avg_len) if avg_len is not None else 0.0
    except:
        al = 0.0

    per_shot_denom = max(ns, 1.0)

    mov_counts = safe_get(jd, "stats.movement_counts", {}) or {}
    mv = {}
    for k, v in mov_counts.items():
        kk = canon_label(k)
        try:
            mv[kk] = float(v) if v is not None else 0.0
        except:
            mv[kk] = 0.0

    for tag in ("pan_horizontal", "static_shot"):
        c = max(mv.get(tag, 0.0), 0.0)
        feats[f"rate_per_shot.{tag}"] = c / per_shot_denom

    return feats

# ========== 主流程 ==========
def main(json_dir, out_root):
    assert os.path.isdir(json_dir), f"JSON 目录不存在：{json_dir}"
    os.makedirs(out_root, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(out_root, f"FakeSV_pkl_3624_Roundxxx_{ts}")
    pkl_dir    = os.path.join(root, "pkl")
    report_dir = os.path.join(root, "report")
    os.makedirs(pkl_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # 报告路径
    index_csv         = os.path.join(report_dir, "feature_index_map.csv")
    numeric_stats_csv = os.path.join(report_dir, "numeric_stats.csv")
    unknown_csv       = os.path.join(report_dir, "unknown_report.csv")
    bad_json_csv      = os.path.join(report_dir, "bad_json_files.csv")

    # 预扫描容器
    logz_median_raw = defaultdict(list)   # 存原始数值，用于分位数截断 + log1p
    z_collector     = defaultdict(list)   # 存 flag / rate 的 raw 值

    # 为了兼容原 report，继续统计 vocab（但不编码进特征）
    cat_single_vocab = {f: set() for f in CAT_SINGLE_FIELDS}
    cat_multi_vocab  = {f: set() for f in CAT_MULTI_FIELDS}
    unknown_counts   = Counter()
    bad_files        = []

    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    if not json_files:
        raise RuntimeError(f"未在 {json_dir} 找到 JSON 文件")

    # -------- Pass-1：统计词表 & 全量统计 ----------
    for jp in json_files:
        try:
            with open(jp, "r", encoding="utf-8") as f:
                jd = json.load(f)
        except Exception as e:
            bad_files.append((os.path.basename(jp), str(e)))
            continue

        # 数值：avg_shot_len_sec, cuts_per_30s（先收集原始值，后面做分位数截断 + log1p）
        for path in NUM_LOGZ_FIELDS:
            val = safe_get(jd, path, None)
            if val is None:
                continue
            try:
                v = float(val)
            except:
                continue
            logz_median_raw[path].append(v)

        # flag & rate：统一用 helper 生成，再丢进 z_collector
        feats = compute_flags_and_rates(jd)
        for k, v in feats.items():
            z_collector[k].append(float(v))

        # 下面这块仅用于生成原有 categorical_vocab__*.csv，便于检查，不进特征向量
        # 单选字段 vocab
        rh = safe_get(jd, "analysis.video_analysis.editing_rhythm", None)
        if rh is None:
            rh = safe_get(jd, "stats.editing_rhythm_guess", None)
        if rh is not None:
            cat_single_vocab["analysis.video_analysis.editing_rhythm"].add(canon_label(rh))

        for path in [
            "stats.lighting.condition",
            "analysis.video_analysis.color_tone",
            "analysis.video_analysis.lighting_style",
        ]:
            val = safe_get(jd, path, None)
            if val is not None:
                cat_single_vocab[path].add(canon_label(val))

        # 多选字段 vocab
        for path in CAT_MULTI_FIELDS:
            vals = ensure_list(safe_get(jd, path, []))
            for t in vals:
                cat_multi_vocab[path].add(canon_label(t))

    # -------- 计算数值统计量（含分位数截断） ----------
    num_logz_median, num_logz_meanstd, logz_clip_hi = {}, {}, {}
    for path in NUM_LOGZ_FIELDS:
        raw = [max(float(v), 0.0) for v in logz_median_raw[path]] if logz_median_raw[path] else []
        if raw:
            hi = float(np.percentile(raw, 99.0))  # 上 99 分位截断
            logz_clip_hi[path] = hi
            transformed = [math.log1p(min(v, hi)) for v in raw]
            med = float(np.median(transformed))
            mean, std = compute_z_stats(transformed)
        else:
            logz_clip_hi[path] = 0.0
            med, mean, std = 0.0, 0.0, 1.0
        num_logz_median[path] = med
        num_logz_meanstd[path] = (mean, std)

    num_z_median, num_z_meanstd = {}, {}
    for path in NUM_Z_FIELDS:
        arr = z_collector[path]
        if arr:
            med = float(np.median(arr))
            mean, std = compute_z_stats(arr)
        else:
            med, mean, std = 0.0, 0.0, 1.0
        num_z_median[path] = med
        num_z_meanstd[path] = (mean, std)

    # movement rate 的统计（从 z_* 中抽取）
    rate_median  = {f: num_z_median[f]   for f in RATE_FIELDS}
    rate_meanstd = {f: num_z_meanstd[f]  for f in RATE_FIELDS}

    # -------- categorical vocab（仅写 CSV，特征中不用） ----------
    def finalize_vocab(vset):
        toks = set(t for t in vset if t not in (None, "", "unknown"))
        return ["unknown"] + sorted(toks)

    cat_single_vocab_final = {f: finalize_vocab(s) for f, s in cat_single_vocab.items()}
    cat_multi_vocab_final  = {f: finalize_vocab(s) for f, s in cat_multi_vocab.items()}

    for f, vocab in cat_single_vocab_final.items():
        p = os.path.join(report_dir, f'categorical_vocab__{f.replace(".", "_")}.csv')
        with open(p, "w", newline="", encoding="utf-8") as fw:
            w = csv.writer(fw)
            w.writerow(["token", "index_in_block"])
            for i, tok in enumerate(vocab):
                w.writerow([tok, i])

    for f, vocab in cat_multi_vocab_final.items():
        p = os.path.join(report_dir, f'categorical_vocab__{f.replace(".", "_")}.csv')
        with open(p, "w", newline="", encoding="utf-8") as fw:
            w = csv.writer(fw)
            w.writerow(["token", "index_in_block"])
            for i, tok in enumerate(vocab):
                w.writerow([tok, i])

    # -------- 特征布局（仅 numeric_*） ----------
    feature_index = []
    offset = 0

    # 数值：log1p+z（avg_shot_len_sec, cuts_per_30s）
    for path in NUM_LOGZ_FIELDS:
        feature_index.append(("numeric_logz", path, None, offset))
        offset += 1

    # 数值：z（flags + rate_per_shot.*）
    for path in NUM_Z_FIELDS:
        feature_index.append(("numeric_z", path, None, offset))
        offset += 1

    dim = offset
    version = ts

    # 写 index 对照
    with open(index_csv, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)
        w.writerow(["index", "kind", "field", "value_or_transform"])
        for kind, field, val, idx in feature_index:
            if kind == "numeric_logz":
                xform = "log1p+z+clip"
            elif kind == "numeric_z":
                xform = "z"
            else:
                xform = ""
            w.writerow([idx, kind, field, (val if val is not None else xform)])

    # 数值统计
    with open(numeric_stats_csv, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)
        w.writerow(["field", "transform", "mean", "std", "median", "notes"])
        for f in NUM_LOGZ_FIELDS:
            mean, std = num_logz_meanstd[f]
            med = num_logz_median[f]
            note = f"clip_hi={logz_clip_hi[f]:.4f}"
            w.writerow([f, "log1p+z+clip", mean, std, med, note])
        for f in NUM_Z_FIELDS:
            mean, std = num_z_meanstd[f]
            med = num_z_median[f]
            note = "movement rate" if f in RATE_FIELDS else ""
            w.writerow([f, "z", mean, std, med, note])

    # 便捷索引表（写向量时使用）
    idx_logz  = {f: i for (k, f, v, i) in feature_index if k == "numeric_logz"}
    idx_zonly = {f: i for (k, f, v, i) in feature_index if k == "numeric_z"}

    def get_logz_z(path, jd):
        raw = safe_get(jd, path, None)
        if raw is None:
            val = num_logz_median[path]  # 已是 log1p 后的中位数
        else:
            try:
                raw_f = max(float(raw), 0.0)
                raw_f = min(raw_f, logz_clip_hi[path])   # 分位数截断
                val = math.log1p(raw_f)
            except:
                val = num_logz_median[path]
        mean, std = num_logz_meanstd[path]
        return zscore(val, mean, std)

    # -------- Pass-2：写每视频向量 PKL ----------
    for jp in json_files:
        try:
            with open(jp, "r", encoding="utf-8") as f:
                jd = json.load(f)
        except Exception as e:
            bad_files.append((os.path.basename(jp), str(e)))
            continue

        vid = safe_get(jd, "video_id", None)
        if not vid:
            bad_files.append((os.path.basename(jp), "missing video_id"))
            continue

        vec = np.zeros((dim,), dtype=np.float32)

        # 数值：avg_shot_len_sec / cuts_per_30s
        for path, idx in idx_logz.items():
            vec[idx] = get_logz_z(path, jd)

        # 数值：flags + rate_per_shot.*
        feats = compute_flags_and_rates(jd)
        for path, idx in idx_zonly.items():
            raw = feats.get(path, None)
            if raw is None:
                raw = num_z_median[path]
            mean, std = num_z_meanstd[path]
            vec[idx] = zscore(raw, mean, std)

        sample = {
            "video_id": vid,
            "lenslang_vec": vec,
            "dim": int(dim),
            "meta": {"version": version, "index_csv": os.path.basename(index_csv)},
        }
        with open(os.path.join(pkl_dir, f"{vid}.pkl"), "wb") as fw:
            pickle.dump(sample, fw, protocol=4)

    # unknown 报告（保持接口，但这里不再统计，统一写 0）
    with open(unknown_csv, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)
        w.writerow(["field", "unknown_count"])
        for f in CAT_SINGLE_FIELDS:
            w.writerow([f, unknown_counts.get(f, 0)])

    # 坏文件清单
    if bad_files:
        with open(bad_json_csv, "w", newline="", encoding="utf-8") as fw:
            w = csv.writer(fw)
            w.writerow(["filename", "error"])
            for fn, err in bad_files:
                w.writerow([fn, err])

    # 全局 meta（存 root 下）
    global_meta = {
        "version": version,
        "dim": int(dim),
        "numeric": {
            "logz_fields": NUM_LOGZ_FIELDS,
            "z_fields": NUM_Z_FIELDS,
            "rate_fields": RATE_FIELDS,
            "logz_meanstd": num_logz_meanstd,
            "logz_median": num_logz_median,
            "logz_clip_hi": logz_clip_hi,
            "z_meanstd": num_z_meanstd,
            "z_median": num_z_median,
            "rate_meanstd": rate_meanstd,
            "rate_median": rate_median,
        },
        "categorical": {
            # 仅用于检查 vocab，不再出现在特征向量中
            "single": cat_single_vocab_final,
            "single_slices": {},
            "multi": cat_multi_vocab_final,
            "multi_slices": {},
        },
        "paths": {
            "feature_index_map_csv": os.path.basename(index_csv),
            "numeric_stats_csv": os.path.basename(numeric_stats_csv),
            "unknown_report_csv": os.path.basename(unknown_csv),
            "bad_json_files_csv": os.path.basename(bad_json_csv) if bad_files else None,
        },
    }
    with open(os.path.join(root, "lenslang_meta.pkl"), "wb") as fw:
        pickle.dump(global_meta, fw, protocol=4)

    print(f"[OK] Done.\n  Root:   {root}\n  pkl:    {pkl_dir}\n  report: {report_dir}\n  Dim:    {dim}\n  Files:  {len(json_files)}")
    if bad_files:
        print(f"  Bad JSON files: {len(bad_files)} -> see {bad_json_csv}")

if __name__ == "__main__":
    main(JSON_DIR, OUT_ROOT)