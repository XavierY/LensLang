#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
json2pkl_lenslang.py  —  LensLang JSON → Fakerecipe 可用 PKL（精简字段版）

运行建议（任选其一）：
  1) 已在 lenslang 环境中：   python json2pkl_lenslang.py
  2) 通过 conda run 调用：    conda run -n lenslang python json2pkl_lenslang.py

目录策略（自动创建时间戳目录）：
  OUT_ROOT / FakeTT_pkl_1991_NoStyleTag_<timestamp> / pkl    # 存 <video_id>.pkl + lenslang_meta.pkl
                                                           / report # 存 CSV 报告（词表、索引映射、统计等）

—— 仅保留的数值/类别字段 —— 
数值（log1p → z-score）：
   stats.num_shots, stats.avg_shot_len_sec, stats.cuts_per_30s,
   stats.movement_counts.{static_shot, pan_horizontal, tilt_vertical, dolly_in, dolly_out}

movement “rate” 两版（仅 z-score）：
   rate_per_shot.{static_shot, pan_horizontal, tilt_vertical, dolly_in, dolly_out}
   rate_per_30s.{static_shot, pan_horizontal, tilt_vertical, dolly_in, dolly_out}

单选 one-hot（缺失→"unknown"；editing_rhythm 以 analysis.* 优先）：
   analysis.video_analysis.editing_rhythm,
   stats.lighting.condition,
   analysis.video_analysis.color_tone,
   analysis.video_analysis.lighting_style

多选 multi-hot（缺失为空）：
   analysis.video_analysis.camera_movements,
   analysis.video_analysis.overall_style
"""

import os, json, glob, math, time, pickle, traceback, csv
from collections import defaultdict, Counter, OrderedDict
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
# 数值：log1p+z
NUM_LOGZ_FIELDS = [
    "stats.num_shots",
    "stats.avg_shot_len_sec",
    "stats.cuts_per_30s",
    "stats.movement_counts.static_shot",
    "stats.movement_counts.pan_horizontal",
    "stats.movement_counts.tilt_vertical",
    "stats.movement_counts.dolly_in",
    "stats.movement_counts.dolly_out",
]

# 数值：仅 z（本版不再使用，但保留空结构以兼容旧代码流程）
NUM_Z_FIELDS = []  # 原来的 lighting.s_mean/v_mean/v_std, confidence_score 已去掉

# 单选：editing_rhythm + condition/color_tone/lighting_style
CAT_SINGLE_FIELDS = [
    "analysis.video_analysis.editing_rhythm",
    "stats.lighting.condition",
    "analysis.video_analysis.color_tone",
    "analysis.video_analysis.lighting_style",
]

# 多选：camera_movements, overall_style
CAT_MULTI_FIELDS = [
    "analysis.video_analysis.camera_movements",
    "analysis.video_analysis.overall_style",
]

# movement key（用于 counts 和 rate）
MOV_KEYS_ENG = ["static_shot", "pan_horizontal", "tilt_vertical", "dolly_in", "dolly_out"]

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
    mean = float(np.mean(arr)) if arr.size > 0 else 0.0
    std  = float(np.std(arr))  if arr.size > 0 else 1.0
    if std < 1e-8:
        std = 1.0
    return mean, std

def zscore(x, mean, std):
    return (x - mean) / std

# ========== 主流程 ==========
def main(json_dir, out_root):
    assert os.path.isdir(json_dir), f"JSON 目录不存在：{json_dir}"
    os.makedirs(out_root, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(out_root, f"FakeSV_pkl_3624_{ts}")
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
    logz_collector, logz_median_raw = defaultdict(list), defaultdict(list)
    z_collector = defaultdict(list)            # 现在不会被填充，但结构保留
    rate_per_shot, rate_per_30s = defaultdict(list), defaultdict(list)

    cat_single_vocab = {f: set() for f in CAT_SINGLE_FIELDS}
    cat_multi_vocab  = {f: set() for f in CAT_MULTI_FIELDS}
    unknown_counts = Counter()
    bad_files = []

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

        num_shots = safe_get(jd, "stats.num_shots", None)
        avg_len   = safe_get(jd, "stats.avg_shot_len_sec", None)

        # 数值：log1p+z
        for path in NUM_LOGZ_FIELDS:
            val = safe_get(jd, path, None)
            if val is None:
                continue
            try:
                v = float(val)
            except:
                continue
            logz_median_raw[path].append(v)
            logz_collector[path].append(math.log1p(max(v, 0.0)))

        # 数值：z（本版无字段，但流程保留）
        for path in NUM_Z_FIELDS:
            val = safe_get(jd, path, None)
            if val is None:
                continue
            try:
                v = float(val)
            except:
                continue
            z_collector[path].append(v)

        # movement rate 相关统计
        mov_counts = safe_get(jd, "stats.movement_counts", {}) or {}
        mv = {}
        for k, v in mov_counts.items():
            kk = canon_label(k)
            mv[kk] = float(v) if v is not None else 0.0

        try:
            ns = float(num_shots) if num_shots is not None else 0.0
        except:
            ns = 0.0
        try:
            al = float(avg_len) if avg_len is not None else 0.0
        except:
            al = 0.0
        duration_sec = ns * al
        per30_denom   = max(duration_sec / 30.0, 1e-6)
        per_shot_denom= max(ns, 1.0)

        for k in MOV_KEYS_ENG:
            c = max(mv.get(k, 0.0), 0.0)
            rate_per_shot[f"rate_per_shot.{k}"].append(c / per_shot_denom)
            rate_per_30s[f"rate_per_30s.{k}"].append(c / per30_denom)

        # 单选：editing_rhythm 优先 analysis.*
        rh = safe_get(jd, "analysis.video_analysis.editing_rhythm", None)
        if rh is None:
            rh = safe_get(jd, "stats.editing_rhythm_guess", None)
        if rh is not None:
            cat_single_vocab["analysis.video_analysis.editing_rhythm"].add(canon_label(rh))

        # 其它单选字段：只保留 condition / color_tone / lighting_style
        for path in [
            "stats.lighting.condition",
            "analysis.video_analysis.color_tone",
            "analysis.video_analysis.lighting_style",
        ]:
            val = safe_get(jd, path, None)
            if val is not None:
                cat_single_vocab[path].add(canon_label(val))

        # 多选：camera_movements, overall_style
        for path in CAT_MULTI_FIELDS:
            vals = ensure_list(safe_get(jd, path, []))
            for t in vals:
                cat_multi_vocab[path].add(canon_label(t))

    # 全量统计（中位数/均值/方差）
    num_logz_median, num_logz_meanstd = {}, {}
    for path in NUM_LOGZ_FIELDS:
        med = float(np.median([math.log1p(max(v, 0.0)) for v in logz_median_raw[path]]) if logz_median_raw[path] else 0.0)
        num_logz_median[path] = med
        mean, std = compute_z_stats(logz_collector[path]) if logz_collector[path] else (0.0, 1.0)
        num_logz_meanstd[path] = (mean, std)

    num_z_median, num_z_meanstd = {}, {}
    for path in NUM_Z_FIELDS:
        med = float(np.median(z_collector[path])) if z_collector[path] else 0.0
        num_z_median[path] = med
        mean, std = compute_z_stats(z_collector[path]) if z_collector[path] else (0.0, 1.0)
        num_z_meanstd[path] = (mean, std)

    rate_fields = list(rate_per_shot.keys()) + list(rate_per_30s.keys())
    rate_median, rate_meanstd = {}, {}
    for k, arr in {**rate_per_shot, **rate_per_30s}.items():
        med = float(np.median(arr)) if arr else 0.0
        rate_median[k] = med
        mean, std = compute_z_stats(arr) if arr else (0.0, 1.0)
        rate_meanstd[k] = (mean, std)

    # 词表（unknown 放首位）
    def finalize_vocab(vset):
        toks = set(t for t in vset if t not in (None, "", "unknown"))
        return ["unknown"] + sorted(toks)

    cat_single_vocab_final = {f: finalize_vocab(s) for f, s in cat_single_vocab.items()}
    cat_multi_vocab_final  = {f: finalize_vocab(s) for f, s in cat_multi_vocab.items()}

    # 特征布局
    feature_index = []
    offset = 0
    # 数值：logz
    for path in NUM_LOGZ_FIELDS:
        feature_index.append(("numeric_logz", path, None, offset)); offset += 1
    # 数值：仅 z（本版为空，但保持流程）
    for path in NUM_Z_FIELDS:
        feature_index.append(("numeric_z", path, None, offset)); offset += 1
    # 数值：rate（per_shot 后 per_30s，均按 key 排序）
    for k in sorted(rate_per_shot.keys()):
        feature_index.append(("numeric_rate", k, None, offset)); offset += 1
    for k in sorted(rate_per_30s.keys()):
        feature_index.append(("numeric_rate", k, None, offset)); offset += 1

    # 单选
    cat_single_slices = {}
    for f, vocab in cat_single_vocab_final.items():
        start = offset
        for tok in vocab:
            feature_index.append(("cat_single", f, tok, offset)); offset += 1
        cat_single_slices[f] = (start, offset)

    # 多选
    cat_multi_slices = {}
    for f, vocab in cat_multi_vocab_final.items():
        start = offset
        for tok in vocab:
            feature_index.append(("cat_multi", f, tok, offset)); offset += 1
        cat_multi_slices[f] = (start, offset)

    dim = offset
    version = ts

    # 写 index 对照
    with open(index_csv, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)
        w.writerow(["index", "kind", "field", "value_or_transform"])
        for kind, field, val, idx in feature_index:
            xform = ""
            if kind == "numeric_logz":
                xform = "log1p+z"
            elif kind in ("numeric_z", "numeric_rate"):
                xform = "z"
            w.writerow([idx, kind, field, (val if val is not None else xform)])

    # 写各词表
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

    # 数值统计
    with open(numeric_stats_csv, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)
        w.writerow(["field", "transform", "mean", "std", "median", "notes"])
        for f in NUM_LOGZ_FIELDS:
            mean, std = num_logz_meanstd[f]; med = num_logz_median[f]
            w.writerow([f, "log1p+z", mean, std, med, ""])
        for f in NUM_Z_FIELDS:
            mean, std = num_z_meanstd[f]; med = num_z_median[f]
            w.writerow([f, "z", mean, std, med, ""])
        for k in sorted(rate_fields):
            mean, std = rate_meanstd[k]; med = rate_median[k]
            w.writerow([k, "z", mean, std, med, "movement rate"])

    # 便捷索引表（写向量时更快）
    idx_logz   = {f: i for (k, f, v, i) in feature_index if k == "numeric_logz"}
    idx_zonly  = {f: i for (k, f, v, i) in feature_index if k == "numeric_z"}
    idx_rates  = [i for (k, f, v, i) in feature_index if k == "numeric_rate"]  # 顺序=per_shot sorted 后 per_30s sorted
    single_slices = cat_single_slices
    multi_slices  = cat_multi_slices

    def get_logz_z(path, jd):
        raw = safe_get(jd, path, None)
        if raw is None:
            val = num_logz_median[path]
        else:
            try:
                val = math.log1p(max(float(raw), 0.0))
            except:
                val = num_logz_median[path]
        mean, std = num_logz_meanstd[path]
        return zscore(val, mean, std)

    def get_z_only(path, jd):
        raw = safe_get(jd, path, None)
        if raw is None:
            val = num_z_median[path]
        else:
            try:
                val = float(raw)
            except:
                val = num_z_median[path]
        mean, std = num_z_meanstd[path]
        return zscore(val, mean, std)

    def get_rates_z(jd):
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
        duration_sec = ns * al
        per30_denom   = max(duration_sec / 30.0, 1e-6)
        per_shot_denom= max(ns, 1.0)

        mov_counts = safe_get(jd, "stats.movement_counts", {}) or {}
        mv = {}
        for k, v in mov_counts.items():
            kk = canon_label(k)
            mv[kk] = float(v) if v is not None else 0.0

        vals = []
        # per_shot（按 key 排序）
        for tag in sorted([k.split(".", 1)[1] for k in rate_per_shot.keys()]):
            rate = max(mv.get(tag, 0.0), 0.0) / per_shot_denom
            mean, std = rate_meanstd[f"rate_per_shot.{tag}"]
            vals.append(zscore(rate, mean, std))
        # per_30s
        for tag in sorted([k.split(".", 1)[1] for k in rate_per_30s.keys()]):
            rate = max(mv.get(tag, 0.0), 0.0) / per30_denom
            mean, std = rate_meanstd[f"rate_per_30s.{tag}"]
            vals.append(zscore(rate, mean, std))
        return vals

    def get_single_token(field, jd):
        if field == "analysis.video_analysis.editing_rhythm":
            rh = safe_get(jd, "analysis.video_analysis.editing_rhythm", None)
            if rh is None:
                rh = safe_get(jd, "stats.editing_rhythm_guess", None)
            tok = canon_label(rh) if rh is not None else "unknown"
        else:
            val = safe_get(jd, field, None)
            tok = canon_label(val) if val is not None else "unknown"
        vocab = cat_single_vocab_final[field]
        if tok not in vocab:
            unknown_counts[field] += 1
            tok = "unknown"
        return tok

    def get_multi_tokens(field, jd):
        vals = ensure_list(safe_get(jd, field, []))
        toks = [canon_label(v) for v in vals if v is not None]
        vocab = set(cat_multi_vocab_final[field])
        return [t for t in toks if t in vocab and t != "unknown"]

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

        # 数值
        for path, idx in idx_logz.items():
            vec[idx] = get_logz_z(path, jd)
        for path, idx in idx_zonly.items():   # 目前为空
            vec[idx] = get_z_only(path, jd)
        rate_vals = get_rates_z(jd)
        vec[idx_rates] = np.asarray(rate_vals, dtype=np.float32)

        # 单选 one-hot
        for field, (start, end) in single_slices.items():
            vocab = cat_single_vocab_final[field]
            tok = get_single_token(field, jd)
            jdx = vocab.index(tok) if tok in vocab else 0
            vec[start + jdx] = 1.0

        # 多选 multi-hot
        for field, (start, end) in multi_slices.items():
            vocab = cat_multi_vocab_final[field]
            toks = get_multi_tokens(field, jd)
            for t in toks:
                jdx = vocab.index(t) if t in vocab else 0
                vec[start + jdx] = 1.0

        sample = {
            "video_id": vid,
            "lenslang_vec": vec,
            "dim": int(dim),
            "meta": {"version": version, "index_csv": os.path.basename(index_csv)},
        }
        with open(os.path.join(pkl_dir, f"{vid}.pkl"), "wb") as fw:
            pickle.dump(sample, fw, protocol=4)

    # unknown 报告
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
            "rate_fields": sorted(rate_fields),
            "logz_meanstd": num_logz_meanstd,
            "logz_median": num_logz_median,
            "z_meanstd": num_z_meanstd,
            "z_median": num_z_median,
            "rate_meanstd": rate_meanstd,
            "rate_median": rate_median,
        },
        "categorical": {
            "single": cat_single_vocab_final,
            "single_slices": cat_single_slices,
            "multi": cat_multi_vocab_final,
            "multi_slices": cat_multi_slices,
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