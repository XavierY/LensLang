# -*- coding: utf-8 -*-
"""
annotate_lenslang_local32b.py
新增能力：
- 预检 out_dir 已有 JSON：判断是否“空/无效”，写 existing_output_<JOBID>.csv
- 仅对“尚未有效完成”的视频执行分析；有效=JSON存在且关键字段非空
- 对于失败的视频不中断，统计并写 unProcessed_<JOBID>.csv（含数量、序号、路径、错误原因、时间）
- 所有报告类CSV写到 --report_dir（建议传 logs/<JOB_NAME>）
"""

import os, gc, json, glob, math, argparse, time, random, re, csv, datetime, traceback
import numpy as np
import cv2
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ----------------------- Taxonomy 与 Schema -----------------------
TAXONOMY = {
    "shot_types": ["远景","中景","近景","特写","极近特写"],
    "camera_movements": ["固定镜头","横移（横摇）","纵移（俯仰）","推镜头","拉镜头"],
    "editing_rhythm": ["very fast","fast","moderate","slow"],
    "composition_styles": ["对称构图","三分法","中心构图","留白","低角度透视","高角度俯视","深焦透视","浅景深"],
    "color_tone": ["冷色调","中性","暖色调"],
    "lighting_style": ["高对比","中等对比","柔和低对比","背光","侧光","顺光","自然光","人造光"],
    "overall_style": ["纪实","叙事","新闻","vlog","商业广告","电影感","紧张","温暖","肃穆","活力"]
}

SCHEMA_TEMPLATE = {
    "video_analysis": {
        "shot_types": [],
        "camera_movements": [],
        "editing_rhythm": "",
        "composition_styles": [],
        "color_tone": "",
        "lighting_style": "",
        "overall_style": [],
        "confidence_score": 0.0
    },
    "key_characteristics": [],
    "style_tags": []
}
SCHEMA_JSON = json.dumps(SCHEMA_TEMPLATE, ensure_ascii=False, indent=2)

# ----------------------- 提示词 -----------------------
SYS_PROMPT = (
    "你是专业的影视镜头语言分析专家。"
    "只输出一个 JSON 对象，不要任何解释、不要 Markdown 代码块、不要反引号、不要 <think> 内容。"
)

USER_TMPL = """请根据下述“可观测统计”和“标签枚举”，
按“固定键名 + 枚举值”的要求输出一个 JSON（仅 JSON，本行以下的 schema 为硬性约束）。

[固定键名（必须完全一致）]
video_analysis, shot_types, camera_movements, editing_rhythm,
composition_styles, color_tone, lighting_style, overall_style, confidence_score,
key_characteristics, style_tags

[JSON Schema（示例结构，仅键名与类型约束）]
{schema}

[可观测统计（仅供参考）]
{stats}

[标签枚举（taxonomy，所有可用取值）]
{taxonomy}

规则：
- 所有取值必须从 taxonomy 中选择；
- style_tags 选 5–10 个来自 taxonomy 的词；
- confidence_score 取 [0,1]。
只返回 JSON 本体。"""

# ----------------------- 基础工具 -----------------------
def ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def list_videos(d):
    exts = (".mp4",".mov",".mkv",".avi",".MP4",".MOV",".MKV",".AVI")
    return sorted([p for p in glob.glob(os.path.join(d, "*")) if p.endswith(exts)])

def extract_json(text: str):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        frag = m.group(0)
        try:
            return json.loads(frag)
        except Exception:
            frag2 = re.sub(r",\s*([}\]])", r"\1", frag)
            try:
                return json.loads(frag2)
            except Exception:
                return {}
    return {}

# ----------------------- 预检相关：有效性判定 -----------------------
REQ_LIST_FIELDS = ["shot_types", "camera_movements", "composition_styles", "overall_style"]
REQ_STR_FIELDS  = ["editing_rhythm", "color_tone", "lighting_style"]

def _is_valid_analysis(analysis: dict) -> bool:
    """是否满足“应填充的内容不能空”的硬条件。"""
    if not isinstance(analysis, dict): return False
    va = analysis.get("video_analysis", {})
    if not isinstance(va, dict): return False
    # 列表字段需非空
    for k in REQ_LIST_FIELDS:
        v = va.get(k, [])
        if not isinstance(v, list) or len(v) == 0:
            return False
    # 字符串字段需非空
    for k in REQ_STR_FIELDS:
        v = va.get(k, "")
        if not isinstance(v, str) or not v.strip():
            return False
    # 置信分数存在且数值
    try:
        _ = float(va.get("confidence_score", 0.0))
    except Exception:
        return False
    return True

def _load_existing_json(path: str):
    if not os.path.isfile(path): return None, False
    try:
        if os.path.getsize(path) == 0:  # 0字节视为空
            return None, False
    except OSError:
        return None, False
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return None, False

    # 兼容两种结构：老格式 row={"video_id","stats","analysis"} 与 直接就是 analysis
    if isinstance(data, dict) and "analysis" in data:
        analysis = data.get("analysis", {})
    else:
        analysis = data
    return analysis, _is_valid_analysis(analysis)

def _iso(t: float) -> str:
    return datetime.datetime.fromtimestamp(t).isoformat(timespec="seconds")

def precheck_out_dir(out_dir: str, report_dir: str, job_id: str, video_dir: str):
    """扫描 out_dir 与 video_dir：
    生成 existing_output_<jobid>.csv（含 1–7 汇总与明细），
    并返回“有效完成”的视频ID集合（用来后续跳过已完成且有效的样本）。
    """
    ensure_dir(report_dir)

    # 1) 读取 out_dir 中的 JSON 列表
    json_paths = sorted(glob.glob(os.path.join(out_dir, "*.json")))
    out_json_total = len(json_paths)

    # 2) 读取数据集视频列表（只取基名作为ID）
    dataset_video_paths = list_videos(video_dir)
    dataset_ids = set(os.path.splitext(os.path.basename(p))[0] for p in dataset_video_paths)

    report_path = os.path.join(report_dir, f"existing_output_{job_id}.csv")
    processed_good = set()   # 校验通过（关键字段不空）的 ID
    json_ids = set()         # out_lang 中所有 json 的 ID（不论是否有效）
    empty_json_count = 0     # 无效/空 JSON 个数（0字节/解析失败/关键字段空）

    # 3) 遍历 JSON，统计有效性并写“明细”准备数据
    rows_detail = []
    for i, p in enumerate(json_paths, 1):
        try:
            st = os.stat(p)
            mtime = _iso(st.st_mtime)   # Linux 无通用创建时间，用 mtime
        except Exception:
            mtime = ""
        fname = os.path.basename(p)
        vid = os.path.splitext(fname)[0]
        json_ids.add(vid)

        analysis, valid = _load_existing_json(p)
        is_empty = (not valid)
        if is_empty:
            empty_json_count += 1
        else:
            processed_good.add(vid)

        rows_detail.append([i, fname, p, mtime, "YES" if is_empty else "NO"])

    # 4) 计算 1–7 指标
    # 2) 能够 match 数据集的数量（不看有效性，只要有同名 .json 即算匹配）
    matched_in_dataset = len(dataset_ids & json_ids)

    # 3) 与原数据集比还有多少未处理（= 数据集总数 - 已匹配数量）
    unprocessed_vs_dataset = max(0, len(dataset_ids) - matched_in_dataset)

    # 5) 没有 match 上的数据（孤儿 json：出现在 out_lang、但数据集里没有同名视频）
    json_without_matching_video_count = len(json_ids - dataset_ids)

    # 6) 总异常 = 空值 json + 孤儿 json
    total_anomaly_count = empty_json_count + json_without_matching_video_count

    # 7) 预估需要处理 = 未处理数量 + 异常数量
    estimated_to_process = unprocessed_vs_dataset + total_anomaly_count

    # 5) 写 CSV：先写 1–7 汇总，再写明细表头与明细
    with open(report_path, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)

        # ---- 头部汇总（按你要求的 1–7 顺序）----
        w.writerow(["out_json_total", out_json_total])                                     # 1
        w.writerow(["matched_in_dataset", matched_in_dataset])                              # 2
        w.writerow(["unprocessed_vs_dataset", unprocessed_vs_dataset])                      # 3
        w.writerow(["empty_json_count", empty_json_count])                                  # 4
        w.writerow(["json_without_matching_video_count", json_without_matching_video_count])# 5
        w.writerow(["total_anomaly_count", total_anomaly_count])                            # 6
        w.writerow(["estimated_to_process", estimated_to_process])                          # 7

        # ---- 明细表头 + 明细（保持不变）----
        w.writerow(["seq", "filename", "full_path", "modified_time_iso", "is_empty"])
        w.writerows(rows_detail)

    print(f"[REPORT] existing outputs -> {report_path} "
          f"(json_total={out_json_total}, matched={matched_in_dataset}, "
          f"unprocessed={unprocessed_vs_dataset}, empty={empty_json_count}, "
          f"orphan_json={json_without_matching_video_count}, "
          f"anomaly={total_anomaly_count}, estimate={estimated_to_process})")

    return processed_good, report_path

# ----------------------- 视觉侧：SBD/运动/光线/节奏 -----------------------
def _lighting_from_bgr(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H,S,V = hsv[...,0]/180.0, hsv[...,1]/255.0, hsv[...,2]/255.0
    v_std  = float(V.std()); v_mean = float(V.mean()); s_mean = float(S.mean())
    quality = "高对比" if v_std>0.23 else ("中等对比" if v_std>0.12 else "柔和低对比")
    cond    = "晴朗/高亮" if v_mean>0.7 else ("阴天/中性" if v_mean>0.45 else "昏暗/弱光")
    cold = float(((H>0.5)&(H<0.75)).mean()); warm = float(((H<0.12)|(H>0.83)).mean())
    temp = "冷色调" if cold>warm else ("暖色调" if warm>cold else "中性")
    return dict(quality=quality, condition=cond, s_mean=float(s_mean), v_mean=float(v_mean), v_std=float(v_std), color_temp=temp)

def _ecc_motion(a, b):
    g1=cv2.cvtColor(a, cv2.COLOR_BGR2GRAY); g2=cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    warp=np.eye(2,3,dtype=np.float32)
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
    try:
        cv2.findTransformECC(g1, g2, warp, cv2.MOTION_AFFINE, criteria)
    except cv2.error:
        pass
    dx,dy = float(warp[0,2]), float(warp[1,2])
    a11,a21 = warp[0,0], warp[1,0]
    scale = math.sqrt(max(1e-6, a11*a11 + a21*a21))
    if abs(scale-1.0) > 0.02:
        mtype = "推镜头" if scale>1 else "拉镜头"
    elif abs(dx) > abs(dy) and abs(dx) > 2.0:
        mtype = "横移（横摇）"
    elif abs(dy) > 2.0:
        mtype = "纵移（俯仰）"
    else:
        mtype = "固定镜头"
    return mtype, "NA"

def simple_sbd(video_path, thr=0.55, min_scene_sec=0.35, sample_stride=1, resize_w=320):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"无法打开视频: {video_path}"
    fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nfrm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    prev_hist=None; cuts=[]; last_cut=-1e9; min_gap = int(fps * min_scene_sec)

    def frame_hist(img):
        if resize_w and w>0:
            rh = int(img.shape[0] * (resize_w / img.shape[1])); img = cv2.resize(img, (resize_w, rh))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv],[0,1,2],None,[16,16,16],[0,180,0,256,0,256])
        cv2.normalize(hist, hist)
        return hist

    fidx=0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if sample_stride>1 and (fidx%sample_stride)!=0:
            fidx+=1; continue
        hist = frame_hist(frame)
        if prev_hist is not None:
            dist = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if dist > thr and (fidx - last_cut) >= min_gap:
                cuts.append(fidx); last_cut=fidx
        prev_hist = hist; fidx += 1
    cap.release()

    def cuts_to_shots(cut_idx, nfrm, fps):
        cuts = sorted(set([c for c in cut_idx if 0 < c < nfrm]))
        starts = [0] + cuts; ends = [c-1 for c in cuts] + [nfrm-1]
        shots=[]
        for s,e in zip(starts, ends):
            ss, ee = s/fps, e/fps
            shots.append({"start_idx":int(s),"end_idx":int(e),
                          "start_sec":round(ss,3),"end_sec":round(ee,3),
                          "mid_idx":int((s+e)//2),"mid_sec":round((ss+ee)/2,3)})
        return shots

    shots = cuts_to_shots(cuts, nfrm, fps)
    return {"fps": float(fps), "num_frames": nfrm, "w": w, "shots": shots}

def analyze_video(video_path):
    sbd = simple_sbd(video_path)
    fps = sbd["fps"]; shots = sbd["shots"]; n = len(shots)
    cap0 = cv2.VideoCapture(video_path)
    cap0.set(cv2.CAP_PROP_POS_FRAMES, shots[0]["mid_idx"] if n else 0)
    ok, f0 = cap0.read()
    lighting = _lighting_from_bgr(f0 if ok else np.zeros((32,32,3), np.uint8))

    move_names = ["固定镜头","横移（横摇）","纵移（俯仰）","推镜头","拉镜头"]
    mv_cnt = {k:0 for k in move_names}
    for sh in shots:
        s, e = sh["start_idx"], sh["end_idx"]
        cap0.set(cv2.CAP_PROP_POS_FRAMES, s); ok1, a = cap0.read()
        cap0.set(cv2.CAP_PROP_POS_FRAMES, e); ok2, b = cap0.read()
        if ok1 and ok2:
            mtype, _ = _ecc_motion(a, b); mv_cnt[mtype] = mv_cnt.get(mtype,0)+1
    cap0.release()

    lens = [(sh["end_sec"]-sh["start_sec"]) for sh in shots] if n else [0.0]
    ASL  = float(np.mean(lens))
    duration = shots[-1]["end_sec"]-shots[0]["start_sec"] if n else 0.0
    cuts_per_30s = (max(0,n-1)/duration*30.0) if duration>0 else 0.0

    def label_speed(c30): return "very fast" if c30>=8 else ("fast" if c30>=5 else ("moderate" if c30>=3 else "slow"))

    stats = {
        "video_name": os.path.basename(video_path),
        "num_shots": n,
        "avg_shot_len_sec": round(ASL,3),
        "cuts_per_30s": round(cuts_per_30s,3),
        "editing_rhythm_guess": label_speed(cuts_per_30s),
        "lighting": {
            "quality": lighting["quality"],
            "condition": lighting["condition"],
            "color_temperature": lighting["color_temp"],
            "s_mean": round(lighting["s_mean"],3),
            "v_mean": round(lighting["v_mean"],3),
            "v_std":  round(lighting["v_std"],3)
        },
        "movement_counts": mv_cnt
    }
    return stats

# ----------------------- vLLM 侧（与原版一致） -----------------------
def build_messages(stats_row, taxonomy):
    stats_str = json.dumps(stats_row, ensure_ascii=False)
    taxo_str  = json.dumps(taxonomy, ensure_ascii=False)
    user = USER_TMPL.format(stats=stats_str, taxonomy=taxo_str, schema=SCHEMA_JSON)
    return [{"role":"system","content":SYS_PROMPT},
            {"role":"user","content":user}]

def chat_to_prompt(tokenizer, messages):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    buf=[]; 
    for m in messages: buf.append(f"### {m['role'].upper()}:\n{m['content']}\n")
    buf.append("### ASSISTANT:\n"); 
    return "\n".join(buf)

def heuristic_from_stats(stats):
    o = json.loads(json.dumps(SCHEMA_TEMPLATE, ensure_ascii=False))
    o["video_analysis"]["editing_rhythm"] = stats.get("editing_rhythm_guess","moderate")
    ct = stats.get("lighting",{}).get("color_temperature","中性")
    if ct in TAXONOMY["color_tone"]: o["video_analysis"]["color_tone"] = ct
    qual = stats.get("lighting",{}).get("quality","中等对比")
    if qual not in TAXONOMY["lighting_style"]:
        qual = {"高对比":"高对比","中等对比":"中等对比","柔和低对比":"柔和低对比"}.get(qual,"中等对比")
    o["video_analysis"]["lighting_style"] = qual
    mv = stats.get("movement_counts",{})
    if mv:
        top = max(mv.items(), key=lambda x:x[1])
        if top[1] > 0 and top[0] in TAXONOMY["camera_movements"]:
            o["video_analysis"]["camera_movements"] = [top[0]]
    o["video_analysis"]["shot_types"] = ["中景"]
    o["video_analysis"]["composition_styles"] = ["三分法"]
    overall = ["纪实","叙事"]
    if ct == "暖色调": overall.append("温暖")
    o["video_analysis"]["overall_style"] = [x for x in overall if x in TAXONOMY["overall_style"]]
    tags = set(o["video_analysis"]["shot_types"] + o["video_analysis"]["camera_movements"]
               + [o["video_analysis"]["editing_rhythm"]]
               + o["video_analysis"]["composition_styles"]
               + [o["video_analysis"]["color_tone"], o["video_analysis"]["lighting_style"]]
               + o["video_analysis"]["overall_style"])
    all_allowed = set()
    for k in TAXONOMY:
        if isinstance(TAXONOMY[k], list):
            all_allowed.update(TAXONOMY[k])
    o["style_tags"] = [t for t in tags if t in all_allowed][:10]
    o["video_analysis"]["confidence_score"] = 0.30
    return o

def run_one_with_vllm(llm, tokenizer, messages, max_tokens=512, temperature=0.2, top_p=0.9, stats=None):
    params  = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    prompt = chat_to_prompt(tokenizer, messages)
    out = llm.generate([prompt], params, use_tqdm=False)[0]
    text = out.outputs[0].text if out.outputs else ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?\s*|```$", "", text, flags=re.I|re.M).strip()
    obj  = extract_json(text)

    if not obj:
        strict_user = ("上一次输出无效。现在请仅输出一个 JSON，严格遵循以下 Schema 键名：\n"
                       f"{SCHEMA_JSON}\n不要任何解释/前后缀/反引号/Markdown/think。")
        msgs2 = [{"role":"system","content":SYS_PROMPT},
                 {"role":"user","content": strict_user},
                 messages[-1]]
        prompt2 = chat_to_prompt(tokenizer, msgs2)
        out2 = llm.generate([prompt2], params, use_tqdm=False)[0]
        text2 = out2.outputs[0].text if out2.outputs else ""
        text2 = re.sub(r"<think>.*?</think>", "", text2, flags=re.S).strip()
        if text2.startswith("```"):
            text2 = re.sub(r"^```(json)?\s*|```$", "", text2, flags=re.I|re.M).strip()
        obj = extract_json(text2)

    if not obj and stats is not None:
        return heuristic_from_stats(stats)
    return clamp_to_taxonomy(obj, TAXONOMY) if obj else heuristic_from_stats(stats or {})

def clamp_to_taxonomy(obj, taxo):
    out = json.loads(json.dumps(SCHEMA_TEMPLATE, ensure_ascii=False))
    va = obj.get("video_analysis", {}) if isinstance(obj, dict) else {}
    def clip_list(field, allowed):
        vals = va.get(field, [])
        if not isinstance(vals, list):
            vals = [vals] if vals else []
        alias = {
            "shot_types": {"大全景":"远景"},
            "camera_movements": {"横摇":"横移（横摇）","平移":"横移（横摇）","俯仰":"纵移（俯仰）"},
            "editing_rhythm": {"慢":"slow","适中":"moderate","快":"fast","很快":"very fast"},
            "overall_style": {"电影风":"电影感"}
        }
        vals = [alias.get(field, {}).get(v, v) for v in vals]
        return [v for v in vals if v in allowed]
    out["video_analysis"]["shot_types"] = clip_list("shot_types", taxo.get("shot_types", []))
    out["video_analysis"]["camera_movements"] = clip_list("camera_movements", taxo.get("camera_movements", []))
    er = va.get("editing_rhythm", "")
    er = {"慢":"slow","适中":"moderate","快":"fast","很快":"very fast"}.get(er, er)
    if er in taxo.get("editing_rhythm", []): out["video_analysis"]["editing_rhythm"] = er
    out["video_analysis"]["composition_styles"] = clip_list("composition_styles", taxo.get("composition_styles", []))
    ct = va.get("color_tone", "");  ls = va.get("lighting_style", "")
    if ct in taxo.get("color_tone", []): out["video_analysis"]["color_tone"] = ct
    if ls in taxo.get("lighting_style", []): out["video_analysis"]["lighting_style"] = ls
    out["video_analysis"]["overall_style"] = clip_list("overall_style", taxo.get("overall_style", []))
    try:
        cs = float(va.get("confidence_score", 0.0)); cs = max(0.0, min(1.0, cs))
    except Exception:
        cs = 0.0
    out["video_analysis"]["confidence_score"] = cs

    kc = obj.get("key_characteristics", []); st = obj.get("style_tags", [])
    if isinstance(kc, list): out["key_characteristics"] = [str(x) for x in kc][:20]
    if not isinstance(st, list): st = [st] if st else []
    allowed_all = set()
    for k in ("shot_types","camera_movements","editing_rhythm","composition_styles","color_tone","lighting_style","overall_style"):
        allowed_all.update(taxo.get(k, []))
    out["style_tags"] = [x for x in st if x in allowed_all][:10]
    return out

# ----------------------- 主流程：预检→选择未处理→逐个生成→收尾报告 -----------------------
def main(args):
    video_dir = args.video_dir
    out_dir   = args.out_dir
    report_dir= args.report_dir or out_dir
    job_id    = args.job_id or "manual"
    ensure_dir(out_dir); ensure_dir(report_dir)

    # 1) 预检已有 JSON，写报告 & 获取“已有效完成”的 video_id 集合
    processed_good, exist_report = precheck_out_dir(out_dir, report_dir, job_id, video_dir)

    # 2) 构建待处理列表：目录下所有视频 - processed_good
    vids = list_videos(video_dir)
    if not vids:
        print(f"[WARN] 没找到视频：{video_dir}")
        return
    to_process = [v for v in vids if os.path.splitext(os.path.basename(v))[0] not in processed_good]
    print(f"[INFO] 总视频 {len(vids)}，已有效完成 {len(processed_good)}，待处理 {len(to_process)}")

    # 3) 初始化模型
    print(f"[INFO] 加载本地模型：{args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    llm = LLM(
        model=args.model,
        dtype="float16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=args.tensor_parallel_size,
        swap_space=args.swap_space,
        enforce_eager=True
    )

    unproc = []  # 未成功条目：seq, filename, path, reason, time
    seq = 0
    try:
        for v in to_process:
            seq += 1
            vid = os.path.splitext(os.path.basename(v))[0]
            out_path = os.path.join(out_dir, vid + ".json")
            try:
                print(f"[RUN] ({seq}/{len(to_process)}) {vid} … 统计中")
                stats = analyze_video(v)

                print(f"[RUN] {vid} … 调用 vLLM 生成")
                messages = build_messages(stats, TAXONOMY)
                analysis = run_one_with_vllm(
                    llm, tokenizer, messages,
                    max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p,
                    stats=stats
                )
                row = {"video_id": vid, "stats": stats, "analysis": analysis}
                with open(out_path, "w", encoding="utf-8") as fw:
                    json.dump(row, fw, ensure_ascii=False, indent=2)

                # 写完再做一次“有效性”校验；如失败则按未处理记录
                _, valid = _load_existing_json(out_path)
                if not valid:
                    raise RuntimeError("写出JSON后验证无效（关键字段空）")

                time.sleep(random.uniform(0.05, 0.15))
                print(f"[OK] 写入：{out_path}")

            except Exception as e:
                emsg = f"{type(e).__name__}: {e}"
                if args.verbose_errors:
                    traceback.print_exc()
                print(f"[SKIP-ERR] {vid} -> {emsg}")
                unproc.append([seq, os.path.basename(v), v, emsg, datetime.datetime.now().isoformat(timespec="seconds")])
                if not args.continue_on_error:
                    break

    finally:
        # 4) 写未处理报告
        unproc_csv = os.path.join(report_dir, f"unProcessed_{job_id}.csv")
        with open(unproc_csv, "w", newline="", encoding="utf-8") as fw:
            w = csv.writer(fw)
            w.writerow(["total_unprocessed", len(unproc)])
            w.writerow(["seq", "filename", "full_path", "error_reason", "error_time"])
            w.writerows(unproc)
        print(f"[REPORT] unprocessed -> {unproc_csv} （未处理 {len(unproc)} 个）")

        # 5) 清理显存与资源
        try:
            if hasattr(llm, "llm_engine") and hasattr(llm.llm_engine, "shutdown"):
                llm.llm_engine.shutdown()
        except Exception:
            pass
        del llm, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print("[CLEAN] 资源释放完成")

# ----------------------- CLI -----------------------
# ---- argparse（替换你那一段）----
if __name__ == "__main__":
    import os, argparse

    ap = argparse.ArgumentParser()

    # 不要给布尔 default；用 None，然后在解析后自行补全
    # 环境变量指定：
    # export LENSLANG_VIDEO_DIR="$VIDEO_DIR"
    # export LENSLANG_OUT_DIR="$OUT_DIR"
    # export LENSLANG_MODEL="$MODEL_PATH"
    ap.add_argument("--video_dir", type=str, default=None,
                    help="视频目录。也可用环境变量 LENSLANG_VIDEO_DIR 指定")
    ap.add_argument("--out_dir",   type=str, default=None,
                    help="输出JSON目录。也可用环境变量 LENSLANG_OUT_DIR 指定")
    ap.add_argument("--model",     type=str, default=None,
                    help="本地模型路径。也可用环境变量 LENSLANG_MODEL 指定")

    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--max_model_len", type=int, default=4096, help="vLLM最大上下文长度")
    ap.add_argument("--gpu_mem_util", type=float, default=0.90, help="显存使用比例(0-1)")
    ap.add_argument("--tensor_parallel_size", type=int, default=1, help="多卡并行数")
    ap.add_argument("--swap_space", type=int, default=8, help="CPU交换空间(GB)")

    # 报告/鲁棒
    ap.add_argument("--job_id", default="manual", help="用于命名CSV报告的作业ID")
    ap.add_argument("--report_dir", type=str, default=None,
                    help="报告输出目录(默认：<out_dir>/../logs/当前作业名)")
    ap.add_argument("--continue_on_error", action="store_true", help="单个视频失败时继续执行")
    ap.add_argument("--verbose_errors", action="store_true", help="打印完整traceback")

    args = ap.parse_args()

    # ---- 路径兜底顺序：CLI > 环境变量 > （可选）内置默认 ----
    # 如需“完全无硬编码”，不要写硬编码默认，保持 None 即可。
    # 若你希望 py 单独运行也能直接用，可在此处给个你本机默认值。
    DEFAULTS = {
        "video_dir": "/data/hyan671/yhproject/FakingRecipe/dataio/FakeSV/videos",
        "out_dir":   "/data/hyan671/yhproject/FakingRecipe/lenslang/outputs/out_lang",
        "model":     "/data/hyan671/models/DeepSeek-R1-Distill-Qwen-32B",
    }

    args.video_dir = args.video_dir or os.environ.get("LENSLANG_VIDEO_DIR") or DEFAULTS["video_dir"]
    args.out_dir   = args.out_dir   or os.environ.get("LENSLANG_OUT_DIR")   or DEFAULTS["out_dir"]
    args.model     = args.model     or os.environ.get("LENSLANG_MODEL")     or DEFAULTS["model"]

    # report_dir 默认：logs/<job_name>；单独跑时用 <out_dir>/../logs/manual
    if args.report_dir is None:
        job_name = os.environ.get("SLURM_JOB_NAME", "manual")
        # 把 out_dir 的上级作为基底放 logs
        base_logs = os.path.join(os.path.dirname(args.out_dir), "..", "logs")
        args.report_dir = os.path.abspath(os.path.join(base_logs, job_name))

    # 基本校验（更早发现错误）
    missing = [k for k in ("video_dir","out_dir","model") if not getattr(args, k)]
    if missing:
        ap.error(f"缺少必需路径: {', '.join(missing)}。"
                 f"请通过 CLI 传入或设置环境变量 LENSLANG_VIDEO_DIR/LENSLANG_OUT_DIR/LENSLANG_MODEL")

    main(args)