# -*- coding: utf-8 -*-
"""
convert_lenslang_to_eng.py

功能：
- 扫描一个目录下的所有 .json 文件
- 根据预先定义的中英对照表，将 JSON 中的中文枚举值替换为英文
- 同时把 movement_counts 里的中文 key（如 "固定镜头"）改为英文 key（如 "static_shot"）
- 将转换后的 JSON 写到输出目录，文件名保持不变

用法示例：
python convert_lenslang_to_eng.py \
  --src /data/hyan671/yhproject/FakingRecipe/lenslang/outputs/lenslang_FakeSV_no_keyChts_3624only \
  --dst /data/hyan671/yhproject/FakingRecipe/lenslang/outputs/lenslang_FakeSV_no_keyChts_3624only_ENG
"""

import os
import json
import argparse
from typing import Any, Dict

# ========= 1. 中 -> 英 的枚举映射（值） =========
CN_TO_EN_VALUE = {
    # lighting quality
    "高对比": "high_contrast",
    "中等对比": "medium_contrast",
    "柔和低对比": "soft_low_contrast",

    # lighting condition
    "晴朗/高亮": "bright_sunny",
    "阴天/中性": "overcast_neutral",
    "昏暗/弱光": "dim_low_light",

    # color tone / temperature
    "冷色调": "cool_tone",
    "中性": "neutral_tone",
    "暖色调": "warm_tone",

    # shot types
    "远景": "long_shot",
    "中景": "medium_shot",
    "近景": "close_shot",
    "特写": "close_up",
    "极近特写": "extreme_close_up",

    # camera movements (also appear in movement_counts keys)
    "固定镜头": "static_shot",
    "横移（横摇）": "pan_horizontal",
    "纵移（俯仰）": "tilt_vertical",
    "推镜头": "dolly_in",
    "拉镜头": "dolly_out",

    # composition styles
    "对称构图": "symmetrical_composition",
    "三分法": "rule_of_thirds",
    "中心构图": "centered_composition",
    "留白": "negative_space",
    "低角度透视": "low_angle_perspective",
    "高角度俯视": "high_angle_perspective",
    "深焦透视": "deep_focus_perspective",
    "浅景深": "shallow_depth_of_field",

    # lighting style – extra
    "背光": "backlight",
    "侧光": "side_light",
    "顺光": "front_light",
    "自然光": "natural_light",
    "人造光": "artificial_light",

    # overall style / mood
    "纪实": "documentary_style",
    "叙事": "narrative_style",
    "新闻": "news_style",
    "vlog": "vlog_style",
    "商业广告": "commercial_advertising",
    "电影感": "cinematic_style",
    "紧张": "tense_mood",
    "温暖": "warm_mood",
    "肃穆": "solemn_mood",
    "活力": "energetic_mood",

    # style_tags 里的节奏枚举（原来是 "very fast" 等）
    "very fast": "very_fast",
    "fast": "fast",
    "moderate": "moderate",
    "slow": "slow",
}

# ========= 2. movement_counts 的 key 映射（键名） =========
MOVEMENT_KEY_MAP = {
    "固定镜头": "static_shot",
    "横移（横摇）": "pan_horizontal",
    "纵移（俯仰）": "tilt_vertical",
    "推镜头": "dolly_in",
    "拉镜头": "dolly_out",
}


def translate_string(s: str) -> str:
    """如果字符串在中英对照表中，则返回对应英文；否则原样返回。"""
    s_strip = s.strip()
    return CN_TO_EN_VALUE.get(s_strip, s_strip)


def translate_obj(obj: Any, parent_key: str = "") -> Any:
    """
    递归地遍历 JSON 对象：
    - 所有字符串：用 CN_TO_EN_VALUE 做一次映射
    - 所有列表：对元素递归处理（并在 style_tags 上做去重）
    - 所有字典：对值递归处理，如果是 movement_counts，则重命名内部 key
    """
    # 字符串：直接映射
    if isinstance(obj, str):
        return translate_string(obj)

    # 列表：逐元素翻译
    if isinstance(obj, list):
        translated_list = [translate_obj(v, parent_key=parent_key) for v in obj]

        # 如果是 style_tags，顺带做一下去重（保留顺序）
        if parent_key == "style_tags":
            dedup_list = []
            seen = set()
            for item in translated_list:
                if isinstance(item, str):
                    key = item
                else:
                    key = repr(item)
                if key not in seen:
                    seen.add(key)
                    dedup_list.append(item)
            return dedup_list

        return translated_list

    # 字典：递归处理
    if isinstance(obj, dict):
        new_dict: Dict[str, Any] = {}

        for k, v in obj.items():
            # movement_counts 内部 key 重命名
            if parent_key == "movement_counts":
                new_key = MOVEMENT_KEY_MAP.get(k, k)
            else:
                new_key = k

            # 对值递归处理，传入当前 key 作为 parent_key
            new_value = translate_obj(v, parent_key=new_key)
            new_dict[new_key] = new_value

        return new_dict

    # 其他类型（数字、布尔、None）原样返回
    return obj


def process_file(src_path: str, dst_path: str) -> bool:
    """处理单个 JSON 文件。成功返回 True，失败返回 False。"""
    try:
        with open(src_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        translated = translate_obj(data)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        with open(dst_path, "w", encoding="utf-8") as f:
            json.dump(translated, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        print(f"[ERROR] Failed to process {src_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=str,
        required=False,
        default="/data/hyan671/yhproject/FakingRecipe/lenslang/outputs/lenslang_FakeSV_no_keyChts_3624only",
        help="源 JSON 目录（包含中文标签）",
    )
    parser.add_argument(
        "--dst",
        type=str,
        required=False,
        default="/data/hyan671/yhproject/FakingRecipe/lenslang/outputs/lenslang_FakeSV_no_keyChts_3624only_ENG",
        help="输出 JSON 目录（英文标签）",
    )
    args = parser.parse_args()

    src_dir = args.src
    dst_dir = args.dst

    if not os.path.isdir(src_dir):
        raise ValueError(f"源目录不存在: {src_dir}")

    os.makedirs(dst_dir, exist_ok=True)

    files = [f for f in os.listdir(src_dir) if f.endswith(".json")]
    total = len(files)
    ok = 0

    print(f"Found {total} JSON files in {src_dir}")

    for fname in files:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)

        if process_file(src_path, dst_path):
            ok += 1

    print(f"Done. Success: {ok}/{total}, Failed: {total - ok}")


if __name__ == "__main__":
    main()