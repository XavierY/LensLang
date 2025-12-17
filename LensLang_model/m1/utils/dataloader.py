# utils/dataloader.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle


# ---------- 小工具：兼容旧版 torch.load / numpy pkl ----------
def safe_torch_load(path):
    """
    兼容旧的用 numpy / pickle 保存的特征文件。
    只在你自己生成的特征上使用（可信源）。
    """
    return torch.load(path, map_location="cpu", weights_only=False)


class FakingRecipe_Dataset(Dataset):
    def __init__(self, vid_path, dataset):
        """
        vid_path: 训练/验证/测试 split 文件路径（vid_time3_*.txt）
        dataset: 'fakesv' 或 'fakett'
        """
        self.dataset = dataset

        # 读取 split vid 列表
        self.vid = []
        with open(vid_path, "r") as fr:
            for line in fr.readlines():
                v = line.strip()
                if v:
                    self.vid.append(v)

        # 读取 metainfo & 各模态特征路径
        if dataset == 'fakesv':
            self.data_all = pd.read_json(
                './fea/fakesv/metainfo.json',
                orient='records',
                dtype=False,
                lines=True
            )
            self.ocr_pattern_fea_path = './fea/fakesv/preprocess_ocr/sam'
            self.ocr_phrase_fea_path = './fea/fakesv/preprocess_ocr/ocr_phrase_fea.pkl'
            self.text_semantic_fea_path = './fea/fakesv/preprocess_text/sem_text_fea.pkl'
            self.text_emo_fea_path = './fea/fakesv/preprocess_text/emo_text_fea.pkl'
            self.audio_fea_path = './fea/fakesv/preprocess_audio'
            self.visual_fea_path = './fea/fakesv/preprocess_visual'

        elif dataset == 'fakett':
            self.data_all = pd.read_json(
                './fea/fakett/metainfo.json',
                orient='records',
                lines=True,
                dtype={'video_id': str}
            )
            self.ocr_pattern_fea_path = './fea/fakett/preprocess_ocr/sam'
            self.ocr_phrase_fea_path = './fea/fakett/preprocess_ocr/ocr_phrase_fea.pkl'
            self.text_semantic_fea_path = './fea/fakett/preprocess_text/sem_text_fea.pkl'
            self.text_emo_fea_path = './fea/fakett/preprocess_text/emo_text_fea.pkl'
            self.audio_fea_path = './fea/fakett/preprocess_audio'
            self.visual_fea_path = './fea/fakett/preprocess_visual'
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        # 子集筛选
        self.data = self.data_all[self.data_all.video_id.isin(self.vid)]
        self.data.reset_index(inplace=True, drop=True)

        # 预先把大 pkl 都读进来（跟原始 dataloader 一致）
        with open(self.ocr_phrase_fea_path, 'rb') as f:
            self.ocr_phrase = torch.load(f)

        with open(self.text_semantic_fea_path, 'rb') as f:
            self.text_semantic_fea = torch.load(f)

        with open(self.text_emo_fea_path, 'rb') as f:
            self.text_emo_fea = torch.load(f)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # label: fake=1, real=0
        label = 1 if item['annotation'] == 'fake' else 0
        label = torch.tensor(label, dtype=torch.long)

        fps = torch.tensor(item['fps'], dtype=torch.float32)
        total_frame = torch.tensor(item['frame_count'], dtype=torch.float32)

        # transnetv2_segs 是 list[[s,e], ...]，这里保持为原生 list
        visual_time_region = item['transnetv2_segs']

        # 文本特征
        all_phrase_semantic_fea = self.text_semantic_fea['last_hidden_state'][vid]   # [L, 768 or 512]
        all_phrase_emo_fea = self.text_emo_fea['pooler_output'][vid]                 # [768]

        # visual 帧特征 [F, 512]
        v_fea_path = os.path.join(self.visual_fea_path, vid + '.pkl')
        raw_visual_frames = safe_torch_load(v_fea_path)
        if isinstance(raw_visual_frames, np.ndarray):
            raw_visual_frames = torch.from_numpy(raw_visual_frames)
        else:
            raw_visual_frames = torch.tensor(raw_visual_frames)

        # audio emo 特征 [1, 768]
        a_fea_path = os.path.join(self.audio_fea_path, vid + '.pkl')
        raw_audio_emo = safe_torch_load(a_fea_path)
        if isinstance(raw_audio_emo, np.ndarray):
            raw_audio_emo = torch.from_numpy(raw_audio_emo)
        # 保证形状 [1, 768]
        if raw_audio_emo.dim() == 1:
            raw_audio_emo = raw_audio_emo.unsqueeze(0)

        # OCR layout pattern: [256, H, W] → downscaling 在模型里做
        ocr_pattern_fea_file_path = os.path.join(self.ocr_pattern_fea_path, vid, 'r0.pkl')
        ocr_pattern_fea = safe_torch_load(ocr_pattern_fea_file_path)
        if isinstance(ocr_pattern_fea, np.ndarray):
            ocr_pattern_fea = torch.from_numpy(ocr_pattern_fea)
        else:
            ocr_pattern_fea = torch.tensor(ocr_pattern_fea)

        # OCR phrase 特征
        ocr_phrase_fea = self.ocr_phrase['ocr_phrase_fea'][vid]   # [N, 512]
        ocr_time_region = self.ocr_phrase['ocr_time_region'][vid] # [N, 2] tensor

        sample = {
            'vid': vid,
            'label': label,
            'fps': fps,
            'total_frame': total_frame,
            'all_phrase_semantic_fea': all_phrase_semantic_fea,
            'all_phrase_emo_fea': all_phrase_emo_fea,
            'raw_visual_frames': raw_visual_frames,
            'raw_audio_emo': raw_audio_emo,
            'ocr_pattern_fea': ocr_pattern_fea,
            'ocr_phrase_fea': ocr_phrase_fea,
            'ocr_time_region': ocr_time_region,
            'visual_time_region': visual_time_region,
        }

        return sample


# ----------------- 一些 pad 工具（保持与原版逻辑一致）-----------------

def pad_frame_sequence(seq_len, lst):
    attention_masks = []
    result = []
    for video in lst:
        video = torch.FloatTensor(video)
        ori_len = video.shape[0]
        if ori_len >= seq_len:
            gap = ori_len // seq_len
            video = video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video = torch.cat(
                (video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.float)),
                dim=0
            )
            mask = np.append(np.ones(ori_len), np.zeros(seq_len - ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)


def pad_frame_by_seg(seq_len, lst, seg):
    result = []
    seg_indicators = []
    sampled_seg = []
    for i in range(len(lst)):
        video = lst[i]
        v_sampled_seg = []
        video = torch.FloatTensor(video)
        ori_len = video.shape[0]
        seg_video = seg[i]
        seg_len = len(seg_video)
        if seg_len >= seq_len:
            gap = seg_len // seq_len
            seg_video = seg_video[::gap][:seq_len]
            sample_index = []
            sample_seg_indicator = []
            for j in range(len(seg_video)):
                v_sampled_seg.append(seg_video[j])
                if seg_video[j][0] == seg_video[j][1]:
                    sample_index.append(seg_video[j][0])
                else:
                    sample_index.append(np.random.randint(seg_video[j][0], seg_video[j][1]))
                sample_seg_indicator.append(j)
            video = video[sample_index]
            mask = sample_seg_indicator
        else:
            if ori_len < seq_len:
                video = torch.cat(
                    (video, torch.zeros([seq_len - ori_len, video.shape[1]], dtype=torch.float)),
                    dim=0
                )

                mask = []
                for j in range(len(seg_video)):
                    v_sampled_seg.append(seg_video[j])
                    mask.extend([j] * (seg_video[j][1] - seg_video[j][0] + 1))
                mask.extend([-1] * (seq_len - len(mask)))

            else:
                sample_index = []
                sample_seg_indicator = []
                seg_len_lst = [(x[1] - x[0]) + 1 for x in seg_video]
                sample_ratio = [seg_len_lst[i] / sum(seg_len_lst) for i in range(len(seg_len_lst))]
                sample_len = [seq_len * sample_ratio[i] for i in range(len(seg_len_lst))]
                sample_per_seg = [int(x) + 1 if x < 1 else int(x) for x in sample_len]

                sample_per_seg = [
                    x if x <= seg_len_lst[i] else seg_len_lst[i]
                    for i, x in enumerate(sample_per_seg)
                ]
                additional_sample = sum(sample_per_seg) - seq_len
                if additional_sample > 0:
                    idx = 0
                    while additional_sample > 0:
                        if idx == len(sample_per_seg):
                            idx = 0
                        if sample_per_seg[idx] > 1:
                            sample_per_seg[idx] = sample_per_seg[idx] - 1
                            additional_sample = additional_sample - 1
                        idx += 1

                elif additional_sample < 0:
                    idx = 0
                    while additional_sample < 0:
                        if idx == len(sample_per_seg):
                            idx = 0
                        if seg_len_lst[idx] - sample_per_seg[idx] >= 1:
                            sample_per_seg[idx] = sample_per_seg[idx] + 1
                            additional_sample = additional_sample + 1
                        idx += 1

                for seg_idx in range(len(sample_per_seg)):
                    sample_seg_indicator.extend([seg_idx] * sample_per_seg[seg_idx])

                for j in range(len(seg_video)):
                    v_sampled_seg.append(seg_video[j])
                    if sample_per_seg[j] == seg_len_lst[j]:
                        sample_index.extend(np.arange(seg_video[j][0], seg_video[j][1] + 1))
                    else:
                        sample_index.extend(
                            np.sort(
                                np.random.randint(seg_video[j][0], seg_video[j][1] + 1, sample_per_seg[j])
                            )
                        )

                sample_index = np.array(sample_index)
                sample_index = np.sort(sample_index)
                video = video[sample_index]
                batch_sample_seg_indicator = np.array(sample_seg_indicator)
                mask = batch_sample_seg_indicator
                v_sampled_seg.sort(key=lambda x: x[0])

        result.append(video)
        mask = torch.IntTensor(mask)
        sampled_seg.append(v_sampled_seg)
        seg_indicators.append(mask)
    return torch.stack(result), torch.stack(seg_indicators), sampled_seg


def pad_segment(seg_lst, target_len):
    for sl_idx in range(len(seg_lst)):
        for s_idx in range(len(seg_lst[sl_idx])):
            seg_lst[sl_idx][s_idx] = torch.tensor(seg_lst[sl_idx][s_idx])
        if len(seg_lst[sl_idx]) < target_len:
            seg_lst[sl_idx].extend(
                [torch.tensor([-1, -1])] * (target_len - len(seg_lst[sl_idx]))
            )
        else:
            seg_lst[sl_idx] = seg_lst[sl_idx][:target_len]
        seg_lst[sl_idx] = torch.stack(seg_lst[sl_idx])

    return torch.stack(seg_lst)


def pad_unnatural_phrase(phrase_lst, target_len):
    for pl_idx in range(len(phrase_lst)):
        if len(phrase_lst[pl_idx]) < target_len:
            phrase_lst[pl_idx] = torch.cat(
                (
                    phrase_lst[pl_idx],
                    torch.zeros(
                        [target_len - len(phrase_lst[pl_idx]),
                         phrase_lst[pl_idx].shape[1]],
                        dtype=torch.long
                    )
                ),
                dim=0
            )
        else:
            phrase_lst[pl_idx] = phrase_lst[pl_idx][:target_len]
    return torch.stack(phrase_lst)


# ----------------- collate_fn -----------------

def collate_fn_FakeingRecipe(batch):
    num_visual_frames = 83
    num_segs = 83
    num_phrase = 80

    vid = [item['vid'] for item in batch]
    label = torch.stack([item['label'] for item in batch])

    fps = torch.stack([item['fps'] for item in batch])
    total_frame = torch.stack([item['total_frame'] for item in batch])

    # 文本语义特征，pad 到 512
    all_phrase_semantic_fea_list = [item['all_phrase_semantic_fea'] for item in batch]
    padded_sem_list = []
    for x in all_phrase_semantic_fea_list:
        x = torch.tensor(x, dtype=torch.float32)
        L, D = x.shape
        if L < 512:
            pad = torch.zeros([512 - L, D], dtype=torch.float32)
            x = torch.cat([x, pad], dim=0)
        else:
            x = x[:512]
        padded_sem_list.append(x)
    all_phrase_semantic_fea = torch.stack(padded_sem_list)  # [B, 512, dim]

    # 文本情绪特征 [B, 768]
    all_phrase_emo_fea = torch.stack(
        [torch.tensor(item['all_phrase_emo_fea'], dtype=torch.float32) for item in batch]
    )

    # visual 帧特征
    raw_visual_frames_list = [item['raw_visual_frames'] for item in batch]
    content_visual_frames, _ = pad_frame_sequence(num_visual_frames, raw_visual_frames_list)

    # audio emo 特征 [B, 768]
    raw_audio_emo = torch.cat(
        [item['raw_audio_emo'] for item in batch],
        dim=0
    )

    # OCR pattern
    ocr_pattern_fea = torch.stack([item['ocr_pattern_fea'] for item in batch])

    # OCR phrase
    ocr_phrase_fea_list = [item['ocr_phrase_fea'] for item in batch]
    ocr_time_region_list = [item['ocr_time_region'] for item in batch]

    # visual seg
    visual_time_region_list = [item['visual_time_region'] for item in batch]

    visual_frames_fea, visual_frames_seg_indicator, sampled_seg = pad_frame_by_seg(
        num_visual_frames,
        raw_visual_frames_list,
        visual_time_region_list
    )
    visual_seg_paded = pad_segment(sampled_seg, num_segs)

    ocr_phrase_fea = pad_unnatural_phrase(ocr_phrase_fea_list, num_phrase)
    ocr_time_region = pad_unnatural_phrase(ocr_time_region_list, num_phrase)

    return {
        'vid': vid,
        'label': label,
        'fps': fps,
        'total_frame': total_frame,
        'all_phrase_semantic_fea': all_phrase_semantic_fea,
        'all_phrase_emo_fea': all_phrase_emo_fea,
        'raw_visual_frames': content_visual_frames,
        'raw_audio_emo': raw_audio_emo,
        'ocr_pattern_fea': ocr_pattern_fea,
        'ocr_phrase_fea': ocr_phrase_fea,
        'ocr_time_region': ocr_time_region,
        'visual_frames_fea': visual_frames_fea,
        'visual_frames_seg_indicator': visual_frames_seg_indicator,
        'visual_seg_paded': visual_seg_paded
    }