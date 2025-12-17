import torch
import torch.nn as nn
import torch.nn.functional as F
from .trm import *
import pandas as pd
import json
from .attention import *
import numpy as np   # PosEncoding_fix 里要用到

# =========================
# MSAM：内容分支（保持不变）
# =========================
class MSAM(torch.nn.Module):
    """
    MSAM: Material Selection-Aware Modeling
    负责 “素材选择” 分支（文本情绪 + 文本语义 + 视觉语义 + 音频情绪）。
    M3 配置下，这部分完全不改。
    """
    def __init__(self, dataset):
        super(MSAM, self).__init__()
        if dataset == 'fakett':
            self.encoded_text_semantic_fea_dim = 512
        elif dataset == 'fakesv':
            self.encoded_text_semantic_fea_dim = 768
        self.input_visual_frames = 83

        # 文本情绪特征 -> 128
        self.mlp_text_emo = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # 文本语义特征 -> 128
        self.mlp_text_semantic = nn.Sequential(
            nn.Linear(self.encoded_text_semantic_fea_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # 图像特征 -> 128
        self.mlp_img = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # 音频情绪特征 -> 128
        self.mlp_audio = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # text–visual 共注意力（内容语义对齐）
        self.co_attention_tv = co_attention(
            d_k=128, d_v=128, n_heads=4, dropout=0.1, d_model=128,
            visual_len=self.input_visual_frames, sen_len=512,
            fea_v=128, fea_s=128, pos=False
        )

        # 情绪与语义各自的 transformer encoder
        self.trm_emo = nn.TransformerEncoderLayer(
            d_model=128, nhead=2, batch_first=True
        )
        self.trm_semantic = nn.TransformerEncoderLayer(
            d_model=128, nhead=2, batch_first=True
        )

        # 最终内容分支分类器：concat(emo, semantic) -> 2 类 logits
        self.content_classifier = nn.Sequential(
            nn.Linear(128 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2)
        )

    def forward(self, **kwargs):
        all_phrase_semantic_fea = kwargs['all_phrase_semantic_fea']
        all_phrase_emo_fea = kwargs['all_phrase_emo_fea']
        raw_visual_frames = kwargs['raw_visual_frames']
        raw_audio_emo = kwargs['raw_audio_emo']

        # ---- emo branch ----
        raw_t_fea_emo = self.mlp_text_emo(all_phrase_emo_fea).unsqueeze(1)  # (B,1,128)
        raw_a_fea_emo = self.mlp_audio(raw_audio_emo).unsqueeze(1)          # (B,1,128)
        fusion_emo_fea = self.trm_emo(torch.cat((raw_t_fea_emo, raw_a_fea_emo), dim=1))
        fusion_emo_fea = torch.mean(fusion_emo_fea, dim=1)                  # (B,128)

        # ---- semantic branch ----
        raw_t_fea_semantic = self.mlp_text_semantic(all_phrase_semantic_fea)   # (B,L_t,128)
        raw_v_fea = self.mlp_img(raw_visual_frames)                            # (B,L_v,128)
        content_v, content_t = self.co_attention_tv(
            v=raw_v_fea,
            s=raw_t_fea_semantic,
            v_len=raw_v_fea.shape[1],
            s_len=raw_t_fea_semantic.shape[1]
        )
        content_v = torch.mean(content_v, dim=-2)  # (B,128)
        content_t = torch.mean(content_t, dim=-2)  # (B,128)

        fusion_semantic_fea = self.trm_semantic(
            torch.cat((content_t.unsqueeze(1), content_v.unsqueeze(1)), dim=1)
        )
        fusion_semantic_fea = torch.mean(fusion_semantic_fea, dim=1)  # (B,128)

        # ---- fuse emo & semantic ----
        msam_fea = torch.cat((fusion_emo_fea, fusion_semantic_fea), dim=1)  # (B,256)
        output_msam = self.content_classifier(msam_fea)                     # (B,2)
        return output_msam


# =====================================================
# 原有的 LayerNorm2d / PosEncoding_fix / DurationEncoding
# get_dura_info_visual / MEAM 保留在文件里，
# 但在 M3 配置的 FakingRecipe_Model 中不再实例化使用。
# （如想彻底去掉，可以移动到其他文件或注释掉。）
# =====================================================

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class PosEncoding_fix(nn.Module):
    def __init__(self, d_word_vec):
        super(PosEncoding_fix, self).__init__()
        self.d_word_vec = d_word_vec
        self.w_k = np.array([
            1 / (np.power(10000, 2 * (i // 2) / d_word_vec))
            for i in range(d_word_vec)
        ])

    def forward(self, inputs):
        pos_embs = []
        for pos in inputs:
            pos_emb = torch.tensor([self.w_k[i] * pos.cpu() for i in range(self.d_word_vec)])
            if pos != 0:
                pos_emb[0::2] = np.sin(pos_emb[0::2])
                pos_emb[1::2] = np.cos(pos_emb[1::2])
                pos_embs.append(pos_emb)
            else:
                pos_embs.append(torch.zeros(self.d_word_vec))
        pos_embs = torch.stack(pos_embs)
        return pos_embs.cuda()

class DurationEncoding(nn.Module):
    def __init__(self, dim, dataset):
        super(DurationEncoding, self).__init__()
        if dataset == 'fakett':
            with open('./fea/fakett/fakett_segment_duration.json', 'r') as json_file:
                seg_dura_info = json.load(json_file)
        elif dataset == 'fakesv':
            with open('./fea/fakesv/fakesv_segment_duration.json', 'r') as json_file:
                seg_dura_info = json.load(json_file)

        self.all_seg_duration = seg_dura_info['all_seg_duration']
        self.all_seg_dura_ratio = seg_dura_info['all_seg_dura_ratio']
        self.absolute_bin_edges = torch.quantile(
            torch.tensor(self.all_seg_duration).to(torch.float64),
            torch.arange(0, 1.01, 0.01).to(torch.float64)
        ).cuda()
        self.relative_bin_edges = torch.quantile(
            torch.tensor(self.all_seg_dura_ratio).to(torch.float64),
            torch.arange(0, 1.01, 0.02).to(torch.float64)
        ).cuda()
        self.ab_duration_embed = torch.nn.Embedding(101, dim)
        self.re_duration_embed = torch.nn.Embedding(51, dim)

        self.ocr_all_seg_duration = seg_dura_info['ocr_all_seg_duration']
        self.ocr_all_seg_dura_ratio = seg_dura_info['ocr_all_seg_dura_ratio']
        self.ocr_absolute_bin_edges = torch.quantile(
            torch.tensor(self.ocr_all_seg_duration).to(torch.float64),
            torch.arange(0, 1.01, 0.01).to(torch.float64)
        ).cuda()
        self.ocr_relative_bin_edges = torch.quantile(
            torch.tensor(self.ocr_all_seg_dura_ratio).to(torch.float64),
            torch.arange(0, 1.01, 0.02).to(torch.float64)
        ).cuda()
        self.ocr_ab_duration_embed = torch.nn.Embedding(101, dim)
        self.ocr_re_duration_embed = torch.nn.Embedding(51, dim)

        self.result_dim = dim

    def forward(self, time_value, attribute):
        all_segs_embedding = []
        if attribute == 'natural_ab':
            for dv in time_value:
                bucket_indice = torch.searchsorted(
                    self.absolute_bin_edges,
                    torch.tensor(dv, dtype=torch.float64)
                )
                dura_embedding = self.ab_duration_embed(bucket_indice)
                all_segs_embedding.append(dura_embedding)
        elif attribute == 'natural_re':
            for dv in time_value:
                bucket_indice = torch.searchsorted(
                    self.relative_bin_edges,
                    torch.tensor(dv, dtype=torch.float64)
                )
                dura_embedding = self.re_duration_embed(bucket_indice)
                all_segs_embedding.append(dura_embedding)
        elif attribute == 'ocr_ab':
            for dv in time_value:
                bucket_indice = torch.searchsorted(
                    self.ocr_absolute_bin_edges,
                    torch.tensor(dv, dtype=torch.float64)
                )
                dura_embedding = self.ocr_ab_duration_embed(bucket_indice)
                all_segs_embedding.append(dura_embedding)
        elif attribute == 'ocr_re':
            for dv in time_value:
                bucket_indice = torch.searchsorted(
                    self.ocr_relative_bin_edges,
                    torch.tensor(dv, dtype=torch.float64)
                )
                dura_embedding = self.ocr_re_duration_embed(bucket_indice)
                all_segs_embedding.append(dura_embedding)

        if len(all_segs_embedding) == 0:
            return torch.zeros((1, self.result_dim)).cuda()
        return torch.stack(all_segs_embedding, dim=0).cuda()

def get_dura_info_visual(segs, fps, total_frame):
    duration_frames = []
    duration_time = []
    for seg in segs:
        if seg[0] == -1 and seg[1] == -1:
            continue
        if seg[0] == 0 and seg[1] == 0:
            continue
        else:
            duration_frames.append(seg[1] - seg[0] + 1)
            duration_time.append((seg[1] - seg[0] + 1) / fps)
    duration_ratio = [min(dura / total_frame, 1) for dura in duration_frames]
    return torch.tensor(duration_time).cuda(), torch.tensor(duration_ratio).cuda()


class MEAM(torch.nn.Module):
    """
    原始 MEAM：Material Editing-Aware Modeling
    M3 配置下不再作为 editing 分支使用，可以保留以便以后对比 M1/M2。
    """
    def __init__(self, dataset):
        super(MEAM, self).__init__()
        # ...（原实现保持不变，这里省略，与之前相同）...
        # 你可以保留原始代码，也可以移动到单独文件中
        raise NotImplementedError("MEAM is not used in M3 (No-Edit + LensLang) configuration.")


# ==========================================
# 新增：LensLang 仅编辑分支（用于 M3）
# ==========================================
class LensLangBranch(nn.Module):
    """
    LensLangBranch: 用 LensLang 向量单独代表 Editing 分支。

    输入：
        lenslang_fea: 形状 (B, lenslang_dim)，例如 dim=10 的瘦身版 LensLang 向量。
    输出：
        logits: (B, 2)，表示编辑分支的真假 logits (Y_E_lenslang)

    说明：
        - 这是 M3 配置中的 editing branch；
        - 不再依赖原来的 MEAM 时长 / OCR / layout / HTSE 等复杂结构；
        - 只用 lenslang_fea -> MLP -> 2 类输出。
    """
    def __init__(self, lenslang_dim: int = 10, hidden_dim: int = 64):
        super(LensLangBranch, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(lenslang_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, **kwargs):
        # 假定 dataloader 在 kwargs 里提供 lenslang_fea: (B, lenslang_dim)
        lenslang_fea = kwargs['lenslang_fea']   # (B, lenslang_dim)
        return self.mlp(lenslang_fea)           # (B, 2)


# ==========================================
# FakingRecipe_Model：M3 版本（No-Edit + LensLang）
# ==========================================
class FakingRecipe_Model(torch.nn.Module):
    """
    M3 配置：
        - 内容分支：仍然使用 MSAM（Text + Visual + Audio）。
        - 编辑分支：不再使用原 MEAM，而是只用 LensLangBranch。
        - 融合方式保持原 FakingRecipe：Y = Y_S * tanh(Y_E).

    注意：
        - 请确保 dataloader 在每个 sample 的 kwargs 中提供 'lenslang_fea'。
    """
    def __init__(self, dataset, lenslang_dim: int = 15):
        super(FakingRecipe_Model, self).__init__()
        self.content_branch = MSAM(dataset=dataset)
        # 原来是 MEAM(dataset=dataset)
        self.editing_branch = LensLangBranch(lenslang_dim=lenslang_dim)
        self.tanh = nn.Tanh()

    def forward(self, **kwargs):
        # MSAM：素材选择分支
        output_msam = self.content_branch(**kwargs)      # (B,2)

        # LensLang 仅编辑分支
        output_lens = self.editing_branch(**kwargs)      # (B,2)

        # Fusing as original: Y = Y_S * tanh(Y_E)
        output = output_msam * self.tanh(output_lens)
        return output, output_msam, output_lens