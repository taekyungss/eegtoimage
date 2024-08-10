import torch.nn as nn
import torch.nn.functional as F
import config
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
import numpy as np
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import math
# from common_spatial_pattern import csp
import torch

np.random.seed(45)
torch.manual_seed(45)


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (22, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


# class ClassificationHead(nn.Sequential):
#     def __init__(self, emb_size, n_classes):
#         super().__init__()
        
#         # global average pooling
#         self.clshead = nn.Sequential(
#             Reduce('b n e -> b e', reduction='mean'),
#             nn.LayerNorm(emb_size),
#             nn.Linear(emb_size, n_classes)
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(2440, 256),
#             nn.ELU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 32),
#             nn.ELU(),
#             nn.Dropout(0.3),
#             nn.Linear(32, 4)
#         )

    # def forward(self, x):
    #     x = x.contiguous().view(x.size(0), -1)
    #     out = self.fc(x)
    #     return x, out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_channel=128, **kwargs):
        super().__init__()
        self.patch_embedding = PatchEmbedding(emb_size)
        self.transofmer_encoder = TransformerEncoder(depth, emb_size)
        self.fc = nn.Linear(emb_size,n_channel)

    def forward(self,x):
        x = self.patch_embedding(x)
        x = self.transofmer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

# def interaug(timg, label):  
#     aug_data = []
#     aug_label = []
#     for cls4aug in range(4):
#         cls_idx = np.where(label.cpu().numpy() == cls4aug + 1)
#         tmp_data = timg[cls_idx]
#         tmp_label = label[cls_idx]

#         tmp_aug_data = np.zeros((int(config.batch_size / 4), 1, 22, 1000))
#         for ri in range(int(config.batch_size / 4)):
#             for rj in range(8):
#                 rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
#                 tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
#                                                                     rj * 125:(rj + 1) * 125]

#         aug_data.append(tmp_aug_data)
#         aug_label.append(tmp_label[:int(config.batch_size / 4)])
#     aug_data = np.concatenate(aug_data)
#     aug_label = np.concatenate(aug_label)
#     aug_shuffle = np.random.permutation(len(aug_data))
#     aug_data = aug_data[aug_shuffle, :, :]
#     aug_label = aug_label[aug_shuffle]

#     aug_data = torch.from_numpy(aug_data).cuda()
#     aug_data = aug_data.float()
#     aug_label = torch.from_numpy(aug_label-1).cuda()
#     aug_label = aug_label.long()
#     return aug_data, aug_label

# def interaug(timg, label):
#     aug_data = []
#     aug_label = []
#     for cls4aug in range(4):
#         cls_idx = np.where(label.cpu().numpy() == cls4aug + 1)
#         tmp_data = timg[cls_idx].cpu().numpy()
#         tmp_label = label[cls_idx]

#         tmp_aug_data = np.zeros((int(config.batch_size / 4), tmp_data.shape[1], tmp_data.shape[2]))  # Shape (batch_size/4, 440, 128)
#         for ri in range(int(config.batch_size / 4)):
#             for rj in range(8):
#                 rand_idx = np.random.randint(0, tmp_data.shape[0], 1)  # 1개의 랜덤 인덱스
#         idx = rand_idx[0] % tmp_data.shape[0]  # 반복 인덱스 사용
#         tmp_aug_data[ri, :, rj * 125:(rj + 1) * 125] = tmp_data[idx, :, rj * 125:(rj + 1) * 125]
#         aug_data.append(tmp_aug_data)
#         aug_label.append(tmp_label[:int(config.batch_size / 4)].cpu().numpy())
        
#     aug_data = np.concatenate(aug_data)
#     aug_label = np.concatenate(aug_label)
#     aug_shuffle = np.random.permutation(len(aug_data))
#     aug_data = aug_data[aug_shuffle, :]
#     aug_label = aug_label[aug_shuffle]

#     aug_data = torch.from_numpy(aug_data).cuda().float()
#     aug_label = torch.from_numpy(aug_label-1).cuda().long()
#     return aug_data, aug_label

    
class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim, bias=False)
        # self.gelu = nn.GELU()
        # self.fc = nn.Linear(projection_dim, projection_dim)
        # self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        # x = self.gelu(projected)
        # x = self.fc(x)
        # x = self.dropout(x)
        # x += projected
        # return self.layer_norm(x)
        return projected