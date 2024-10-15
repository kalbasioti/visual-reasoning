import ipdb
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from basic_model import BasicModel
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()

    def forward(self, x):
        return x


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, nc, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, ceil_mode=False
        )  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LSA(nn.Module):
    def __init__(self, dim, heads=3, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head ** -0.5)))

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()

        mask = torch.eye(dots.shape[-1], device=dots.device, dtype=torch.bool)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            LSA(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SAViRT(BasicModel):
    def __init__(self, args):
        super(SAViRT, self).__init__(args)

        self.dim_1 = 49
        self.dim_2_v0 = 2048
        self.dim_2 = 512
        num_patches = 49
        dim = self.dim_2
        depth, heads = 1, 5
        dim_head, mlp_dim = 64, 2*self.dim_2
        dropout = 0.1

        self.linear = nn.Linear(self.dim_2_v0, self.dim_2)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.dataset = args.dataset

        self.img_size = args.img_size

        gate_function_1 = [
            nn.Linear(3 * self.dim_2, 3 * self.dim_2),
            nn.ReLU(),
            nn.Linear(3 * self.dim_2, self.dim_2),
            nn.ReLU(),
        ]
        gate_function_2 = [
            nn.Linear(4 * self.dim_2, 4 * self.dim_2),
            nn.ReLU(),
            nn.Linear(4 * self.dim_2, self.dim_2),
            nn.ReLU(),
        ]
        gate_function_3 = [
            nn.Linear(2*self.dim_2, 2*self.dim_2),
            nn.ReLU(),
            nn.Linear(2*self.dim_2, self.dim_2),
            nn.ReLU(),
            nn.Linear(self.dim_2, self.dim_2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.dim_2, self.dim_2),
        ]

        self.h1 = nn.Sequential(*gate_function_1)
        self.h2 = nn.Sequential(*gate_function_2)
        self.h3 = nn.Sequential(*gate_function_3)

        self.a = nn.Parameter(torch.ones([1])/2)

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=1e-4,
            eps=args.epsilon,
        )
        self.meta_beta = args.meta_beta
        self.row_to_column_idx = [0, 3, 6, 1, 4, 7, 2, 5]
        self.use_cell = args.use_cell
        self.use_ind = args.use_ind
        self.use_eco = args.use_eco

    def compute_loss(self, output, target):
        pred = output
        target_loss = F.cross_entropy(pred, target)

        loss = target_loss
        return loss

    def compute_loss_VPROM(self, output, target):
        pred, meta_target_pred, = (output[0], output[1])
        target_loss = F.cross_entropy(pred, target)

        loss = target_loss
        return loss

    def get_pair(self, row1, row2):
        try:
            x = torch.cat((row1, row2), dim=2)
            y = torch.cat((row2, row1), dim=2)
            z = torch.stack((x, y), dim=2)
        except Exception as e:
            print(e)
        return z

    def forward(self, x):
        B = x.size(0)

        ###########################################

        x = x.permute(0,1,2,4,5,3)
        x = x.reshape(-1,self.dim_2_v0)
        x = self.linear(x)
        x = x.reshape(B, 16, 1, 7, 7, self.dim_2)
        x = x.permute(0,1,2,5,3,4).contiguous()
        attnd_panel_features = (
            self.transformer(x.view(-1,self.dim_2,self.dim_1).permute(0,2,1)).view(B, -1, self.dim_1, self.dim_2).permute(0, 2, 1, 3)
        ) * self.use_cell

        ###########################################

        panel_features = x.view(B, -1, self.dim_1, self.dim_2).permute(0, 2, 1, 3)

        row3_12features = (
            panel_features[:, :, 6:8, :].unsqueeze(2).repeat(1, 1, 8, 1, 1)
        )
        candidate_features = panel_features[:, :, 8:16, :].unsqueeze(3)
        row3_features = torch.cat((row3_12features, candidate_features), dim=3)
        row_features = [
            panel_features[:, :, 0:3, :].unsqueeze(2),
            panel_features[:, :, 3:6, :].unsqueeze(2),
            row3_features,
        ]
        row_features = torch.cat(row_features, dim=2).view(-1, 3, self.dim_2)
        attnd_row_features = self.transformer(row_features).sum(-2)
        attnd_row_features = attnd_row_features.view(B, self.dim_1, 10, self.dim_2) * self.use_ind

        ###########################################

        choice_rows = torch.cat(
            (
                panel_features[:, :, 6:8, :].unsqueeze(2).repeat(1, 1, 8, 1, 1),
                panel_features[:, :, 8:16, :].unsqueeze(3),
            ),
            dim=3,
        )

        pairwise_row_list = (
            [self.get_pair(panel_features[:, :, 0:3, :], panel_features[:, :, 3:6, :])]
            + [
                self.get_pair(panel_features[:, :, 0:3, :], choice_rows[:, :, i, :])
                for i in range(8)
            ]
            + [
                self.get_pair(panel_features[:, :, 3:6, :], choice_rows[:, :, i, :])
                for i in range(8)
            ]
        )
        pairwise_row_features = torch.stack(pairwise_row_list, dim=2).view(-1, 6, self.dim_2)
        attnd_pairwise_row_features = self.transformer(pairwise_row_features).sum(-2)
        attnd_pairwise_row_features = attnd_pairwise_row_features.view(
            B, self.dim_1, 17, 2, self.dim_2
        ).sum(-2) * self.use_eco

        #################################

        attnd_row3_12features = (
            attnd_panel_features[:, :, 6:8, :].unsqueeze(2).repeat(1, 1, 8, 1, 1)
        )
        attnd_candidate_features = attnd_panel_features[:, :, 8:16, :].unsqueeze(3)
        attnd_row3_features = torch.cat(
            (attnd_row3_12features, attnd_candidate_features), dim=3
        )

        intra_row_features = [
            attnd_panel_features[:, :, 0:3, :].unsqueeze(2),
            attnd_panel_features[:, :, 3:6, :].unsqueeze(2),
            attnd_row3_features,
        ]
        intra_row_features = torch.cat(intra_row_features, dim=2).view(-1, 3, self.dim_2)

        row_relations = self.h1(intra_row_features.view(-1, 3*self.dim_2))
        row_relations = torch.cat(
            (row_relations, attnd_row_features.view(-1, self.dim_2)), dim=1
        )

        row_relations = row_relations.view(B, self.dim_1, 10, 2*self.dim_2)

        inter_row_list = (
            [self.get_pair(row_relations[:, :, 0, :], row_relations[:, :, 1, :])]
            + [
                self.get_pair(row_relations[:, :, 0, :], row_relations[:, :, i, :])
                for i in range(2, 10)
            ]
            + [
                self.get_pair(row_relations[:, :, 1, :], row_relations[:, :, i, :])
                for i in range(2, 10)
            ]
        )

        inter_row_list = torch.stack(inter_row_list, dim=2)

        rules = self.h2(inter_row_list.view(-1, 4*self.dim_2))
        rules = torch.sum(rules.view(B, self.dim_1, 17, 2, self.dim_2), dim=-2)

        rules = torch.cat((rules, attnd_pairwise_row_features), dim=-1)

        rules = self.h3(rules)

        rules = rules.mean(1)

        dominant_rule = rules[:, 0, :].unsqueeze(1)
        pseudo_rules = rules[:, 1:, :]

        similarity = torch.bmm(
            dominant_rule, torch.transpose(pseudo_rules, 1, 2)
        ).squeeze(1)

        alpha = F.sigmoid(self.a)
        similarity = (1-alpha)*similarity[:, :8] + alpha*similarity[:, 8:]

        return similarity