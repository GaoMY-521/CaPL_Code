import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import build_dataset
from datasets.utils import build_data_loader
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils import *
from torch.autograd import Variable
from torch.distributions import uniform

class Query(nn.Module):
    def __init__(self, dim, K, hard_prompt):
        super(Query, self).__init__()
        self.query = hard_prompt if hard_prompt is not None else torch.randn(K, dim)
        self.query = nn.Parameter(self.query, requires_grad=True)
        self.dim = dim
        self.K = K
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.scale = 2 ** -0.5
        self.apply(weights_init)

    def forward(self, attribute):
        b = attribute.size()[0]
        q = self.to_q(self.query)
        k, v = self.to_k(attribute), self.to_v(attribute)
        k = k.unsqueeze(1).repeat(1, b, 1)
        sim = torch.einsum('a d, n b d -> n a b', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('n a b, b d -> n a d', attn, v)
        out = self.out(out)
        return out

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim=512, num_tokens=10):
        super().__init__()
        self.attn_fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1)
        )

    def forward(self, x):
        attn_scores = self.attn_fc(x)
        attn_weights = F.softmax(attn_scores, dim=0)

        fused = (attn_weights * x).sum(dim=0)
        return fused

class Granule(nn.Module):
    def __init__(self, hard_prompt):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )
        self.query = Query(hard_prompt)
        self.query_ess = Query(None)
        self.fuse = AttentionFusion()
        self.apply(weights_init)

    def forward(self, nonindi, indi):
        b, d = nonindi.size()[0], nonindi.size()[1]
        device = nonindi.device
        indi_value = self.query(indi)
        nonindi_value = self.query_ess(nonindi)
        a = indi_value.size()[1]

        factual_indi = torch.cat([nonindi.unsqueeze(1).expand(-1, a, -1), indi_value], dim=-1)
        counter_swap = torch.cat([nonindi.unsqueeze(1).expand(-1, b, -1), indi.unsqueeze(0).expand(b, -1, -1)], dim=-1)
        factual_indi = self.net(factual_indi.view(-1, 2 * d)).view(b, a, -1)
        counter_swap = self.net(counter_swap.view(-1, 2 * d)).view(b, b, -1)

        return factual_indi, counter_swap



