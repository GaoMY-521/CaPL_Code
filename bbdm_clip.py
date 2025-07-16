import math
from random import random
from functools import partial
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm

import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

import clip

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def linear_generate_mt(T):
    m_min, m_max = 0.001, 0.999
    m_t = np.linspace(m_min, m_max, T)
    return m_t

def cosine_generate_mt(T):
    m_t = 1.0075 ** np.linspace(0, T, T)
    m_t = m_t / m_t[-1]
    m_t[-1] = 0.999
    return m_t

def extract(a, t, x_shape):
    b, *_ = t.shape
    device = t.device
    # out = a.gather(-1, t)
    time = F.one_hot(t.to(torch.int64), 1000).float().to(device)
    time.requires_grad = True
    out = a * time
    out = torch.mm(out, out.T).sqrt()
    out_ = torch.diag(out)
    return out_.reshape(b, *((1,) * (len(x_shape) - 1)))

class GaussianDiffusion1D(nn.Module):
    def __init__(
        self,
        model,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        mid_step = True,
        loss_type = 'l1',
        objective = 'noise',
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        scheduler = 'cosine'
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.seq_length = seq_length
        self.objective = objective
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.max_var = 2
        self.mid_step = mid_step
        self.scheduler = scheduler
        self.eta = 1
        self.clip_denoised = True

        if scheduler == 'linear':
            m_t = linear_generate_mt(self.num_timesteps)
        elif scheduler == 'cosine':
            m_t = cosine_generate_mt(self.num_timesteps)
        else:
            raise NotImplementedError

        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = y - objective_recon
        elif self.objective == 'x0':
            x0_recon = objective_recon
        else:
            raise NotImplementedError
        return x0_recon

    @torch.no_grad()
    def sample(self, y, mode=None):
        device = y.device
        if self.mid_step:
            if self.scheduler == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sampling_timesteps - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.scheduler == 'cosine':
                steps = np.linspace(start=-1, stop=self.num_timesteps - 1, num=self.sampling_timesteps + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

        times = list(self.steps.int().tolist())
        time_pairs = list(zip(times[:-1], times[1:]))

        # noise = torch.randn_like(y)
        # img = noise
        img = y
        condition = y.view(y.size()[0], -1)

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((img.size()[0],), time, device=device, dtype=torch.float32)

            objective_recon = self.model(img, time_cond, condition)
            x0_recon = self.predict_x0_from_objective(img, y, time_cond, objective_recon=objective_recon)
            if self.clip_denoised:
                x0_recon.clamp_(-1., 1.)

            if mode == 'half':
                if time_next < 500:
                    img = x0_recon
                    continue

            if time_next == 0:
                img = x0_recon
                continue

            m_t = self.m_t[time]
            m_nt = self.m_t[time_next]
            var_t = self.variance_t[time]
            var_nt = self.variance_t[time_next]
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta
            delta_t_nt = var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt **2)

            noise = torch.randn_like(img)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (img - (1. - m_t) * x0_recon - m_t * y)

            img = x_tminus_mean + sigma_t * noise

        img = unnormalize_to_zero_to_one(img)
        return img

    def q_sample(self, x0, y, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        # noise_end = torch.randn_like(y)
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        if self.objective == 'grad':
            objective = m_t * (y - x0) + sigma_t * noise
        elif self.objective == 'noise':
            objective = noise
        elif self.objective == 'ysubx':
            objective = y - x0
        elif self.objective == 'x0':
            objective = x0
        else:
            raise NotImplementedError()

        return (
            (1. - m_t) * x0 + m_t * y + sigma_t * noise,
            objective
        )

    def p_losses(self, x0, y, t):
        noise = torch.randn_like(x0)
        b = x0.size()[0]

        x_t, objective = self.q_sample(x0 = x0, y=y, t = t, noise = noise)

        condition = y.view(y.size()[0], -1)
        model_out = self.model(x_t, t, condition)
        recloss = F.mse_loss(objective, model_out)
        # regloss = F.mse_loss(x0, (y - model_out))

        # x0_recon = self.predict_x0_from_objective(x_t, y.view(y.size()[0], self.model.channels, self.seq_length), t, model_out)
        # gaploss = F.mse_loss(x0, x0_recon)

        # loss = recloss + regloss

        return recloss, model_out.view(b, -1)

    def forward(self, img, txt):
        b, device, seq_length, = img.size()[0], img.device, self.seq_length
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        img = self.normalize(img)
        return self.p_losses(img, txt, t)

