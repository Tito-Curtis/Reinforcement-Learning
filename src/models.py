import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random

from env import HESSMicrogridEnv,TrainConfig,EnvConfig



class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, act_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.idx = 0
        self.full = False

    def push(self, o, a, r, no, d):
        self.obs[self.idx] = o
        self.actions[self.idx] = a
        self.rewards[self.idx] = r
        self.next_obs[self.idx] = no
        self.dones[self.idx] = d
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int):
        max_idx = self.capacity if self.full else self.idx
        idxs = np.random.randint(0, max_idx, size=batch_size)
        return (
            torch.from_numpy(self.obs[idxs]),
            torch.from_numpy(self.actions[idxs]),
            torch.from_numpy(self.rewards[idxs]),
            torch.from_numpy(self.next_obs[idxs]),
            torch.from_numpy(self.dones[idxs]),)


class DQNAgent:
    def __init__(self, obs_dim: int, act_dim: int, tcfg: TrainConfig):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.tcfg = tcfg
        self.q = QNetwork(obs_dim, act_dim).to(tcfg.device)
        self.q_target = QNetwork(obs_dim, act_dim).to(tcfg.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=tcfg.lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.step_count = 0
        self.eps = tcfg.eps_start

    def act(self, obs: np.ndarray) -> int:
        self.step_count += 1
        decay = max(0.0, min(1.0, self.step_count / self.tcfg.eps_decay_steps))
        self.eps = self.tcfg.eps_start + decay * (self.tcfg.eps_end - self.tcfg.eps_start)
        if random.random() < self.eps:
            return random.randrange(self.act_dim)
        with torch.no_grad():
            o = torch.tensor(obs, dtype=torch.float32, device=self.tcfg.device).unsqueeze(0)
            qvals = self.q(o)
            return int(torch.argmax(qvals, dim=1).item())

    def update(self, buf: ReplayBuffer, batch_size: int) -> float:
        o, a, r, no, d = buf.sample(batch_size)
        o = o.to(self.tcfg.device)
        a = a.to(self.tcfg.device)
        r = r.to(self.tcfg.device)
        no = no.to(self.tcfg.device)
        d = d.to(self.tcfg.device)

        qvals = self.q(o).gather(1, a.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            next_q = self.q_target(no).max(dim=1)[0]
            target = r + (1.0 - d) * self.tcfg.gamma * next_q
        loss = self.loss_fn(qvals, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.tcfg.max_grad_norm)
        self.opt.step()
        return float(loss.item())

    def sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())

