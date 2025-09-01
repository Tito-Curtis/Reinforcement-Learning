import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List


@dataclass
class EnvConfig:
    col_renewable_kw: str = "Combined_Renewable_kW"
    col_load_kw: str = "Load_Demand_kW"
    col_price: str = "Electricity_Price_USDperkWh"
    col_batt_soc: str = "Battery_SoC_%"
    col_sc_energy: str = "SC_Energy_kWh_raw"
    
    
    batt_capacity_kwh: float = 100.0
    batt_max_c_rate: float = 0.5
    batt_eta_charge: float = 0.96
    batt_eta_discharge: float = 0.96
    batt_soc_min: float = 0.1
    batt_soc_max: float = 0.9
    batt_throughput_cost: float = 0.02
    
    sc_capacity_kwh: float = 5.0
    sc_max_c_rate: float = 10.0
    sc_eta_charge: float = 0.98
    sc_eta_discharge: float = 0.98
    sc_soc_min: float = 0.05
    sc_soc_max: float = 0.95
    sc_throughput_cost: float = 0.005
    
    unmet_load_penalty: float = 2.0
    overflow_penalty: float = 2.0
    soc_violation_penalty: float = 5.0

@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 128
    replay_size: int = 50_000
    start_steps: int = 2_000
    train_after: int = 1_000
    train_freq: int = 1
    target_sync: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000
    max_grad_norm: float = 5.0


class HESSMicrogridEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, cfg: EnvConfig):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.T = len(self.df)
        self.ep_len = len(self.df)

        self.action_list: List[Tuple[float, float]] = []
        batt_p_max = self.cfg.batt_capacity_kwh * self.cfg.batt_max_c_rate
        sc_p_max = self.cfg.sc_capacity_kwh * self.cfg.sc_max_c_rate
        for pb in np.round(np.linspace(-batt_p_max, batt_p_max, 5), 3):
            for ps in np.round(np.linspace(-sc_p_max, sc_p_max, 5), 3):
                self.action_list.append((float(pb), float(ps)))
        
        self.action_space = spaces.Discrete(len(self.action_list))
        # The observation space now has 7 dimensions with the cyclical features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        
        self.ptr = 0
        self.t = 0
        self.batt_soc = None
        self.sc_soc = None
    
    def _obs(self) -> np.ndarray:
        row = self.df.loc[self.ptr + self.t]    
        obs = np.array([
            row[self.cfg.col_renewable_kw] / self.cfg.batt_capacity_kwh,
            row[self.cfg.col_load_kw] / self.cfg.batt_capacity_kwh,
            row[self.cfg.col_price],
            float(self.batt_soc),
            float(self.sc_soc),
            row['sin_hour'],
            row['cos_hour']
        ], dtype=np.float32)
        return obs
        
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.ptr = 0
        self.t = 0
        self.batt_soc = 0.5
        self.sc_soc = 0.5
        return self._obs(), {}
    
    def step(self, action_idx: int):
        p_batt_kw, p_sc_kw = self.action_list[action_idx]
        dt = 1.0
        
        # Calculate energy flow and SOC updates
        batt_e_out = max(0.0, p_batt_kw) * dt * self.cfg.batt_eta_discharge
        batt_e_in = max(0.0, -p_batt_kw) * dt / max(1e-6, self.cfg.batt_eta_charge)
        sc_e_out = max(0.0, p_sc_kw) * dt * self.cfg.sc_eta_discharge
        sc_e_in = max(0.0, -p_sc_kw) * dt / max(1e-6, self.cfg.sc_eta_charge)
        
        batt_energy_next = self.batt_soc * self.cfg.batt_capacity_kwh - batt_e_out + batt_e_in
        sc_energy_next = self.sc_soc * self.cfg.sc_capacity_kwh - sc_e_out + sc_e_in
        
        batt_soc_next = batt_energy_next / self.cfg.batt_capacity_kwh
        sc_soc_next = sc_energy_next / self.cfg.sc_capacity_kwh
        
        # Handle SOC violations
        soc_violation = 0.0
        if batt_soc_next < self.cfg.batt_soc_min:
            soc_violation += (self.cfg.batt_soc_min - batt_soc_next)
            batt_soc_next = self.cfg.batt_soc_min
        if batt_soc_next > self.cfg.batt_soc_max:
            soc_violation += (batt_soc_next - self.cfg.batt_soc_max)
            batt_soc_next = self.cfg.batt_soc_max
            
        # Calculate reward
        row = self.df.loc[self.ptr + self.t]
        renewable = row[self.cfg.col_renewable_kw]
        load = row[self.cfg.col_load_kw]
        price = row[self.cfg.col_price]

        grid_import_kw = max(0, load - renewable - max(0, p_batt_kw) - max(0, p_sc_kw))
        energy_cost = grid_import_kw * dt * price
        
        batt_throughput = batt_e_out + batt_e_in
        sc_throughput = sc_e_out + sc_e_in
        degr_cost = batt_throughput * self.cfg.batt_throughput_cost + sc_throughput * self.cfg.sc_throughput_cost
        
        penalty = self.cfg.soc_violation_penalty * soc_violation
        
        reward = -(energy_cost + degr_cost + penalty)

        self.batt_soc = batt_soc_next
        self.sc_soc = sc_soc_next
        self.t += 1
        terminated = (self.t >= self.ep_len)
        truncated = False
        
        next_obs = self._obs() if not terminated else np.zeros(7)
        info = {}
        
        return next_obs, reward, terminated, truncated, info