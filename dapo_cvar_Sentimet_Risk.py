from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm import tqdm
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta
import base64
from scipy.stats import skew, kurtosis
from finrl.config import INDICATORS
import os
from typing import Sequence, Optional
# Global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Neural Network Utilities
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)
class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError
    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError
    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a
class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)
    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)
class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)
    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)
class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(512, 512), activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)
        self.to(device)
    def step(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
        return a.cpu().numpy(), logp_a.cpu().numpy()
    def act_batch(self, obs, num_samples=10):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
            actions = []
            logps = []
            for _ in range(num_samples):
                pi = self.pi._distribution(obs)
                a = pi.sample()
                logp_a = self.pi._log_prob_from_distribution(pi, a)
                actions.append(a.cpu().numpy())
                logps.append(logp_a.cpu().numpy())
            return actions, logps
    def act(self, obs):
        return self.step(obs)[0]
# Stock Trading Environment
class StockTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    @staticmethod
    def _ensure_list_len(x, name: str, want_len: int):
        if len(x) != want_len:
            raise ValueError(f"{name} must have length {want_len}, got {len(x)}")
    @staticmethod
    def _require_columns(df: pd.DataFrame, cols: Sequence[str]):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in df: {missing}")
    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: float,
        num_stock_shares: Sequence[int],
        buy_cost_pct: Sequence[float],
        sell_cost_pct: Sequence[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: Sequence[str],
        turbulence_threshold: Optional[float] = None,
        risk_indicator_col: str = "turbulence",
        llm_sentiment_col: str = "llm_sentiment",
        llm_risk_col: str = "llm_risk",
        make_plots: bool = False,
        print_verbosity: int = 10,
        day: int = 0,
        initial: bool = True,
        previous_state: Sequence[float] = (),
    ):
        self.df = df.copy()
        self.stock_dim = int(stock_dim)
        self.hmax = int(hmax)
        self.initial_amount = float(initial_amount)
        self.reward_scaling = float(reward_scaling)
        self.tech_indicator_list = list(tech_indicator_list)
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.llm_sentiment_col = llm_sentiment_col
        self.llm_risk_col = llm_risk_col
        self.make_plots = make_plots
        self.print_verbosity = int(print_verbosity)
        self.num_stock_shares = list(map(int, num_stock_shares))
        self.buy_cost_pct = list(map(float, buy_cost_pct))
        self.sell_cost_pct = list(map(float, sell_cost_pct))
        self._ensure_list_len(self.num_stock_shares, "num_stock_shares", self.stock_dim)
        self._ensure_list_len(self.buy_cost_pct, "buy_cost_pct", self.stock_dim)
        self._ensure_list_len(self.sell_cost_pct, "sell_cost_pct", self.stock_dim)
        base_cols = ["date", "tic", "close"]
        self._require_columns(self.df, base_cols + list(self.tech_indicator_list))
        for c in [self.llm_sentiment_col, self.llm_risk_col]:
            if c not in self.df.columns:
                self.df[c] = 3
        self.day = int(day)
        self.initial = bool(initial)
        self.previous_state = list(previous_state) if previous_state else []
        self.terminal = False
        self.action_dim = int(action_space)
        if self.action_dim != self.stock_dim:
            self.action_dim = self.stock_dim
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32)
        self.data = self.df.loc[self.day, :] if self._index_has_day(self.df.index, self.day) else self.df.iloc[0]
        self.reward = 0.0
        self.turbulence = 0.0
        self.cost = 0.0
        self.trades = 0
        self.episode = 0
        self.state = self._initiate_state()
        self.state_space = len(self.state)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.state_space,), dtype=np.float32)
        self.asset_memory = [
            self.initial_amount + np.sum(
                np.array(self.num_stock_shares, dtype=np.float32) * np.array(self.state[1:1 + self.stock_dim], dtype=np.float32)
            )
        ]
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = []
        self.date_memory = [self._get_date()]
        self.seed()
    @staticmethod
    def _index_has_day(index, day: int) -> bool:
        try:
            _ = index.get_loc(day)
            return True
        except Exception:
            return False
    def _get_prices(self) -> np.ndarray:
        if len(self.df.tic.unique()) > 1:
            vals = np.asarray(self.data["close"].values, dtype=np.float32)
        else:
            vals = np.asarray([self.data["close"]], dtype=np.float32)
        if vals.shape[0] != self.stock_dim:
            vals = self._fit_vector(vals, self.stock_dim, fill=0.0)
        return vals
    @staticmethod
    def _fit_vector(v: np.ndarray, target_len: int, fill: float = 0.0) -> np.ndarray:
        if v.shape[0] == target_len:
            return v
        if v.shape[0] > target_len:
            return v[:target_len]
        pad = np.full((target_len - v.shape[0],), fill, dtype=v.dtype)
        return np.concatenate([v, pad], axis=0)
    def _get_indicators_flat(self) -> np.ndarray:
        chunks = []
        for ind in self.tech_indicator_list:
            if len(self.df.tic.unique()) > 1:
                vec = np.asarray(self.data[ind].values, dtype=np.float32)
            else:
                vec = np.asarray([self.data[ind]], dtype=np.float32)
            vec = self._fit_vector(vec, self.stock_dim, fill=0.0)
            chunks.append(vec)
        if not chunks:
            return np.zeros((0,), dtype=np.float32)
        return np.concatenate(chunks, axis=0)
    def _get_llm_block(self, col: str, default_val: float = 3.0) -> np.ndarray:
        if len(self.df.tic.unique()) > 1:
            vec = np.asarray(self.data[col].values, dtype=np.float32)
        else:
            vec = np.asarray([self.data[col]], dtype=np.float32)
        vec = np.where(np.isnan(vec), default_val, vec)
        vec = self._fit_vector(vec, self.stock_dim, fill=default_val)
        return vec
    def _initiate_state(self) -> list[float]:
        if self.initial:
            cash = [float(self.initial_amount)]
            prices = self._get_prices().tolist()
            shares = list(map(float, self.num_stock_shares))
            indicators = self._get_indicators_flat().tolist()
            sent = self._get_llm_block(self.llm_sentiment_col).tolist()
            risk = self._get_llm_block(self.llm_risk_col).tolist()
            state = cash + prices + shares + indicators + sent + risk
        else:
            prev_cash = [float(self.previous_state[0])] if len(self.previous_state) > 0 else [float(self.initial_amount)]
            prices = self._get_prices().tolist()
            if len(self.previous_state) >= (self.stock_dim * 2 + 1):
                prev_shares = list(map(float, self.previous_state[1 + self.stock_dim:1 + 2*self.stock_dim]))
            else:
                prev_shares = [0.0] * self.stock_dim
            indicators = self._get_indicators_flat().tolist()
            sent = self._get_llm_block(self.llm_sentiment_col).tolist()
            risk = self._get_llm_block(self.llm_risk_col).tolist()
            state = prev_cash + prices + prev_shares + indicators + sent + risk
        return state
    def _update_state(self) -> list[float]:
        cash = [float(self.state[0])]
        prices = self._get_prices().tolist()
        shares = list(map(float, self.state[1 + self.stock_dim:1 + 2*self.stock_dim]))
        indicators = self._get_indicators_flat().tolist()
        sent = self._get_llm_block(self.llm_sentiment_col).tolist()
        risk = self._get_llm_block(self.llm_risk_col).tolist()
        return cash + prices + shares + indicators + sent + risk
    def _sell_stock(self, index: int, action: float) -> float:
        def _do_sell_normal() -> float:
            current_shares = self.state[1 + self.stock_dim + index]
            if current_shares > 0:
                sell_num = float(min(abs(action), current_shares))
                price = float(self.state[1 + index])
                sell_amount = price * sell_num * (1.0 - self.sell_cost_pct[index])
                self.state[0] += sell_amount
                self.state[1 + self.stock_dim + index] -= sell_num
                self.cost += price * sell_num * self.sell_cost_pct[index]
                self.trades += 1
            else:
                sell_num = 0.0
            return sell_num
        if self.turbulence_threshold is not None and self.turbulence >= self.turbulence_threshold:
            price = float(self.state[1 + index])
            current_shares = float(self.state[1 + self.stock_dim + index])
            if price > 0 and current_shares > 0:
                sell_num = current_shares
                sell_amount = price * sell_num * (1.0 - self.sell_cost_pct[index])
                self.state[0] += sell_amount
                self.state[1 + self.stock_dim + index] = 0.0
                self.cost += price * sell_num * self.sell_cost_pct[index]
                self.trades += 1
            else:
                sell_num = 0.0
        else:
            sell_num = _do_sell_normal()
        return sell_num
    def _buy_stock(self, index: int, action: float) -> float:
        def _do_buy() -> float:
            price = float(self.state[1 + index])
            if price <= 0:
                return 0.0
            available_amount = self.state[0] // (price * (1.0 + self.buy_cost_pct[index]))
            buy_num = float(min(available_amount, max(0.0, action)))
            buy_amount = price * buy_num * (1.0 + self.buy_cost_pct[index])
            self.state[0] -= buy_amount
            self.state[1 + self.stock_dim + index] += buy_num
            self.cost += price * buy_num * self.buy_cost_pct[index]
            self.trades += 1
            return buy_num
        if self.turbulence_threshold is not None and self.turbulence >= self.turbulence_threshold:
            return 0.0
        return _do_buy()
    def step(self, actions: Sequence[float]):
        terminated = self.day >= (len(self.df.index.unique()) - 1)
        truncated = False
        if terminated:
            end_total_asset = float(
                self.state[0] + np.sum(
                    np.array(self.state[1:1 + self.stock_dim], dtype=np.float32) *
                    np.array(self.state[1 + self.stock_dim:1 + 2*self.stock_dim], dtype=np.float32)
                )
            )
            obs = np.asarray(self.state, dtype=np.float32)
            self.reward = float(self.reward)
            return obs, self.reward, bool(terminated), bool(truncated), {}
        try:
            actions = np.asarray(actions, dtype=np.float32).flatten()
            actions = np.clip(actions, -1.0, 1.0)
            if actions.shape[0] != self.stock_dim:
                actions = self._fit_vector(actions, self.stock_dim, fill=0.0)
            actions = (actions * self.hmax).astype(np.int32)
            begin_total_asset = float(
                self.state[0] + np.sum(
                    np.array(self.state[1:1 + self.stock_dim], dtype=np.float32) *
                    np.array(self.state[1 + self.stock_dim:1 + 2*self.stock_dim], dtype=np.float32)
                )
            )
            sell_index = np.where(actions < 0)[0]
            for idx in sell_index:
                self._sell_stock(int(idx), float(abs(actions[idx])))
            buy_index = np.where(actions > 0)[0]
            for idx in buy_index:
                self._buy_stock(int(idx), float(actions[idx]))
            self.actions_memory.append(actions.copy())
            self.day += 1
            self.data = self.df.loc[self.day, :] if self._index_has_day(self.df.index, self.day) else self.df.iloc[self.day]
            if self.turbulence_threshold is not None:
                try:
                    if len(self.df.tic.unique()) == 1:
                        self.turbulence = float(self.data[self.risk_indicator_col])
                    else:
                        self.turbulence = float(np.asarray(self.data[self.risk_indicator_col].values, dtype=np.float32)[0])
                except Exception:
                    self.turbulence = 0.0
            self.state = self._update_state()
            end_total_asset = float(
                self.state[0] + np.sum(
                    np.array(self.state[1:1 + self.stock_dim], dtype=np.float32) *
                    np.array(self.state[1 + self.stock_dim:1 + 2*self.stock_dim], dtype=np.float32)
                )
            )
            raw_reward = end_total_asset - begin_total_asset
            tail_lambda = 0.5
            tail_window = 30
            try:
                asset_hist = list(self.asset_memory) if hasattr(self, "asset_memory") else []
                arr = np.array(asset_hist + [begin_total_asset], dtype=float)
                if arr.size >= 2:
                    returns = np.diff(arr) / arr[:-1]
                    window = min(len(returns), tail_window)
                    recent = returns[-window:]
                    alpha = 0.01
                    var = np.percentile(recent, alpha * 100)
                    tail_losses = recent[recent <= var]
                    cvar = tail_losses.mean() if tail_losses.size > 0 else var
                    tail_loss = -cvar * begin_total_asset if cvar < 0 else 0.0
                else:
                    tail_loss = 0.0
            except Exception:
                tail_loss = 0.0
            adj_raw_reward = raw_reward - tail_lambda * tail_loss
            self.reward = float(adj_raw_reward * self.reward_scaling)
            self.rewards_memory.append(self.reward)
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.state_memory.append(self.state)
            obs = np.asarray(self.state, dtype=np.float32)
            return obs, self.reward, bool(terminated), bool(truncated), {}
        except Exception as e:
            obs = np.asarray(self.state, dtype=np.float32)
            return obs, 0.0, False, False, {"error": str(e)}
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.day = 0
        self.data = self.df.loc[self.day, :] if self._index_has_day(self.df.index, self.day) else self.df.iloc[0]
        self.state = self._initiate_state()
        self.asset_memory = [
            float(self.initial_amount) + float(np.sum(
                np.array(self.num_stock_shares, dtype=np.float32) * np.array(self.state[1:1 + self.stock_dim], dtype=np.float32)
            ))
        ]
        self.turbulence = 0.0
        self.cost = 0.0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = [self.state]
        self.date_memory = [self._get_date()]
        self.episode += 1
        obs = np.asarray(self.state, dtype=np.float32)
        return obs, {}
    def render(self):
        return self.state
    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            try:
                return self.data["date"].unique()[0]
            except Exception:
                return self.data["date"].iloc[0]
        else:
            return self.data["date"]
    def save_state_memory(self):
        date_list = self.date_memory[:-1]
        return pd.DataFrame({"date": date_list, "state": self.state_memory})
    def save_asset_memory(self):
        return pd.DataFrame({"date": self.date_memory, "account_value": self.asset_memory})
    def save_action_memory(self):
        date_list = self.date_memory[:-1]
        df_actions = pd.DataFrame(self.actions_memory)
        try:
            tickers = self.data["tic"].values if hasattr(self.data["tic"], "values") else None
            if tickers is not None and len(tickers) >= df_actions.shape[1]:
                df_actions.columns = list(tickers)[:df_actions.shape[1]]
        except Exception:
            pass
        df_actions.insert(0, "date", date_list)
        return df_actions
    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
# DAPO Prediction
def custom_DAPO_prediction(act, environment, device):
    state, _ = environment.reset()
    s_tensor = torch.as_tensor(state[None, :], dtype=torch.float32, device=device)
    episode_assets = []
    episode_dates = []
    episode_states = []
    while True:
        actions, _ = act.act_batch(s_tensor, num_samples=1)
        action = actions[0].squeeze()
        next_state, reward, terminated, truncated, _ = environment.step(action)
        done = terminated or truncated
        episode_assets.append(environment.asset_memory[-1])
        episode_dates.append(environment.date_memory[-1])
        episode_states.append(environment.state_memory[-1])
        state = next_state
        s_tensor = torch.as_tensor(state[None, :], dtype=torch.float32, device=device)
        if done:
            break
    return episode_assets, episode_dates, episode_states
def enhanced_DRL_prediction(act, environment):
    assets, dates, states = custom_DAPO_prediction(act, environment, device=device)
    if len(assets) < 2:
        pass
    return assets, dates, states
# Performance Metrics
def calculate_performance_metrics(portfolio_series, trading_days=252):
    try:
        returns = portfolio_series.pct_change(1).dropna()
        if len(returns) < 2:
            return {k: 0 for k in ["Lợi nhuận Hàng năm (%)", "Lợi nhuận Tích lũy (%)", "Biến động Hàng năm (%)",
                                   "Tỷ lệ Sharpe", "Tỷ lệ Sortino", "Sụt giảm Tối đa (Max DD)", "VaR (1%)",
                                   "Biên độ", "Tối đa", "Tối thiểu", "Độ lệch (Skewness)", "Độ nhọn (Kurtosis)"]}
        num_years = (portfolio_series.index[-1] - portfolio_series.index[0]).days / 365.25
        cagr = ((portfolio_series.iloc[-1] / portfolio_series.iloc[0]) ** (1 / num_years)) - 1 if num_years > 0 else 0
        cumulative_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
        annual_volatility = returns.std() * np.sqrt(trading_days)
        sharpe_ratio = returns.mean() / (returns.std() + 1e-8) * np.sqrt(trading_days)
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std() if len(negative_returns) > 1 else 1e-8
        sortino_ratio = returns.mean() / (downside_std + 1e-8) * np.sqrt(trading_days)
        running_max = portfolio_series.cummax()
        drawdown = (portfolio_series - running_max) / running_max
        metrics = {
            "Lợi nhuận Hàng năm (%)": cagr * 100,
            "Lợi nhuận Tích lũy (%)": cumulative_return * 100,
            "Biến động Hàng năm (%)": annual_volatility * 100,
            "Tỷ lệ Sharpe": sharpe_ratio,
            "Tỷ lệ Sortino": sortino_ratio,
            "Sụt giảm Tối đa (Max DD)": drawdown.min(),
            "VaR (1%)": returns.quantile(0.01),
            "Biên độ": returns.max() - returns.min(),
            "Tối đa": returns.max(),
            "Tối thiểu": returns.min(),
            "Độ lệch (Skewness)": skew(returns),
            "Độ nhọn (Kurtosis)": kurtosis(returns),
        }
        return metrics
    except Exception as e:
        return {k: 0 for k in ["Lợi nhuận Hàng năm (%)", "Lợi nhuận Tích lũy (%)", "Biến động Hàng năm (%)",
                               "Tỷ lệ Sharpe", "Tỷ lệ Sortino", "Sụt giảm Tối đa (Max DD)", "VaR (1%)",
                               "Biên độ", "Tối đa", "Tối thiểu", "Độ lệch (Skewness)", "Độ nhọn (Kurtosis)"]}
# Treemap Visualization
def plot_final_allocation_treemap(df_allocations, model_name):
    try:
        df_alloc = df_allocations.T.reset_index()
        df_alloc.columns = ['ticker', 'percentage']
        total = df_alloc['percentage'].sum()
        if total > 0:
            df_alloc['percentage'] = df_alloc['percentage'] / total * 100
        else:
            df_alloc['percentage'] = 0.0
            df_alloc.loc[df_alloc['ticker'] == 'CASH', 'percentage'] = 100.0
        plot_data = df_alloc[df_alloc['percentage'] > 0.1].sort_values(by='percentage', ascending=False)
        if plot_data.empty:
            st.warning("Không có dữ liệu hợp lệ để vẽ biểu đồ treemap.")
            return
        colors = ['#AB63FA', '#636EFA', '#A1CAF1', '#19D3F3', '#00CC96', '#B6E880', '#FECB52', '#FFA15A', '#EF553B', '#D62728']
        fig = px.treemap(
            plot_data,
            path=['ticker'],
            values='percentage',
            color='percentage',
            color_continuous_scale=colors,
            custom_data=['ticker', 'percentage']
        )
        fig.update_traces(
            hovertemplate='<b>%{customdata[0]}</b><br>Tỷ trọng: %{customdata[1]:.2f}%<extra></extra>',
            texttemplate='<b>%{label}<br>%{value:.2f}%</b>',
            textposition="middle center",
            textfont=dict(size=10, family="Arial Black")
        )
        fig.update_layout(
            title=f"Phân bổ Danh mục Trung bình Cuối kỳ dạng Treemap - Model: {model_name}",
            width=1000,
            height=800,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Lỗi khi vẽ treemap: {e}")
# Backtest Function
def run_backtest(start_date, end_date):
    try:
        trade_risk = pd.read_excel(RISK_DATA_PATH, engine="openpyxl")
        trade_sent = pd.read_excel(SENTIMENT_DATA_PATH, engine="openpyxl")
        past_data = pd.read_excel(PAST_DATA_PATH, engine="openpyxl")
    except FileNotFoundError as e:
        st.error(f"Lỗi: Không tìm thấy file dữ liệu: {e}")
        return None, None, None, None
    # Chuẩn hóa cột và định dạng ngày
    for df in (trade_risk, trade_sent, past_data):
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
    # Gộp dữ liệu từ PAST_DATA_PATH với trade_risk và trade_sent
  
    base_cols = ["date", "tic", "close"] + INDICATORS
    for df in (trade_risk, trade_sent, past_data):
        for col in base_cols:
            if col not in df.columns:
                df[col] = 0.0 if col != "date" and col != "tic" else df.get(col, None)
    # Gộp trade_risk và trade_sent trước
    trade = pd.merge(trade_risk, trade_sent, on=["date", "tic"], suffixes=("", "_sent"), how="outer")
   
    # Gộp với past_data
    trade = pd.concat([trade, past_data], ignore_index=True)
   
    # Xử lý các cột llm_sentiment và llm_risk
    if "llm_sentiment_sent" in trade.columns:
        trade["llm_sentiment"] = trade["llm_sentiment_sent"].fillna(3)
        trade.drop(columns=["llm_sentiment_sent"], inplace=True)
    for col in ["llm_sentiment", "llm_risk"]:
        if col not in trade.columns:
            trade[col] = 3
        trade[col] = trade[col].fillna(3)
    # Lọc dữ liệu theo khoảng thời gian
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    trade = trade[(trade["date"] >= start_date) & (trade["date"] <= end_date)].copy()
    if trade.empty:
        st.error(f"Lỗi: Không có dữ liệu giao dịch từ {start_date} đến {end_date}.")
        return None, None, None, None
    # Tạo chỉ số ngày duy nhất
    uniq_dates = trade["date"].sort_values().unique()
    date_to_idx = {d: i for i, d in enumerate(uniq_dates)}
    trade["new_idx"] = trade["date"].map(date_to_idx)
    trade.set_index("new_idx", inplace=True)
    stock_dim = len(trade["tic"].unique())
    state_space = 1 + 2 * stock_dim + (len(INDICATORS) + 2) * stock_dim
    initial_amount = 1_000_000
    env = StockTradingEnv(
        df=trade, stock_dim=stock_dim, hmax=100, initial_amount=initial_amount,
        num_stock_shares=[0] * stock_dim, buy_cost_pct=[0.001] * stock_dim,
        sell_cost_pct=[0.001] * stock_dim, reward_scaling=1e-4, state_space=state_space,
        action_space=stock_dim, tech_indicator_list=INDICATORS, turbulence_threshold=120,
        risk_indicator_col="turbulence", llm_sentiment_col="llm_sentiment",
        llm_risk_col="llm_risk", make_plots=False, print_verbosity=1000
    )
    model_name = "DAPO (Cvar 0.01 Phobert 1a.3b)"
    model_seed_runs = []
    model_seed_states = []
    run_dates = None
    for seed in tqdm(SEEDS, desc=f"Chạy seeds cho {model_name}", disable=True):
        torch.manual_seed(seed)
        np.random.seed(seed)
        env.seed(seed)
        policy = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=(512, 512), activation=nn.ReLU)
        try:
            ckpt = torch.load(MODEL_PATH, map_location=device)
            sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            policy.load_state_dict(sd, strict=False)
            policy.eval().to(device)
        except Exception as e:
            st.error(f"Lỗi: Không tải được mô hình {model_name}: {e}")
            return None, None, None, None
        assets, dates, states = enhanced_DRL_prediction(policy, env)
        model_seed_runs.append(assets)
        model_seed_states.append(states)
        if run_dates is None:
            run_dates = dates
    if not model_seed_runs:
        st.error(f"Lỗi: Không có kết quả chạy thành công cho mô hình {model_name}.")
        return None, None, None, None
    min_len = min(len(run) for run in model_seed_runs)
    aligned_runs = [run[:min_len] for run in model_seed_runs]
    aligned_states = [states[:min_len] for states in model_seed_states]
    average_assets = np.mean(aligned_runs, axis=0)
    current_dates = pd.to_datetime(run_dates[:min_len])
    metrics = calculate_performance_metrics(pd.Series(average_assets, index=current_dates))
    final_states_for_model = [run_states[-1] for run_states in aligned_states]
    allocations_for_model = []
    for final_state in final_states_for_model:
        cash = final_state[0]
        prices = np.array(final_state[1:1 + stock_dim])
        holdings = np.array(final_state[1 + stock_dim:1 + 2 * stock_dim])
        stock_values = prices * holdings
        total_assets = cash + np.sum(stock_values)
        if total_assets > 0:
            stock_pct = stock_values / total_assets
            cash_pct = cash / total_assets
        else:
            stock_pct = np.zeros(stock_dim)
            cash_pct = 1.0
        allocation = np.concatenate([stock_pct, [cash_pct]])
        allocation_sum = np.sum(allocation)
        if allocation_sum > 0:
            allocation = allocation / allocation_sum
        else:
            allocation = np.array([0.0] * stock_dim + [1.0])
        allocations_for_model.append(allocation)
    final_allocations = np.mean(allocations_for_model, axis=0) * 100
    total_alloc = np.sum(final_allocations)
    if total_alloc > 0:
        final_allocations = final_allocations / total_alloc * 100
    else:
        final_allocations = np.array([0.0] * stock_dim + [100.0])
    tickers = list(trade["tic"].unique()) + ["CASH"]
    df_allocations = pd.DataFrame({model_name: final_allocations}, index=tickers).T
    return average_assets, current_dates, metrics, df_allocations
# Config
st.set_page_config(
    page_title="DAPO_APP news signals and CVAR",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Data Paths
DATA_DIR = "./data"
BACKGROUND_PATH = os.path.join(DATA_DIR, "background.png")
BIG_LOGO_PATH = os.path.join(DATA_DIR, "big_logo.png")
BANNER_PATH = os.path.join(DATA_DIR, "banner.png")
RISK_DATA_PATH = os.path.join(DATA_DIR, "test2025_risk - Copy.xlsx")
SENTIMENT_DATA_PATH = os.path.join(DATA_DIR, "test2025_sentiment - Copy.xlsx")
PAST_DATA_PATH = os.path.join(DATA_DIR, "train2025_risk - Copy.xlsx")
MODEL_PATH = os.path.join(DATA_DIR, "agent_dapo_both_a1.0_b3.0.pth")
QUY_TRINH_PATH = os.path.join(DATA_DIR, "quy_trinh.png")
NUM_SEEDS_TO_TEST = 1
SEEDS = [i * 42 for i in range(NUM_SEEDS_TO_TEST)]
# BACKGROUND / CSS
def add_bg_from_local(image_file):
    try:
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/png;base64,{encoded});
                background-size: cover;
                background-color: rgba(255, 255, 255, 0.5);
                background-blend-mode: lighten;
            }}
            
            .custom-title {{
                color: #00D4FF;
                font-family: 'Orbitron', sans-serif;
                font-weight: 700;
                text-shadow: 0 0 10px rgba(0, 212, 255, 0.7);
            }}
            
            .stMarkdown, .stText {{
                color: #E6E6FA !important;
            }}
            
            /* Styling cho bảng - nền trắng với viền đen */
            .stDataFrame, [data-testid="stDataFrame"] {{
                background-color: white !important;
            }}
            
            .stDataFrame table, [data-testid="stDataFrame"] table {{
                background-color: white !important;
                border: 2px solid black !important;
            }}
            
            .stDataFrame th, [data-testid="stDataFrame"] th {{
                background-color: #f0f0f0 !important;
                color: black !important;
                border: 1px solid black !important;
                font-weight: bold !important;
                padding: 8px !important;
            }}
            
            .stDataFrame td, [data-testid="stDataFrame"] td {{
                background-color: white !important;
                color: black !important;
                border: 1px solid black !important;
                padding: 8px !important;
            }}
            
            /* Styling cho st.table */
            table {{
                background-color: white !important;
                border: 2px solid black !important;
                border-collapse: collapse !important;
            }}
            
            table th {{
                background-color: #f0f0f0 !important;
                color: black !important;
                border: 1px solid black !important;
                font-weight: bold !important;
                padding: 8px !important;
            }}
            
            table td {{
                background-color: white !important;
                color: black !important;
                border: 1px solid black !important;
                padding: 8px !important;
            }}
            
            .stButton > button {{
                font-weight: bold !important;
                font-size: 20px !important;
                color: #00D4FF !important;
                background-color: rgba(0, 0, 0, 0.7);
                border: 2px solid #00D4FF !important;
                border-radius: 5px !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 0 8px rgba(0, 212, 255, 0.5);
            }}
            
            .stButton > button:hover {{
                background-color: #00D4FF !important;
                color: #000000 !important;
                box-shadow: 0 0 12px rgba(0, 212, 255, 0.8);
            }}
            
            .stDateInput > label {{
                font-size: 20px !important;
                font-weight: bold !important;
                color: #00D4FF !important;
                font-family: 'Roboto Mono', monospace;
            }}
            
            [data-testid="stSidebar"] {{
                background-color: white !important;
                border-right: 2px solid #00D4FF;
                box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
            }}
            
            [data-testid="stSidebar"] .stRadio > label {{
                font-family: 'Roboto Mono', monospace;
                color: #000000 !important;
                font-size: 18px;
                padding: 10px;
                border-radius: 5px;
                background: #f0f0f0;
                transition: all 0.3s ease;
            }}
            
            [data-testid="stSidebar"] .stRadio > label:hover {{
                background: #e0e0e0;
                color: #000000 !important;
                box-shadow: 0 0 8px rgba(0, 212, 255, 0.7);
            }}
            
            [data-testid="stSidebar"] .stRadio > label > div > input:checked + div {{
                background-color: #00D4FF !important;
                border-color: #000000 !important;
            }}
            
            /* Chỉnh màu text trong sidebar */
            [data-testid="stSidebar"] * {{
                color: #000000 !important;
            }}
            
            /* Chỉnh logo và text trong sidebar */
            [data-testid="stSidebar"] .stMarkdown {{
                color: #000000 !important;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Roboto+Mono:wght@400;700&display=swap" rel="stylesheet">
            """,
            unsafe_allow_html=True,
        )
    except FileNotFoundError:
        st.warning("Hình nền không tìm thấy. Vui lòng kiểm tra file `background.png`.")
add_bg_from_local(BACKGROUND_PATH)
# SIDEBAR
try:
    st.sidebar.image(BIG_LOGO_PATH, width=120)
except FileNotFoundError:
    st.sidebar.warning("big_logo.png không tìm thấy.")
# Khởi tạo session state
if "page" not in st.session_state:
    st.session_state.page = "Main"
if "backtest_run" not in st.session_state:
    st.session_state.backtest_run = False
choice = st.sidebar.radio(
    "MENU",
    ["Main", "Explain & Guide"],
    index=["Main", "Explain & Guide"].index(st.session_state.page) if st.session_state.page in ["Main", "Explain & Guide"] else 0,
)
st.session_state.page = choice
st.sidebar.markdown("---")
# HEADER / BANNER
try:
    st.image(BANNER_PATH, use_container_width=True)
except FileNotFoundError:
    st.warning("Banner không tìm thấy. Vui lòng kiểm tra file `banner.png`.")
st.markdown('<h1 class="custom-title">ỨNG DỤNG MÔ HÌNH HỌC TĂNG CƯỜNG SÂU TÍCH HỢP TÍN HIỆU TIN TỨC VÀ PHÒNG NGỪA RỦI RO ĐUÔI TRONG QUẢN LÝ DANH MỤC ĐẦU TƯ TRÊN THỊ TRƯỜNG CHỨNG KHOÁN VIỆT NAM</h1>', unsafe_allow_html=True)
# MAIN CONTENT
if st.session_state.page == "Main":
    st.markdown("Vui lòng chọn khoảng thời gian để chạy kiểm thử. Khoảng thời gian phải nằm trong phạm vi dữ liệu và kéo dài ít nhất 4 tuần.")
    try:
        # Tải dữ liệu để xác định khoảng thời gian
        trade_risk = pd.read_excel(RISK_DATA_PATH, engine="openpyxl")
        trade_sent = pd.read_excel(SENTIMENT_DATA_PATH, engine="openpyxl")
        past_data = pd.read_excel(PAST_DATA_PATH, engine="openpyxl")
        for df in (trade_risk, trade_sent, past_data):
            df["date"] = pd.to_datetime(df["date"], errors='coerce')
            df.dropna(subset=['date'], inplace=True)
        # Tìm ngày sớm nhất và muộn nhất từ tất cả các nguồn dữ liệu
        min_date = min(trade_risk["date"].min(), trade_sent["date"].min(), past_data["date"].min()).date()
        max_date = max(trade_risk["date"].max(), trade_sent["date"].max(), past_data["date"].max()).date()
    except FileNotFoundError:
        min_date = datetime.today().date() - timedelta(days=365)
        max_date = datetime.today().date()
        st.warning("Không tìm thấy file dữ liệu. Sử dụng khoảng thời gian mặc định.")
    default_start_date = max_date - timedelta(weeks=8) if max_date else datetime.today().date() - timedelta(weeks=8)
    default_end_date = max_date if max_date else datetime.today().date()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(":red[Chọn ngày bắt đầu]", value=default_start_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input(":red[Chọn ngày kết thúc]", value=default_end_date, min_value=min_date, max_value=max_date)
    today = datetime.today().date()

    if start_date and end_date:
        if end_date > today:
            st.error("Lỗi: Ngày kết thúc không được vượt quá ngày hiện tại.")
        elif start_date >= end_date:
            st.error("Lỗi: Ngày kết thúc phải sau ngày bắt đầu.")
        elif (end_date - start_date) < timedelta(weeks=4):
            st.error("Lỗi: Khoảng thời gian phải dài ít nhất 4 tuần.")
        elif start_date < min_date or end_date > max_date:
            st.error(f"Lỗi: Khoảng thời gian phải nằm trong phạm vi dữ liệu từ {min_date} đến {max_date}.")
        else:
            
            if st.button("CHẠY KIỂM THỬ", use_container_width=True):
                with st.spinner("Đang chạy backtest..."):
                    st.session_state.backtest_run = True
                    assets, dates, metrics, df_allocations = run_backtest(start_date, end_date)
                    if assets is not None:
                        st.session_state.assets = assets
                        st.session_state.dates = dates
                        st.session_state.metrics = metrics
                        st.session_state.df_allocations = df_allocations
                        st.session_state.initial_amount = 1_000_000
    if "backtest_run" in st.session_state and st.session_state.backtest_run:
        st.subheader(":red[KẾt QUẢ KIỂM THỬ]")
        initial_amount = st.session_state.initial_amount
        final_amount = st.session_state.assets[-1]
        st.write(f"**Số tiền ban đầu**: ${initial_amount:,.2f}")
        st.write(f"**Số tiền đạt được sau backtest**: ${final_amount:,.2f}")
        st.write("--- Bảng các chỉ số hiệu suất ---")
        metrics = st.session_state.metrics
        df_metrics = pd.DataFrame([
            {
                "Chỉ số": k,
                "Giá trị": f"{v:.2f}" if k in [
                    "Lợi nhuận Hàng năm (%)",
                    "Lợi nhuận Tích lũy (%)",
                    "Biến động Hàng năm (%)",
                    "Sụt giảm Tối đa (Max DD)",
                    "VaR (1%)",
                    "Biên độ",
                    "Tối đa",
                    "Tối thiểu"
                ] else f"{v:.4f}"
            }
            for k, v in metrics.items()
        ])
        desired_order = ["Lợi nhuận Hàng năm (%)", "Lợi nhuận Tích lũy (%)", "Biến động Hàng năm (%)",
                        "Tỷ lệ Sharpe", "Tỷ lệ Sortino", "Sụt giảm Tối đa (Max DD)", "VaR (1%)",
                        "Biên độ", "Tối đa", "Tối thiểu", "Độ lệch (Skewness)", "Độ nhọn (Kurtosis)"]
        df_metrics = df_metrics.set_index("Chỉ số").reindex(desired_order).reset_index()
        st.table(df_metrics)
        st.write("--- Bảng Phân bổ Danh mục Trung bình Cuối kỳ (%) ---")
        st.dataframe(st.session_state.df_allocations)
        plot_final_allocation_treemap(st.session_state.df_allocations, "DAPO (Cvar 0.01 Phobert 1a.3b)")
elif st.session_state.page == "Explain & Guide":
    st.subheader(":red[Giải thích & Hướng dẫn]")
   
    try:
        st.image(QUY_TRINH_PATH, use_container_width=True)
    except FileNotFoundError:
        st.warning("quy_trinh.png không tìm thấy.")
    st.markdown("""
    ### Giải thích về kiểm soát rủi ro đo bằng cách penalty thêm rủi ro đuôi (tail risk penalty) dựa trên CVaR vào mô hình của chúng tôi
    **Vấn đề**
    Trong giao dịch chứng khoán, một chiến lược có thể tạo lợi nhuận trung bình cao, nhưng lại rất dễ thua lỗ nặng trong những ngày xấu nhất.
    Ví dụ:
    - Bình thường mỗi ngày bạn lời +1%.
    - Nhưng thỉnh thoảng lại có ngày lỗ tận -20%.
    - Nếu chỉ nhìn trung bình thì thấy “ổn”, nhưng rủi ro thật sự lại nằm ở đuôi phân phối lợi nhuận – tức những ngày cực kỳ xấu.
    **VaR (Value at Risk)**
    - VaR (5%) nghĩa là: trong 100 ngày giao dịch, có 5 ngày tệ nhất thì lỗ sẽ không vượt quá một mức nào đó.
    - Ví dụ: VaR 5% = -10% ⇒ 95 ngày bình thường thì lỗ không quá 10%.
    **CVaR (Conditional Value at Risk)**
    - CVaR đi xa hơn VaR: nó đo mức lỗ trung bình trong những ngày tệ nhất.
    - Ví dụ: nếu 5 ngày tệ nhất lần lượt lỗ: -10%, -12%, -15%, -18%, -20%
    - VaR 5% = -10%
    - CVaR 5% = (-12% -15% -18% -20%) / 4 = -16.25%
    - 👉 Tức là, nếu rơi vào “vùng rủi ro đuôi”, bạn trung bình sẽ lỗ 16.25%, nặng hơn nhiều so với chỉ nhìn VaR.
    **Tail Risk Penalty trong code**
    Trong môi trường giao dịch này:
    - Mỗi bước, hệ thống tính lợi nhuận tài khoản.
    - Sau đó ước tính CVaR trong một khoảng thời gian gần đây (ví dụ 30 ngày). Nếu CVaR cho thấy có nguy cơ thua lỗ lớn ở đuôi phân phối, thì phần thưởng (reward) sẽ bị trừ thêm một khoản penalty. Nói cách khác:
        - Chiến lược nào kiếm lời đều nhưng hay gặp cú sập mạnh ⇒ sẽ bị phạt nặng.
        - Chiến lược nào ổn định, ít rủi ro đuôi ⇒ được thưởng cao hơn.
    **Ý nghĩa**
    Mục tiêu của penalty này là:
    - Khuyến khích mô hình RL không chỉ chạy theo lợi nhuận trung bình, mà còn tránh những chiến lược liều lĩnh, dễ sập mạnh.
    - Kết quả: mô hình sẽ hướng đến lợi nhuận bền vững, ít cú sốc lớn, giống như cách các quỹ đầu tư chuyên nghiệp quản trị rủi ro.
    """)
    st.markdown("""
    ### Giải thích về DAPO
    **Lợi ích của DAPO so với PPO truyền thống**
    1. **Dynamic Sampling (lấy nhiều hành động thay vì một)**
    - Trong PPO truyền thống: mỗi trạng thái (state) chỉ được lấy một hành động rồi huấn luyện.
    - Trong DAPO: mỗi trạng thái có thể sinh ra nhiều hành động khác nhau từ cùng một policy, rồi so sánh với nhau.
    👉 Lợi ích:
    - Mô hình hiểu rõ hơn hành động nào tốt hơn trong cùng một hoàn cảnh.
    - Giảm sự “may rủi” do ngẫu nhiên (random action).
    - Học nhanh hơn và ổn định hơn.
    2. **Group Advantage (so sánh trong nhóm hành động)**
    - PPO chỉ tính lợi thế (advantage) so với baseline chung.
    - DAPO tính lợi thế tương đối giữa các hành động trong cùng một trạng thái.
    👉 Lợi ích:
    - Hành động tốt hơn trong nhóm sẽ được “khuyến khích mạnh”, còn hành động kém thì “bị phạt rõ ràng”.
    - Giúp policy học ra đường đi chính xác hơn, tránh bị mơ hồ.
    3. **Decoupled Clipping (tách biên trên/dưới khi update)**
    - PPO gốc dùng một hệ số kẹp (clipping) ±ε để tránh update quá đà.
    - DAPO tách riêng ngưỡng trên và ngưỡng dưới (epsilon_high, epsilon_low).
    👉 Lợi ích:
    - Kiểm soát tốt hơn khi nào nên “giới hạn update” (khi lợi thế quá cao hoặc quá thấp).
    - Tránh hiện tượng policy “ngừng học” do clipping quá chặt.
    - Linh hoạt hơn cho các thị trường biến động mạnh như chứng khoán.
    4. **Tích hợp Risk và Sentiment**
    - DAPO không chỉ dựa vào lợi nhuận mà còn dùng thêm risk penalty và sentiment boost.
    - Reward được điều chỉnh thông minh:
        - Risk cao → bị phạt.
        - Sentiment tích cực → được thưởng thêm.
    👉 Lợi ích:
    - Giúp mô hình thực tế hơn khi áp dụng vào tài chính (vì ngoài lợi nhuận, nhà đầu tư thật cũng cân nhắc rủi ro và tâm lý thị trường).
    - Tránh chiến lược liều lĩnh kiểu “cờ bạc”.
    5. **Huấn luyện song song (MPI + GPU)**
    - DAPO hỗ trợ huấn luyện nhiều process (multi-core, multi-GPU).
    - Đồng bộ tham số tự động qua MPI.
    👉 Lợi ích:
    - Học nhanh hơn, xử lý được khối lượng dữ liệu tài chính lớn.
    - Thích hợp khi backtest trên hàng chục năm dữ liệu hoặc nhiều thị trường.
    """)
