# Description: Environment classes for multi-agent auction games

import random
import numpy as np
from gym import Env
from gym.spaces import Box
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)


class BaseAuctionEnv(Env):
    def __init__(self, n_players, bid_dim=1, obs_dim=1):
        self.N = n_players
        self.bid_space = Box(low=np.zeros(bid_dim), high=np.ones(bid_dim), dtype=np.float32)
        self.observation_space = Box(low=np.zeros(obs_dim), high=np.ones(obs_dim), dtype=np.float32)
        self.states_shape = self.observation_space.shape


class MAFirstPriceAuctionEnv(BaseAuctionEnv):
    def __init__(self, n_players):
        super().__init__(n_players)

    def reward_n_players(self, values, bids, r, t):
        rewards = [0] * self.N
        idx = np.argmax(bids)
        winner_reward = values[idx] - bids[idx]
        rewards[idx] = winner_reward**r if winner_reward > 0 else winner_reward
        return rewards

    def step(self, states, actions, r, t):
        return self.reward_n_players(states, actions, r, t)

    def reset(self):
        return [random.random() for _ in range(self.N)]


class MASecondPriceAuctionEnv(BaseAuctionEnv):
    def __init__(self, n_players):
        super().__init__(n_players)

    def reward_n_players(self, values, bids, r, t):
        rewards = [0] * self.N
        idxs = np.argsort(bids)[-2:]
        max_idx, second_max_idx = idxs[1], idxs[0]
        winner_reward = values[max_idx] - bids[second_max_idx]
        rewards[max_idx] = winner_reward**r if winner_reward > 0 else winner_reward
        return rewards

    def step(self, states, actions, r, t):
        return self.reward_n_players(states, actions, r, t)

    def reset(self):
        return [random.random() for _ in range(self.N)]


class MATariffDiscountEnv(BaseAuctionEnv):
    def __init__(self, n_players, max_revenue):
        super().__init__(n_players)
        self.max_revenue = max_revenue

    def reward_n_players(self, costs, bids, r):
        rewards = [0] * self.N
        idx = np.argmax(bids)
        winner_reward = self.max_revenue * (1 - bids[idx]) - costs[idx]
        rewards[idx] = winner_reward**r if winner_reward > 0 else winner_reward
        return rewards

    def step(self, states, actions, r):
        return self.reward_n_players(states, actions, r)

    def reset(self):
        return [random.random() * self.max_revenue for _ in range(self.N)]


class MAAllPayAuctionEnv(BaseAuctionEnv):
    def __init__(self, n_players):
        super().__init__(n_players)

    def reward_n_players(self, values, bids, r, t):
        alpha = 0.1
        rewards = [-b - alpha for b in bids]
        idx = np.argmax(bids)
        winner_reward = values[idx] - bids[idx]
        rewards[idx] = winner_reward**r if winner_reward > 0 else winner_reward
        return rewards

    def step(self, states, actions, r, t):
        return self.reward_n_players(states, actions, r, t)

    def reset(self):
        return [random.random() for _ in range(self.N)]


class MAPartialAllPayAuctionEnv(BaseAuctionEnv):
    '''
    All-Pay Auction with a parameter t:
    - t = 0: First-Price Auction
    - t = 1: All-Pay Auction
    - 0 < t < 1: Hybrid Auction
    '''
    def __init__(self, n_players):
        super().__init__(n_players)

    def reward_n_players(self, values, bids, r, t):
        rewards = [-b*t for b in bids]
        idx = np.argmax(bids)
        winner_reward = values[idx] - bids[idx]
        rewards[idx] = winner_reward**r if winner_reward > 0 else winner_reward
        return rewards

    def step(self, states, actions, r, t):
        return self.reward_n_players(states, actions, r, t)
    
    def reset(self):
        return [random.random() for _ in range(self.N)]
    


class MAScoreAuctionEnv(Env):
    """
    Score Auction:
    - Each agent has (value, cost)
    - Submits (bid, effort)
    - Score = effort - bid
    - Reward = value - bid - effort * cost
    """
    def __init__(self, n_players):
        self.N = n_players
        self.bid_space = Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)
        self.states_shape = (self.N, 2)

    def score_function(self, bid, effort):
        return effort - bid

    def reward_n_players(self, values_costs, actions):
        scores = [self.score_function(b, e) for (b, e) in actions]
        idx = np.argmax(scores)
        value, cost = values_costs[idx]
        bid, effort = actions[idx]
        rewards = [0.0] * self.N
        rewards[idx] = value - bid - effort * cost
        return rewards

    def step(self, states, actions):
        return self.reward_n_players(states, actions)

    def reset(self):
        return [(random.random(), random.random()) for _ in range(self.N)]
