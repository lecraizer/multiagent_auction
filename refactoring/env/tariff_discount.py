import random
import numpy as np
from gym import Env
from math import sqrt
from gym.spaces import Box

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

class MATariffDiscountEnv(Env):
    def __init__(self, n_players, max_revenue):
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32) # actions space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.n_players = n_players
        self.max_revenue = max_revenue 

    def reward_n_players(self, costs, bids, r):
        rewards = [0]*self.n_players
        idx = np.argmax(bids)
        winner_reward = self.max_revenue*(1 - bids[idx]) - costs[idx]

        if winner_reward > 0:
            rewards[idx] = winner_reward**r
        else:
            rewards[idx] = winner_reward        
        return rewards
        
    def step(self, states, actions, r):
        rewards = self.reward_n_players(states, actions, r)
        return rewards

    def reset(self):
        # Reset state - input new random private value
        self.costs = [random.random()*self.max_revenue for _ in range(self.n_players)]
        return self.costs