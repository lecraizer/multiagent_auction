import random
import numpy as np
from gym import Env
from math import sqrt
from gym.spaces import Box

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

class MACommonPriceAuctionEnv(Env):
    def __init__(self, n_players, vl=0, vh=1, eps=0.1):
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32) # actions space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.n_players = n_players
        self.vl, self.vh = (vl, vh)
        self.eps = eps

    def argmax_with_random_tie(self, A):
        max_indices = np.where(A == np.max(A))[0]
        if len(max_indices) == 1:
            return max_indices[0]
        else:
            return random.choice(max_indices)
        
    def reward_n_players(self, common_value, bids):
        rewards = [0]*self.n_players
        idx = self.argmax_with_random_tie(bids)
        rewards[idx] = common_value - bids[idx]
        return rewards

    def step(self, state, actions):
        rewards = self.reward_n_players(state, actions)
        return rewards

    def reset(self):
        # random common value in [vl, vh]
        self.common_value = random.uniform(self.vl, self.vh)

        least = max(self.common_value - self.eps, self.vl)
        top = min(self.common_value + self.eps, self.vh)

        # list of signals in [common_value - epsilon, common_value + epsilon]
        self.signals = [random.uniform(least, top) for _ in range(self.n_players)]

        # clamp 0,1
        # self.signals = np.clip(self.signals, 0, 1)
        
        return self.common_value, self.signals