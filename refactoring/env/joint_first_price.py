import random
import numpy as np
from gym import Env
from math import sqrt
from gym.spaces import Box

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

class MAJointFirstPriceAuctionEnv(Env):
    def __init__(self, n_players, mean=0.5, cov=0.1):
        super(MAJointFirstPriceAuctionEnv, self).__init__()
        self.n_players = n_players
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)  # actions space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
             
    def reward_n_players(self, values, bids, r):
        rewards = [0] * self.n_players
        idx = np.argmax(bids)
        winner_reward = values[idx] - bids[idx]
        if winner_reward > 0:
            rewards[idx] = winner_reward**r
        else:
            rewards[idx] = winner_reward        
        return rewards
        
    def step(self, states, actions, r=1):
        rewards = self.reward_n_players(states, actions, r)
        return rewards

    def reset(self):
        # Sample private values from a joint uniform distribution
        u = random.random()
        x = (-1 + sqrt(1 + 8*u))/2
        v = random.random()
        y = (-1 + sqrt(1+8*x*v*(1+2.0*x)))/(4.0*x)
        self.values = [x, y]
        return self.values