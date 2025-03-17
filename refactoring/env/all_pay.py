import random
import numpy as np
from gym import Env
from math import sqrt
from gym.spaces import Box

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

class MAAllPayAuctionEnv(Env):
    def __init__(self, n_players):
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32) # actions space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.n_players = n_players        

    def reward_n_players(self, values, bids, r):
        alpha = 0.1
        # rewards equal to negative bids list
        # rewards = [-b for b in bids]
        rewards = [-b - alpha for b in bids]
        # rewards = [-0.5 + b for b in bids]
        # for k, b in enumerate(bids):
        #     if (b < 0.01) and (values[k] > 0.5):
        #         rewards[bids.index(b)] = -1

        idx = np.argmax(bids)
        # winner_reward = values[idx] - bids[idx]
        winner_reward = values[idx] - bids[idx]
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

        # self.values = T.tensor([random.random() for _ in range(self.n_players)])
        self.values = [random.random() for _ in range(self.n_players)]
        return self.values