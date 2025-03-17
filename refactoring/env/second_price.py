import random
import numpy as np
from gym import Env
from math import sqrt
from gym.spaces import Box

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

class MASecondPriceAuctionEnv(Env):
    def __init__(self, n_players):
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32) # actions space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.n_players = n_players        

    def reward_n_players(self, values, bids, r):
        rewards = [0]*self.n_players
        end_list = np.argsort(bids)[-2:]
        second_max_idx = end_list[0]
        max_idx = end_list[1]
        winner_reward = values[max_idx] - bids[second_max_idx]

        if winner_reward > 0:
            rewards[max_idx] = winner_reward**r
        else:
            rewards[max_idx] = winner_reward
        return rewards
        
    def step(self, states, actions, r):
        rewards = self.reward_n_players(states, actions, r)
        return rewards

    def reset(self):
        # Reset state - input new random private value
        self.values = [random.random() for _ in range(self.n_players)]
        return self.values