import random
import numpy as np
from gym import Env
from math import sqrt
from gym.spaces import Box

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

class MACoreSelectingAuctionEnv(Env):
    """
    Local-local-global n=3 fixed multi-item auction
    """
    def __init__(self):
        self.n_players = 3 # number of players
        self.L = 2 # number of locals
        self.G = 1 # number of globals
        self.M = 2 # number of items       

    def reward_n_players(self, values, bids, r):
        v1, v2, value_global = values[0], values[1], values[2]
        b1, b2, B = bids[0], bids[1], bids[2]
        bid_locals = b1 + b2

        payment1, payment2, payment_global = 0, 0, 0
        reward1, reward2, reward_global = 0, 0, 0
        if bid_locals > B: # locals win the auction
            if B > b2:
                payment1 = B - (bid_locals - b1)
            else:
                payment1 = 0.0
            if B > b1:
                payment2 = B - (bid_locals - b2)
            else:
                payment2 = 0.0
            
            reward1 = v1 - payment1
            reward2 = v2 - payment2
        else: # global wins the auction
            payment_global = bid_locals
            reward_global = value_global - payment_global
        
        return reward1, reward2, reward_global

    def step(self, states, actions, r):
        rewards = self.reward_n_players(states, actions, r)
        return rewards

    def reset(self):
        w = 0.0
        u = random.random()
        v1 = u*w + random.random()*(1-w)
        v2 = u*w + random.random()*(1-w)
        g = random.random()
        self.values = [v1, v2, g]
        # self.values = [random.random() for _ in range(self.n_players)]
        return self.values