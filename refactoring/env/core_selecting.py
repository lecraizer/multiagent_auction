import random
import numpy as np
from gym import Env
from gym.spaces import Box

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

class MACoreSelectingAuctionEnv(Env):
    """
    Local-local-global n=3 fixed multi-item auction environment.
    
    Models an auction with three players: two local bidders who can each win one item,
    and one global bidder who wants both items. 
    """
    def __init__(self):
        """
        Initialize the auction environment with fixed parameters.
        
        The environment is set up with three players: two local bidders and one global bidder.
        There are two items being auctioned, where local bidders want one item each and
        the global bidder wants both items.
        """
        self.n_players = 3
        self.L = 2 # number of local bidders
        self.G = 1 # number of global bidders
        self.M = 2 # number of items       

    def reward_n_players(self, values, bids, r):
        """
        Calculate rewards for all players based on their private values and bids.
        The local bidders win if their combined bid exceeds the global bidder's bid. 
        
        Args:
            values (list): Private values for each player [v1, v2, value_global].
                - v1: First local bidder's value.
                - v2: Second local bidder's value.
                - value_global: Global bidder's value.
            bids (list): Bids made by each player [b1, b2, B].
                b1: First local bidder's bid.
                b2: Second local bidder's bid.
                B: Global bidder's bid.
        
        Returns:
            tuple: Rewards (reward1, reward2, reward_global) for each player.
        """
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
        """
        Execute a step in the environment.
        
        Args:
            states (list): Private values for each player [v1, v2, g].
            actions (list): Bids made by each player [b1, b2, B].
        
        Returns:
            tuple: Rewards (reward1, reward2, reward_global) for each player.
        """
        rewards = self.reward_n_players(states, actions, r)
        return rewards

    def reset(self):
        """
        Reset the environment, generating new random private values for each player.
        
        The values for local bidders v1 and v2 are generated with a correlation parameter w.
        The global bidder's value g is independently drawn from a uniform distribution.
        
        Returns:
            list: New private values [v1, v2, g] for each player.
        """
        w = 0.0
        u = random.random()
        v1 = u*w + random.random()*(1-w)
        v2 = u*w + random.random()*(1-w)
        g = random.random()
        self.values = [v1, v2, g]

        return self.values