import random
import numpy as np
from gym import Env
from gym.spaces import Box

import warnings
with warnings.catch_warnings(): 
    warnings.filterwarnings("ignore", category=RuntimeWarning)

class MAFirstPriceAuctionEnv(Env):
    """
    Implement a first-price sealed-bid auction environment.
    The highest bidder wins the item and pays the amount they bid.
    All other bidders pay nothing and receive no reward.
    """
    def __init__(self, n_players):
        """
        Initialize the first-price auction environment.
        
        Args:
            n_players (int): Number of players participating in the auction.
        """
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.n_players = n_players        

    def reward_n_players(self, values, bids, r):
        """
        Calculate rewards for all players based on their private values and bids.
        
        Args:
            values (list): Private values for each player.
            bids (list): Bids made by each player.
            r (float): Scaling parameter for the winner's reward when positive.
        
        Returns:
            list: Rewards for each player. Only the winner gets a non-zero reward.
        """
        rewards = [0]*self.n_players
        idx = np.argmax(bids)
        winner_reward = values[idx] - bids[idx]
        rewards[idx] = winner_reward**r if winner_reward > 0 else winner_reward        
        return rewards
        
    def step(self, states, actions, r):
        """
        Execute a step in the environment.
        
        Args:
            states (list): Private values for each player.
            actions (list): Bids made by each player.
            r (float): Scaling parameter for the winner's reward when positive.
        
        Returns:
            list: Rewards for each player.
        """
        rewards = self.reward_n_players(states, actions, r)
        return rewards

    def reset(self):
        """
        Reset the environment, generating new random private values for each player.
        
        Returns:
            list: New private values for each player, uniformly sampled from [0,1].
        """
        self.values = [random.random() for _ in range(self.n_players)]
        return self.values
