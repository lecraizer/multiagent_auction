import random
import numpy as np
from gym import Env
from gym.spaces import Box

import warnings
with warnings.catch_warnings(): 
    warnings.filterwarnings("ignore", category=RuntimeWarning)

class MAAllPayAuctionEnv(Env):
    """
    Implement an all-pay auction environment for multiple agents.
    All participants must pay their bids regardless of whether they win 
    or not, but only the highest bidder wins the item.
    """
    def __init__(self, n_players):
        """
        Initialize the all-pay auction environments.
        
        Args:
            n_players (int): Number of players participating in the auction.
        """
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.n_players = n_players        

    def reward_n_players(self, values, bids, r):
        """
        Calculate the reward of all players based on their private values and bids.
        The player with the highest bid wins the item and receives a reward based on the
        difference between their private value and their bid, raised to the power of r.
        
        Args:
            values (list): Private value of each player.
            bids (list): Bids made by each player.
            r (float): Scaling parameter for the winner's reward.
        
        Returns:
            list: Reward of each player.
        """
        alpha = 0.1
        rewards = [-b - alpha for b in bids] # rewards equal to negative bids minus alpha

        idx = np.argmax(bids)
        winner_reward = values[idx] - bids[idx]
        rewards[idx] = winner_reward**r if winner_reward > 0 else winner_reward  
        return rewards
        
    def step(self, states, actions, r):
        """
        Execute a step in the environment.
        
        Args:
            states (list): Private values of each player.
            actions (list): Bids made by each player.
            r (float): Scaling parameter for the winner's reward.
        
        Returns:
            list: Reward of each player.
        """
        rewards = self.reward_n_players(states, actions, r)
        return rewards

    def reset(self):
        """
        Reset the environment, generating new random private values for each player.
        
        Returns:
            list: New private values.
        """
        self.values = [random.random() for _ in range(self.n_players)]
        return self.values