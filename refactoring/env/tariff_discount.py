import random
import numpy as np
from gym import Env
from gym.spaces import Box

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

class MATariffDiscountEnv(Env):
    """
    Implement a first-price auction environment. The agent with the highest bid wins the auction
    and receives a reward based on the difference between maximum revenue and 
    their bid, adjusted by their costs.
    """
    def __init__(self, n_players, max_revenue):
        """
        Initialize the tariff discount auction environment.

        Args:
            n_players (int): Number of players participating in the auction.
            max_revenue (float): Maximum revenue that can be achieved in the auction.
        """
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.n_players = n_players
        self.max_revenue = max_revenue 

    def reward_n_players(self, costs, bids, r):
        """
        Calculate the rewards for all players based on their bids and private costs.
        The player with the highest bid wins the auction. 

        Args:
            costs (list): Private costs for each player.
            bids (list): Bids submitted by each player.

        Returns:
            list: Rewards assigned to each player.
        """
        rewards = [0]*self.n_players
        idx = np.argmax(bids)
        winner_reward = self.max_revenue*(1 - bids[idx]) - costs[idx]
        rewards[idx] = winner_reward**r if winner_reward > 0 else winner_reward       
        return rewards
        
    def step(self, states, actions, r):
        """
        Execute a step in the environment.

        Args:
            states (list): Private cost values for each player.
            actions (list): Bids submitted by each player.

        Returns:
            list: Rewards for each player.
        """
        rewards = self.reward_n_players(states, actions, r)
        return rewards

    def reset(self):
        """
        Reset the environment by generating new private cost values for each player.
        Each cost is randomly sampled from a uniform distribution in the range [0, max_revenue].

        Returns:
            list: New private costs for each player.
        """
        self.costs = [random.random()*self.max_revenue for _ in range(self.n_players)]
        return self.costs