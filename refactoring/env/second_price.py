import random
import numpy as np
from gym import Env
from gym.spaces import Box

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

class MASecondPriceAuctionEnv(Env):
    """
    Implement a second-price auction environment. The agent with the highest bid wins the auction
    and pays the second-highest bid, receiving a reward based on the difference between 
    their private value and the second-highest bid.
    """
    def __init__(self, n_players):
        """
        Initialize the second-price auction environment.

        Args:
            n_players (int): Number of players participating in the auction.
        """
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.n_players = n_players        

    def reward_n_players(self, values, bids, r):
        """
        Calculate the rewards for all players based on their bids and private values.
        The player with the highest bid wins the auction but pays the second-highest bid.

        Args:
            values (list): Private values for each player.
            bids (list): Bids submitted by each player.
            r (float): Risk adjustment parameter.

        Returns:
            list: Rewards assigned to each player.
        """
        rewards = [0]*self.n_players
        end_list = np.argsort(bids)[-2:]
        second_max_idx, max_idx = end_list[0], end_list[1]
        winner_reward = values[max_idx] - bids[second_max_idx]
        rewards[max_idx] = winner_reward**r if winner_reward > 0 else winner_reward
        return rewards
        
    def step(self, states, actions, r):
        """
        Execute a step in the environment.

        Args:
            states (list): Private values for each player.
            actions (list): Bids submitted by each player.
            r (float): Risk adjustment parameter.

        Returns:
            list: Rewards for each player.
        """
        rewards = self.reward_n_players(states, actions, r)
        return rewards

    def reset(self):
        """
        Reset the environment by generating new private values for each player.
        Each value is independently sampled from a uniform distribution in the range [0, 1].

        Returns:
            list: New private values for each player.
        """
        self.values = [random.random() for _ in range(self.n_players)]
        return self.values