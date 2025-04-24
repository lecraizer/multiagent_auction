import random
import numpy as np
from gym import Env
from math import sqrt
from gym.spaces import Box

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

class MAJointFirstPriceAuctionEnv(Env):
    """
    Implement a first-price auction environment. The agent with the highest bid wins the auction
    and receives a reward based on the difference between their private value and
    their bid. The private values of players are sampled from a joint uniform distribution.
    """
    def __init__(self, n_players, mean=0.5, cov=0.1):
        """
        Initialize the joint first-price auction environment.
        Args:
            n_players (int): Number of players participating in the auction.
            mean (float, optional): Mean of the distribution. Defaults is 0.5.
            cov (float, optional): Covariance parameter for the distribution. Defaults is 0.1.
        """
        super(MAJointFirstPriceAuctionEnv, self).__init__()
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.n_players = n_players
             
    def reward_n_players(self, values, bids, r):
        """
        Calculate the reward for all players.
        The player with the highest bid wins the auction.

        Args:
            values (list): Private values for each player.
            bids (list): Bids submitted by each player.
            r (float): Risk adjustment parameter.

        Returns:
            list: Rewards assigned to each player.
        """
        rewards = [0] * self.n_players
        idx = np.argmax(bids)
        winner_reward = values[idx] - bids[idx]
        rewards[idx] = winner_reward**r if winner_reward > 0 else winner_reward        
        return rewards
        
    def step(self, states, actions, r=1):
        """
        Execute a step in the environment.

        Args:
            states (list): Private values for each player.
            actions (list): Bids submitted by each player.
            r (float, optional): Risk adjustment parameter. Defaults is 1.

        Returns:
            list: Rewards for each player.
        """
        rewards = self.reward_n_players(states, actions, r)
        return rewards

    def reset(self):
        """
        Reset the environment by generating new private values for each player.
        Values are sampled from a joint uniform distribution.

        Returns:
            list: New private values for each player.
        """
        u = random.random()
        v = random.random()
        x = (-1 + sqrt(1 + 8*u))/2
        y = (-1 + sqrt(1+8*x*v*(1+2.0*x)))/(4.0*x)
        self.values = [x, y]
        return self.values