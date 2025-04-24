import random
import numpy as np
from gym import Env
from gym.spaces import Box

import warnings
with warnings.catch_warnings(): 
    warnings.filterwarnings("ignore", category=RuntimeWarning)

class MACommonPriceAuctionEnv(Env):
    """
    Implement a common-value auction environment for multiple agents.
    The item has the same intrinsic value for all bidders,
    but each bidder receives a private signal about this value.
    """
    def __init__(self, n_players, vl=0, vh=1, eps=0.1):
        """
        Initialize the common-value auction environment.
        
        Args:
            n_players (int): Number of players participating in the auction.
            vl (float, optional): Lower bound for the common value distribution. Default is 0.
            vh (float, optional): Upper bound for the common value distribution. Default is 1.
            eps (float, optional): Noise parameter determining the maximum deviation of signals. 
                                   Defaults to 0.1.
        """
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape

        self.n_players = n_players
        self.vl, self.vh = (vl, vh)
        self.eps = eps

    def argmax_with_random_tie(self, bids):
        """
        Find the index of the maximum bid, with random tie-breaking if multiple maximum bids exist.
        
        Args:
            bids (list or array): Bids made by each player.
        
        Returns:
            int: Index of the selected maximum bid.
        """
        max_indices = np.where(bids == np.max(bids))[0]
        if len(max_indices) == 1:
            return max_indices[0]
        else:
            return random.choice(max_indices)
        
    def reward_n_players(self, common_value, bids):
        """
        Calculate rewards of all players based on the common value and their bids.
        Only the highest bidder receives a reward, which is the difference between
        the common value and their bid.
        
        Args:
            common_value (float): The common value of the item.
            bids (list): Bids made by each player.
        
        Returns:
            list: Rewards for each player.
        """
        rewards = [0]*self.n_players
        idx = self.argmax_with_random_tie(bids)
        rewards[idx] = common_value - bids[idx]
        return rewards

    def step(self, state, actions):
        """
        Execute a step in the environment.
        
        Args:
            state (float): Common value of the item.
            actions (list): Bids made by each player.
        
        Returns:
            list: Rewards for each player.
        """
        rewards = self.reward_n_players(state, actions)
        return rewards

    def reset(self):
        """
        Reset the environment, generating a new common value and private signals for each player.
        The common value is randomly sampled from the interval [vl, vh].
        Each player receives a private signal about the common value, which is
        randomly sampled from [common_value - eps, common_value + eps],
        constrained to stay within [vl, vh].
        
        Returns:
            tuple:
                - common_value (float): Common value of the item.
                - signals (list): Private signals received by each player.
        """
        self.common_value = random.uniform(self.vl, self.vh)

        least = max(self.common_value - self.eps, self.vl)
        top = min(self.common_value + self.eps, self.vh)

        self.signals = [random.uniform(least, top) for _ in range(self.n_players)]
        
        return self.common_value, self.signals