import random
import numpy as np
from gym import Env
from gym.spaces import Box

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

class BaseAuctionEnv(Env):
    def __init__(self, n_players, bid_dim=1, obs_dim=1):
        self.N = n_players
        self.bid_space = Box(low=np.zeros(bid_dim), high=np.ones(bid_dim), dtype=np.float32)
        self.observation_space = Box(low=np.zeros(obs_dim), high=np.ones(obs_dim), dtype=np.float32)
        self.states_shape = self.observation_space.shape

class MAFirstPriceAuctionEnv(BaseAuctionEnv):
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
        super().__init__(n_players)

    def reward_n_players(self, values, bids, r):
        """
        Calculate rewards for all players based on their private values and bids.
        
        Args:
            values (list): Private values for each player.
            bids (list): Bids made by each player.
            r (float): Risk adjustment parameter.
        
        Returns:
            list: Rewards for each player. Only the winner gets a non-zero reward.
        """
        rewards = [0] * self.N
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
            r (float): Risk adjustment parameter.
        
        Returns:
            list: Rewards for each player.
        """
        return self.reward_n_players(states, actions, r)

    def reset(self):
        """
        Reset the environment, generating new random private values for each player.
        
        Returns:
            list: New private values for each player, uniformly sampled from [0,1].
        """
        return [random.random() for _ in range(self.N)]

class MASecondPriceAuctionEnv(BaseAuctionEnv):
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
        super().__init__(n_players)

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
        rewards = [0] * self.N
        idxs = np.argsort(bids)[-2:]
        max_idx, second_max_idx = idxs[1], idxs[0]
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
        return self.reward_n_players(states, actions, r)

    def reset(self):
        """
        Reset the environment by generating new private values for each player.

        Returns:
            list: New private values for each player.
        """
        return [random.random() for _ in range(self.N)]


class MATariffDiscountEnv(BaseAuctionEnv):
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
        super().__init__(n_players)
        self.max_revenue = max_revenue

    def reward_n_players(self, costs, bids, r):
        """
        Calculate the rewards for all players based on their bids and private costs.
        The player with the highest bid wins the auction. 

        Args:
            costs (list): Private costs for each player.
            bids (list): Bids submitted by each player.
            r (float): Risk adjustment parameter.

        Returns:
            list: Rewards assigned to each player.
        """
        rewards = [0] * self.N
        idx = np.argmax(bids)
        winner_reward = self.max_revenue * (1 - bids[idx]) - costs[idx]
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
        return self.reward_n_players(states, actions, r)

    def reset(self):
        """
        Reset the environment by generating new private cost values for each player.
        Each cost is randomly sampled from a uniform distribution in the range [0, max_revenue].

        Returns:
            list: New private costs for each player.
        """
        return [random.random() * self.max_revenue for _ in range(self.N)]


class MAAllPayAuctionEnv(BaseAuctionEnv):
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
        super().__init__(n_players)

    def reward_n_players(self, values, bids, r):
        """
        Calculate the reward of all players based on their private values and bids.
        
        Args:
            values (list): Private value of each player.
            bids (list): Bids made by each player.
            r (float): Risk adjustment parameter.
        
        Returns:
            list: Reward of each player.
        """
        alpha = 0.1
        rewards = [-b - alpha for b in bids]
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
            r (float): Risk adjustment parameter.
        
        Returns:
            list: Reward of each player.
        """
        return self.reward_n_players(states, actions, r)

    def reset(self):
        """
        Reset the environment, generating new random private values for each player.
        
        Returns:
            list: New private values.
        """
        return [random.random() for _ in range(self.N)]


class MAScoreAuctionEnv(Env):
    """
    Score Auction:
    - Each agent has (value, cost)
    - Submits (bid, effort)
    - Score = effort - bid
    - Reward = value - bid - effort * cost
    """
    def __init__(self, n_players):
        self.N = n_players
        self.bid_space = Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)
        self.states_shape = (self.N, 2)

    def score_function(self, bid, effort):
        return effort - bid

    def reward_n_players(self, values_costs, actions):
        scores = [self.score_function(b, e) for (b, e) in actions]
        idx = np.argmax(scores)
        value, cost = values_costs[idx]
        bid, effort = actions[idx]
        rewards = [0.0] * self.N
        rewards[idx] = value - bid - effort * cost
        return rewards

    def step(self, states, actions):
        return self.reward_n_players(states, actions)

    def reset(self):
        return [(random.random(), random.random()) for _ in range(self.N)]