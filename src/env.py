import random
import numpy as np

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

class BaseAuctionEnv():
    """
    Implement a base auction environment.
    """
    def __init__(self, n_players:int, lower_bound:float=0, upper_bound:float=1):
        """
        Initialize the auction environment.

        Args:
            n_players (int): Number of players participating in the auction.
            lower_bound (float): Lower bound of action space.
            upper_bound (float): Upper bound of action space.
        """
        self.n_players = n_players
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
    
    def value_paid(self, own_bid: float, bids: float):
        return own_bid

class MAFirstPriceAuctionEnv(BaseAuctionEnv):
    """
    Implement a first-price sealed-bid auction environment.
    The highest bidder wins the item and pays the amount they bid.
    All other bidders pay nothing and receive no reward.
    """
    def __init__(self, n_players: int) -> None:
        """
        Initialize the first-price auction environment.

        Args:
            n_players (int): Number of players participating in the auction.
        """
        super().__init__(n_players)

    def step(self, values: list[float], bids: list[float], r: float, t: float) -> list[float]:
        """
        Calculate rewards for all players based on their private values and bids.
        
        Args:
            values (list): Private values for each player.
            bids (list): Bids made by each player.
            r (float): Risk adjustment parameter.
        
        Returns:
            list: Rewards for each player. Only the winner gets a non-zero reward.
        """
        rewards = [0] * self.n_players
        idx = np.argmax(bids)
        winner_reward = values[idx] - bids[idx]
        rewards[idx] = winner_reward**r if winner_reward > 0 else winner_reward
        return rewards

    def reset(self) -> list[float]:
        """
        Reset the environment, generating new random private values for each player.
        
        Returns:
            list: New private values for each player, uniformly sampled from [0,1].
        """
        return [random.random() for _ in range(self.n_players)]

class MASecondPriceAuctionEnv(BaseAuctionEnv):
    """
    Implement a second-price auction environment. 
    The agent with the highest bid wins the auction and pays the second-highest bid, 
    receiving a reward based on the difference between their private value and the 
    second-highest bid.
    """
    def __init__(self, n_players: int) -> None:
        """
        Initialize the second-price auction environment.

        Args:
            n_players (int): Number of players participating in the auction.
        """
        super().__init__(n_players)

    def step(self, values: list[float], bids: list[float], r: float, t: float) -> list[float]:
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
        rewards = [0] * self.n_players
        idxs = np.argsort(bids)[-2:]
        max_idx, second_max_idx = idxs[1], idxs[0]
        winner_reward = values[max_idx] - bids[second_max_idx]
        rewards[max_idx] = winner_reward**r if winner_reward > 0 else winner_reward
        return rewards

    def reset(self) -> list[float]:
        """
        Reset the environment by generating new private values for each player.

        Returns:
            list: New private values for each player.
        """
        return [random.random() for _ in range(self.n_players)]
    
    def value_paid(self, own_bid, bids):
        bids_below_own_bid = [b for b in bids if b < own_bid]
        if not bids_below_own_bid: return 0.0
        return np.mean(bids_below_own_bid)


class MATariffDiscountEnv(BaseAuctionEnv):
    """
    Implement a first-price auction environment. 
    The agent with the highest bid wins the auction
    and receives a reward based on the difference between maximum revenue and 
    their bid, adjusted by their costs.
    """
    def __init__(self, n_players: int, max_revenue: float) -> None:
        """
        Initialize the tariff discount auction environment.

        Args:
            n_players (int): Number of players participating in the auction.
            max_revenue (float): Maximum revenue that can be achieved in the auction.
        """
        super().__init__(n_players)
        self.max_revenue = max_revenue

    def step(self, costs: list[float], bids: list[float], r: float) -> list[float]:
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
        rewards = [0] * self.n_players
        idx = np.argmax(bids)
        winner_reward = self.max_revenue * (1 - bids[idx]) - costs[idx]
        rewards[idx] = winner_reward**r if winner_reward > 0 else winner_reward
        return rewards

    def reset(self) -> list[float]:
        """
        Reset the environment by generating new private cost values for each player.
        Each cost is randomly sampled from a uniform distribution in the range [0, max_revenue].

        Returns:
            list: New private costs for each player.
        """
        return [random.random() * self.max_revenue for _ in range(self.n_players)]


class MAAllPayAuctionEnv(BaseAuctionEnv):
    """
    Implement an all-pay auction environment for multiple agents.
    All participants must pay their bids regardless of whether they win 
    or not, but only the highest bidder wins the item.
    """
    def __init__(self, n_players: int) -> None:
        """
        Initialize the all-pay auction environments.
        
        Args:
            n_players (int): Number of players participating in the auction.
        """
        super().__init__(n_players)

    def step(self, values: list[float], bids: list[float], r: float, t: float) -> list[float]:
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

    def reset(self) -> list[float]:
        """
        Reset the environment, generating new random private values for each player.
        
        Returns:
            list: New private values.
        """
        return [random.random() for _ in range(self.n_players)]


class MAPartialAllPayAuctionEnv(BaseAuctionEnv):
    """
    Implement a partial all-pay auction environment for multiple agents.
    This environment generalizes between first-price and all-pay auctions
    using a parameter `t`:

    - t = 0: Equivalent to a first-price auction (only the winner pays).
    - t = 1: Equivalent to an all-pay auction (all players pay their bids).
    - 0 < t < 1: Hybrid auction where players partially pay their bids
      according to the value of `t`.
    """
    def __init__(self, n_players: int) -> None:
        """
        Initialize the partial all-pay auction environment.

        Args:
            n_players (int): Number of players participating in the auction.
        """
        super().__init__(n_players)

    def step(self, values: list[float], bids: list[float], r: float, t: float) -> list[float]:
        """
        Calculate the rewards for all players based on their bids and private values.
        The parameter `t` controls how much of each bid is paid regardless of winning.

        Args:
            values (list): Private values for each player.
            bids (list): Bids submitted by each player.
            r (float): Risk adjustment parameter.
            t (float): Payment parameter controlling the fraction of the bid
                       paid by all players (0 = first-price, 1 = all-pay).

        Returns:
            list: Rewards assigned to each player.
        """
        rewards = [-b*t for b in bids]
        idx = np.argmax(bids)
        winner_reward = values[idx] - bids[idx]
        rewards[idx] = winner_reward**r if winner_reward > 0 else winner_reward
        return rewards
    
    def reset(self) -> list[float]:
        """
        Reset the environment by generating new private values for each player.

        Returns:
            list: New private values for each player, uniformly sampled from [0,1].
        """
        return [random.random() for _ in range(self.n_players)]

class MACustomAuctionEnv(BaseAuctionEnv):
    """
    Implement a custom auction.
    """
    def __init__(self, n_players: int) -> None:
        """
        Initialize the auction environment.

        Args:
            n_players (int): Number of players participating in the auction.
        """
        super().__init__(n_players)

    def step(self, values: list[float], bids: list[float], r: float, t: float) -> list[float]:
        """
        Calculate the rewards for all players based on their bids and private values.

        Args:
            values (list): Private values for each player.
            bids (list): Bids submitted by each player.
            r (float): Risk adjustment parameter.
            t (float): Payment parameter controlling the fraction of the bid
                       paid by all players (0 = first-price, 1 = all-pay).

        Returns:
            list: Rewards assigned to each player.
        """
        rewards = [0] * self.n_players
        idx = np.argmax(bids)
        winner_reward = values[idx] - bids[idx]
        rewards[idx] = winner_reward**r if winner_reward > 0 else winner_reward
        return rewards
    
    def reset(self) -> list[float]:
        """
        Reset the environment by generating new private values for each player.

        Returns:
            list: New private values for each player.
        """
        return [random.random() for _ in range(self.n_players)]