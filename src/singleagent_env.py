import random
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box


class FirstPriceAuctionEnv(Env):
    def __init__(self, n_players):
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32) # actions space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.N = n_players

    def reward_n_players(self, own_bid, others_bids):
        if own_bid >= max(others_bids):
            reward = self.value - own_bid
        else:
            reward = 0
        return reward
    
    def step(self, action):
        others_bids = [random.random() for _ in range(self.N-1)]
        bid = action[0]
        reward = self.reward_n_players(bid, others_bids)

        # Return step information
        return reward

    def reset(self):
        # Reset state - input new random private value
        self.value = random.random()
        return self.value
    

class SecondPriceAuctionEnv(Env):
    def __init__(self, n_players):
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32) # actions space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.N = n_players

    def reward_n_players(self, own_bid, others_bids):
        best_bid = max(others_bids)
        if own_bid >= best_bid:
            reward = self.value - best_bid
        else:
            reward = 0
        return reward
    
    def step(self, action):
        others_bids = [random.random() for _ in range(self.N-1)]
        bid = action[0]
        reward = self.reward_n_players(bid, others_bids)
        
        # End episode
        done = True
        info = {} # set placeholder for info

        # Return step information
        return self.value, reward, done, info
    
    def reset(self):
        # Reset state - input new random private value
        self.value = random.random()
        return self.value
    

class AllPayAuctionEnv(Env):
    def __init__(self, n_players):
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32) # actions space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        # self.value = random.random()
        self.N = n_players

    def reward_n_players(self, own_bid, others_bids):
        if own_bid >= max(others_bids):
            reward = self.value - own_bid
        else:
            reward = -own_bid
        return reward
    
    def step(self, action):
        others_bids = [random.random() for _ in range(self.N-1)]
        # others_bids = [(self.value**2.0)/2.0 for _ in range(self.N-1)]
        # others_bids = [0.0 for _ in range(self.N-1)]

        bid = action[0]
        reward = self.reward_n_players(bid, others_bids)
        
        # Return step information
        return reward

    def reset(self):
        # Reset state - input new random private value
        self.value = random.random()
        return self.value