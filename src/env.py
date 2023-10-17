import random
import numpy as np
from gym import Env
from gym.spaces import Discrete, Box

import warnings
# Code leading to the warning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)


class MAFirstPriceAuctionEnv(Env):
    def __init__(self, n_players):
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32) # actions space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.N = n_players        

    def reward_n_players(self, values, bids, r):
        rewards = [0]*self.N
        idx = np.argmax(bids)
        winner_reward = values[idx] - bids[idx]
        if winner_reward > 0:
            rewards[idx] = winner_reward**r
        else:
            rewards[idx] = winner_reward        
        return rewards
        
    def step(self, states, actions, r):
        rewards = self.reward_n_players(states, actions, r)

        # Return step information
        return rewards

    def reset(self):
        # Reset state - input new random private value

        # self.values = T.tensor([random.random() for _ in range(self.N)])
        self.values = [random.random() for _ in range(self.N)]
        return self.values
    

class MASecondPriceAuctionEnv(Env):
    def __init__(self, n_players):
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32) # actions space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.N = n_players        

    def reward_n_players(self, values, bids, r):
        rewards = [0]*self.N
        end_list = np.argsort(bids)[-2:]
        second_max_idx = end_list[0]
        max_idx = end_list[1]
        winner_reward = values[max_idx] - bids[second_max_idx]
        if winner_reward > 0:
            rewards[max_idx] = winner_reward**r
        else:
            rewards[max_idx] = winner_reward
        return rewards
        
    def step(self, states, actions, r):
        rewards = self.reward_n_players(states, actions, r)

        # Return step information
        return rewards

    def reset(self):
        # Reset state - input new random private value
        self.values = [random.random() for _ in range(self.N)]
        # self.values = [random.random()**2 for _ in range(self.N)]
        return self.values
    

class MATariffDiscountEnv(Env):
    def __init__(self, n_players, max_revenue):
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32) # actions space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.N = n_players
        self.max_revenue = max_revenue 


    def reward_n_players(self, costs, bids, r):
        rewards = [0]*self.N
        idx = np.argmax(bids)
        winner_reward = self.max_revenue*(1 - bids[idx]) - costs[idx]
        if winner_reward > 0:
            rewards[idx] = winner_reward**r
        else:
            rewards[idx] = winner_reward        
        return rewards
        
    def step(self, states, actions, r):
        rewards = self.reward_n_players(states, actions, r)
        
        # Return step information
        return rewards

    def reset(self):
        # Reset state - input new random private value
        self.costs = [random.random()*self.max_revenue for _ in range(self.N)]
        return self.costs


class MACommonPriceAuctionEnv(Env):
    def __init__(self, n_players, vl=0, vh=1, eps=0.1):
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32) # actions space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.N = n_players
        self.vl, self.vh = (vl, vh)
        self.eps = eps

    def argmax_with_random_tie(self, A):
        max_indices = np.where(A == np.max(A))[0]
        if len(max_indices) == 1:
            return max_indices[0]
        else:
            return random.choice(max_indices)
        
    def reward_n_players(self, common_value, bids):
        rewards = [0]*self.N
        idx = self.argmax_with_random_tie(bids)
        rewards[idx] = common_value - bids[idx]
        return rewards

    def step(self, state, actions):
        rewards = self.reward_n_players(state, actions)

        # Return step information
        return rewards

    def reset(self):

        # random common value in [vl, vh]
        self.common_value = random.uniform(self.vl, self.vh)

        least = max(self.common_value - self.eps, self.vl)
        top = min(self.common_value + self.eps, self.vh)

        # list of signals in [common_value - epsilon, common_value + epsilon]
        self.signals = [random.uniform(least, top) for _ in range(self.N)]

        # clamp 0,1
        # self.signals = np.clip(self.signals, 0, 1)
        
        return self.common_value, self.signals
    

class MAAlternativeCommonPriceAuctionEnv(Env):
    def __init__(self, n_players):
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32) # actions space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.N = n_players
    
    def argmax_with_random_tie(self, A):
        max_indices = np.where(A == np.max(A))[0]
        if len(max_indices) == 1:
            return max_indices[0]
        else:
            return random.choice(max_indices)

    def reward_n_players(self, value, bids):
        rewards = [0]*self.N
        idx = self.argmax_with_random_tie(bids)
        if bids[idx] > 0.0:
            rewards[idx] = value - bids[idx]
        return rewards
        
    def step(self, state, actions):
        rewards = self.reward_n_players(state, actions)

        # Return step information
        return rewards

    def reset(self):
        signals = [random.random() for _ in range(self.N)]

        # common value is the sum of the signals
        common_value = sum(signals)
        return common_value, signals
    

class MASecondCommonPriceAuctionEnv(Env):
    def __init__(self, n_players):
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32) # actions space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.N = n_players
    
    def argmax_with_random_tie(self, A):
        max_indices = np.where(A == np.max(A))[0]
        if len(max_indices) == 1:
            return max_indices[0]
        else:
            return random.choice(max_indices)

    def reward_n_players(self, value, bids):
        rewards = [0]*self.N
        end_list = np.argsort(bids)[-2:]
        second_max_idx, max_idx = end_list[0], end_list[1]
        rewards[max_idx] = value - bids[second_max_idx]
        return rewards
        
    def step(self, state, actions):
        rewards = self.reward_n_players(state, actions)

        # Return step information
        return rewards

    def reset(self):
        signals = [random.random() for _ in range(self.N)]

        # common value is the sum of the signals
        common_value = sum(signals)
        return common_value, signals
    

class MAAllPayAuctionEnv(Env):
    def __init__(self, n_players):
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32) # actions space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.N = n_players        

    def reward_n_players(self, values, bids, r):
        # rewards equal to negative bids list
        rewards = [-b for b in bids]
        idx = np.argmax(bids)
        winner_reward = values[idx] - bids[idx]
        if winner_reward > 0:
            rewards[idx] = winner_reward**r
        else:
            rewards[idx] = winner_reward        
        return rewards
        
    def step(self, states, actions, r):
        rewards = self.reward_n_players(states, actions, r)

        # Return step information
        return rewards

    def reset(self):
        # Reset state - input new random private value

        # self.values = T.tensor([random.random() for _ in range(self.N)])
        self.values = [random.random() for _ in range(self.N)]
        return self.values
    

# class MAAssymetricFirstPriceAuctionEnv(Env):
#     def __init__(self, n_players):
#         self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32) # actions space
#         self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
#         self.states_shape = self.observation_space.shape
#         self.N = n_players        

#     def reward_n_players(self, values, bids, r):
#         rewards = [0]*self.N
#         idx = np.argmax(bids)
#         winner_reward = values[idx] - bids[idx]
#         if winner_reward > 0:
#             rewards[idx] = winner_reward**r[idx]
#         else:
#             rewards[idx] = winner_reward        
#         return rewards
        
#     def step(self, states, actions, r):
#         rewards = self.reward_n_players(states, actions, r)

#         # Return step information
#         return rewards

#     def reset(self):
#         # Reset state - input new random private value

#         # self.values = T.tensor([random.random() for _ in range(self.N)])
#         self.values = [random.random() for _ in range(self.N)]
#         return self.values
    


class MAAssymetricFirstPriceAuctionEnv(Env):
    def __init__(self, n_players):
        self.bid_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32) # actions space
        self.observation_space = Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.states_shape = self.observation_space.shape
        self.N = n_players        

    def reward_n_players(self, values, bids, r):
        rewards = [0]*self.N
        idx = np.argmax(bids)
        winner_reward = values[idx] - bids[idx]
        if winner_reward > 0:
            rewards[idx] = winner_reward**r
        else:
            rewards[idx] = winner_reward        
        return rewards
        
    def step(self, states, actions, r):
        rewards = self.reward_n_players(states, actions, r)

        # Return step information
        return rewards

    def reset(self):
        # Reset state - input new random private value
        self.values = [random.random(), random.random()*2]
        return self.values