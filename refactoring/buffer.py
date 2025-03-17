# Module for setting up the replay buffer for the multi-agent DDPG algorithm

import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, num_agents=2):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)

        self.others_states = np.zeros((self.mem_size, input_shape*(num_agents-1)))
        self.others_actions = np.zeros((self.mem_size, n_actions*(num_agents-1)))

    def get_values(self, idx):
        states = self.state_memory[idx]
        actions = self.action_memory[idx]
        rewards = self.reward_memory[idx]

        others_states = self.others_states[idx]
        others_actions = self.others_actions[idx]

        return states, actions, rewards, others_states, others_actions

    def store_transition(self, state, action, reward, others_states, others_actions):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.mem_cntr += 1

        self.others_states[index] = others_states
        self.others_actions[index] = others_actions

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
    
        return self.get_values(batch)
    
    def sample_last_buffer(self, batch_size):
        '''
        Sample the last batch_size elements of the buffer
        '''
        if self.mem_cntr < batch_size: batch_size = self.mem_cntr 

        return self.get_values(range(self.mem_cntr-batch_size, self.mem_cntr))