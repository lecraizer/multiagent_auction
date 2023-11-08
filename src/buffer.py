import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        
        self.state_memory2 = np.zeros((self.mem_size, input_shape))
        self.action_memory2 = np.zeros((self.mem_size, n_actions))

    def store_transition(self, state, action, reward, state2, action2):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.mem_cntr += 1

        self.state_memory2[index] = state2
        self.action_memory2[index] = action2

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]

        states2 = self.state_memory2[batch]
        actions2 = self.action_memory2[batch]

        return states, actions, rewards, states2, actions2