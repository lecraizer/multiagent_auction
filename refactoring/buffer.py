# Module for setting up the replay buffer for the multi-agent DDPG algorithm

import numpy as np

class ReplayBuffer(object):
    """
    Implements a replay buffer for storing agent interactions with the environment.
    """
    def __init__(self, max_size, input_shape, n_actions, num_agents=2):
        """
        Initializes the buffer.

        Args:
            max_size (int): The maximum size of the memory buffer.
            input_shape (int): The size of the input states.
            n_actions (int): The number of possible actions for the agent.
            num_agents (int, optional): The number of agents interacting in the environment (default is 2).
        """
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)

        self.others_states = np.zeros((self.mem_size, input_shape*(num_agents-1)))
        self.others_actions = np.zeros((self.mem_size, n_actions*(num_agents-1)))

    def get_values(self, idx):
        """
        Retrieves the stored values (states, actions, rewards, others_states, others_actions) 
        from memory at the specified indice.

        Parameters:
            idx (int or list): The indices of the values to retrieve.

        Returns:
            tuple: A tuple containing states, actions, rewards, others_states and others_actions.
        """
        states = self.state_memory[idx]
        actions = self.action_memory[idx]
        rewards = self.reward_memory[idx]

        others_states = self.others_states[idx]
        others_actions = self.others_actions[idx]

        return states, actions, rewards, others_states, others_actions

    def store_transition(self, state, action, reward, others_states, others_actions):
        """
        Stores state, action, reward, others_states and others_actions in the memory buffer.

        Parameters:
            state (np.ndarray): The current state.
            action (np.ndarray): The action taken in the current state.
            reward (float): The reward received after taking the action.
            others_states (np.ndarray): The states of other entities in the environment.
            others_actions (np.ndarray): The actions of other entities in the environment.

        The memory counter is updated after storing the transition.
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward

        self.others_states[index] = others_states
        self.others_actions[index] = others_actions
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """
        Samples a random subset of the memory buffer.

        Parameters:
            batch_size (int): The number of elements to sample.

        Returns:
            tuple: The set of values sampled, which is limited by mem_cntr or mem_size.
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
    
        return self.get_values(batch)
    
    def sample_last_buffer(self, batch_size):
        """
        Samples the last batch_size elements from the buffer.

        Parameters:
            batch_size (int): The number of elements to sample.

        Returns:
            tuple: The last batch_size values (states, actions, rewards, other agents' states, other agents' actions).
        """
        if self.mem_cntr < batch_size: batch_size = self.mem_cntr 

        return self.get_values(range(self.mem_cntr-batch_size, self.mem_cntr))