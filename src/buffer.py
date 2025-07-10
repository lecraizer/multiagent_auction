# Module for setting up the replay buffer for the multi-agent DDPG algorithm

import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, num_agents=2):
        """
        Initialize the replay buffer with fixed-size memory for each component.

        Args:
            max_size (int): Maximum number of transitions to store.
            input_shape (int): Dimension of the state vector.
            n_actions (int): Dimension of the action vector.
            num_agents (int): Total number of agents in the environment.
        """
        self.mem_size = max_size
        self.mem_cntr = 0

        # Buffers for agent's own experience
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)

        # Buffers for other agents' experiences
        self.others_states = np.zeros((self.mem_size, input_shape*(num_agents-1)))
        self.others_actions = np.zeros((self.mem_size, n_actions*(num_agents-1)))


    def store_transition(self, state, action, reward, others_states, others_actions):
        """
        Store a new transition in the buffer using circular indexing.

        Args:
            state (ndarray): Current agent's observation.
            action (ndarray): Current agent's action.
            reward (float): Reward received.
            others_states (ndarray): Concatenated observations of other agents.
            others_actions (ndarray): Concatenated actions of other agents.
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.mem_cntr += 1

        self.others_states[index] = others_states
        self.others_actions[index] = others_actions


    def sample_buffer(self, batch_size):
        """
        Sample a random batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, others_states, others_actions)
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]

        others_states = self.others_states[batch]
        others_actions = self.others_actions[batch]

        return states, actions, rewards, others_states, others_actions
    

    def sample_last_buffer(self, batch_size):
        """
        Sample the most recent `batch_size` transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to retrieve.

        Returns:
            Tuple of (states, actions, rewards, others_states, others_actions)
        """
        end = self.mem_cntr
        start = max(0, end - batch_size)

        states = self.state_memory[start:end]
        actions = self.action_memory[start:end]
        rewards = self.reward_memory[start:end]
        others_states = self.others_states[start:end]
        others_actions = self.others_actions[start:end]

        return states, actions, rewards, others_states, others_actions