import numpy as np

class ReplayBuffer(object):
    """
    Implements a replay buffer for storing agent interactions with the environment.
    """
    def __init__(self, max_size: int, input_shape: int, n_actions: int, num_agents: int = 2) -> None:
        """
        Initialize the replay buffer with fixed-size memory for each component.

        Args:
            max_size (int): Maximum number of transitions to store.
            input_shape (int): Dimension of the state vector.
            n_actions (int): Dimension of the action vector.
            num_agents (int, optional): Total number of agents in the environment (default is 2).
        """
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)

        self.others_states = np.zeros((self.mem_size, input_shape*(num_agents-1)))
        self.others_actions = np.zeros((self.mem_size, n_actions*(num_agents-1)))

    def get_values(self, idx: int | list) -> tuple:
        """
        Retrieves the stored values (states, actions, rewards, others_states, others_actions) 
        from memory at the specified indice.

        Args:
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

    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float, others_states: np.ndarray, 
                         others_actions: np.ndarray) -> None:
        """
        Stores state, action, reward, others_states and others_actions in the memory buffer.

        Args:
            state (np.ndarray): Current agent's observation.
            action (np.ndarray): Current agent's action.
            reward (float): Reward received after taking the action.
            others_states (np.ndarray): Concatenated observations of other agents.
            others_actions (np.ndarray): Concatenated actions of other agents.

        The memory counter is updated after storing the transition.
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward

        self.others_states[index] = others_states
        self.others_actions[index] = others_actions
        self.mem_cntr += 1

    def sample_buffer(self, batch_size: int) -> tuple:
        """
        Samples a random subset of the memory buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: The set of values sampled, which is limited by mem_cntr or mem_size.
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
    
        return self.get_values(batch)
    
    def sample_last_buffer(self, batch_size: int) -> tuple:
        """
        Samples the last batch_size elements from the buffer.

        Args:
            batch_size (int): Number of transitions to retrieve.

        Returns:
            tuple: The last batch_size values (states, actions, rewards, other agents' states, other agents' actions).
        """
        if self.mem_cntr < batch_size: batch_size = self.mem_cntr 

        return self.get_values(range(self.mem_cntr-batch_size, self.mem_cntr))