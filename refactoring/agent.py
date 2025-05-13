import numpy as np
import torch as T
from networks import ActorNetwork, CriticNetwork

class Agent(object):
    """
    Agent implementation with actor-critic architecture.
    """
    def __init__(self, 
                 alpha: float, 
                 beta: float, 
                 input_dims: int, 
                 tau: float, 
                 gamma: float = 0.99, 
                 n_agents: int = 2, 
                 n_actions: int = 1, 
                 layer1_size: int = 400, 
                 layer2_size: int = 300, 
                 batch_size: int = 64, 
                 total_eps: int = 100000, 
                 noise_std: float = 0.2, 
                 tl_flag: bool = False, 
                 extra_players: int = 0):
        """
        Initialize the agent with actor and critic networks.
        
        Args:
            alpha (float): Learning rate for the actor network.
            beta (float): Learning rate for the critic network.
            input_dims (int): Dimensionality of the input state space.
            tau (float): Update parameter for target network updates.
            gamma (float, optional): Discount factor for future rewards. Defaults is 0.99.
            n_agents (int, optional): Number of agents in the environment. Defaults is 2.
            n_actions (int, optional): Dimensionality of the action space. Defaults is 1.
            layer1_size (int, optional): Size of the first hidden layer in networks. Defaults is 400.
            layer2_size (int, optional): Size of the second hidden layer in networks. Defaults is 300.
            batch_size (int, optional): Size of batches for training. Defaults is 64.
            total_eps (int, optional): Total number of episodes for training. Defaults is 100000.
            noise_std (float, optional): Standard deviation of exploration noise. Defaults is 0.2.
            tl_flag (bool, optional): Flag to enable transfer learning. Defaults is False.
            extra_players (int, optional): Number of additional players for transfer learning. Defaults is 0.
        """
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.total_episodes = total_eps
        self.noise_std = noise_std
        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, 
                                  name='actor', n_agents=n_agents)

        self.critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions, 
                                    name='critic', n_agents=n_agents, flag=tl_flag, extra=extra_players)

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions=n_actions, 
                                         name='target_actor', n_agents=n_agents)
        
        self.target_critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions,
                                           name='target_critic', n_agents=n_agents, flag=tl_flag, extra=extra_players)

        self.update_network_parameters(tau=1)

    def get_networks_dicts(self):
        """
        Get state dictionaries for all networks.
        
        Returns:
            tuple: Four dictionaries containing the named parameters of the critic,
                  actor, target critic, and target actor networks respectively.
        """
        critic_state_dict = dict(self.critic.named_parameters())
        actor_state_dict = dict(self.actor.named_parameters())
        target_critic_dict = dict(self.target_critic.named_parameters())
        target_actor_dict = dict(self.target_actor.named_parameters())

        return critic_state_dict, actor_state_dict, target_critic_dict, target_actor_dict
    
    def choose_action(self, observation, episode, evaluation = False):
        """
        Select an action based on the current policy and exploration strategy.
        
        Args:
            observation: The current state of the environment.
            episode (int): Current episode number, used for noise decay.
            evaluation (bool, optional): If True, no exploration noise is added. Defaults is False.
            
        Returns:
            numpy.ndarray: The selected action.
        """
        self.actor.eval()
        observation = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        
        if evaluation:
            mu_prime = mu
        else:
            noise = T.tensor(np.random.normal(0, self.noise_std), dtype=T.float).to(self.actor.device)
            mu_prime = mu + (noise*(1-(episode/self.total_episodes)))
            mu_prime = mu_prime.clamp(0, 1)

        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def update_network_parameters(self, tau=None):
        """
        Update target network parameters.
        
        Args:
            tau (float, optional): Update parameter. If None, use the
                                  instance's tau value. Defaults is None.
        """
        if tau is None: tau = self.tau
        critic_state_dict, actor_state_dict, target_critic_dict, target_actor_dict = self.get_networks_dicts()

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_dict[name].clone()
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        
    def save_models(self, name):
        """
        Save all network models to checkpoint files.
        
        Args:
            name (str): Identifier to append to the checkpoint filenames.
        """
        for network in (self.actor, self.target_actor, self.critic, self.target_critic):
            network.save_checkpoint(name)

    def load_models(self, name):
        """
        Load all network models from checkpoint files.
        
        Args:
            name (str): Identifier of the checkpoint files to load.
        """
        for network in (self.actor, self.target_actor, self.critic, self.target_critic):
            network.load_checkpoint(name)