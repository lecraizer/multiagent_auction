import random
import numpy as np
import torch as T
from networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, alpha, beta, input_dims, tau, gamma=0.99,
                 n_agents=2, n_actions=1, layer1_size=400, layer2_size=300, 
                 batch_size=64, total_eps=100000, noise_std=0.2, 
                 tl_flag=False, extra_players=0):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.total_episodes = total_eps
        self.noise_std = noise_std

        self.actor = self._build_actor(alpha, input_dims, layer1_size, layer2_size, n_actions, n_agents, 'actor')
        self.critic = self._build_critic(beta, input_dims, layer1_size, layer2_size, n_actions, n_agents, 'critic', tl_flag, extra_players)

        self.target_actor = self._build_actor(alpha, input_dims, layer1_size, layer2_size, n_actions, n_agents, 'target_actor')
        self.target_critic = self._build_critic(beta, input_dims, layer1_size, layer2_size, n_actions, n_agents, 'target_critic', tl_flag, extra_players)

        self.update_network_parameters(tau=1)

    def _build_actor(self, alpha, input_dims, fc1, fc2, n_actions, n_agents, name):
        return ActorNetwork(alpha, input_dims, fc1, fc2, n_actions, name, n_agents)

    def _build_critic(self, beta, input_dims, fc1, fc2, n_actions, n_agents, name, tl_flag, extra):
        return CriticNetwork(beta, input_dims, fc1, fc2, n_actions, name, n_agents, flag=tl_flag, extra=extra)

    def choose_action(self, observation, episode, evaluation=False):
        self.actor.eval()
        obs_tensor = T.tensor([observation], dtype=T.float).to(self.actor.device)
        action = self.actor.forward(obs_tensor).to(self.actor.device)

        if not evaluation:
            noise = T.tensor(np.random.normal(0, self.noise_std), dtype=T.float).to(self.actor.device)
            decay = 1 - (episode / self.total_episodes)
            action += noise * decay
            action = action.clamp(0, 1)

        self.actor.train()
        return action.cpu().detach().numpy()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_models(self, name):
        self.actor.save_checkpoint(name)
        self.target_actor.save_checkpoint(name)
        self.critic.save_checkpoint(name)
        self.target_critic.save_checkpoint(name)

    def load_models(self, name):
        self.actor.load_checkpoint(name)
        self.target_actor.load_checkpoint(name)
        self.critic.load_checkpoint(name)
        self.target_critic.load_checkpoint(name)
