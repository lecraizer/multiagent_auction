import random
import numpy as np
import torch as T
from networks import ActorNetwork, CriticNetwork

class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, gamma=0.99,
                 n_actions=1, layer1_size=400, layer2_size=300, 
                 batch_size=64, total_eps=100000):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.total_episodes = total_eps

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, n_actions=n_actions,
                                  name='actor')

        self.critic = CriticNetwork(beta, input_dims, layer1_size,
                                    layer2_size, n_actions=n_actions,
                                    name='critic')

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, n_actions=n_actions,
                                         name='target_actor')
        self.target_critic = CriticNetwork(beta, input_dims, layer1_size,
                                           layer2_size, n_actions=n_actions,
                                           name='target_critic')

        self.update_network_parameters(tau=1)
    
        # from torchviz import make_dot 
        # dot = make_dot(self.actor(T.randn(400, 1)), params=dict(self.actor.named_parameters()))
        # dot.format = 'png'
        # dot.render('actor', format='png')


    def choose_action(self, observation, episode, evaluation=False):
        self.actor.eval()
        observation = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        
        if evaluation:
            mu_prime = mu
        else:
            noise = T.tensor(np.random.normal(0, 0.2), dtype=T.float).to(self.actor.device)
            mu_prime = mu + (noise*(1-(episode/self.total_episodes)))
            mu_prime = mu_prime.clamp(0, 1)

            # # agent has chance = exploration_rate to play randomly 
            # exploration_rate = 1-(episode/self.total_episodes)
            # if random.random() < exploration_rate:
            #     mu_prime = T.tensor([np.random.uniform(0, 1)], dtype=T.float).to(self.actor.device)
            # else:
            #     mu_prime = mu

            # mu_prime = mu_prime.clamp(0, 1)
        
            # mu_prime = mu.clamp(0, 1)

        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)
        
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

    def check_actor_params(self):
        current_actor_params = self.actor.named_parameters()
        current_actor_dict = dict(current_actor_params)
        original_actor_dict = dict(self.original_actor.named_parameters())
        original_critic_dict = dict(self.original_critic.named_parameters())
        current_critic_params = self.critic.named_parameters()
        current_critic_dict = dict(current_critic_params)
        print('Checking Actor parameters')

        for param in current_actor_dict:
            print(param, T.equal(original_actor_dict[param], current_actor_dict[param]))
        print('Checking critic parameters')
        for param in current_critic_dict:
            print(param, T.equal(original_critic_dict[param], current_critic_dict[param]))
        input()