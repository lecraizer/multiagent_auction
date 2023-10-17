import random
import numpy as np
import torch as T
import torch.nn.functional as F
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

# from torchviz import make_dot

class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 n_actions=2, max_size=1000000, layer1_size=400,
                 layer2_size=300, batch_size=64, total_eps=100000):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.total_episodes = total_eps

        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, n_actions=n_actions,
                                  name='actor')
                    
        # dot = make_dot(self.actor(T.randn(400, 1)), params=dict(self.actor.named_parameters()))
        # dot.format = 'png'
        # dot.render('actor', format='png')

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

    def choose_action(self, observation, episode, evaluation=False):
        self.actor.eval()
        observation = T.tensor([observation], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        
        if evaluation:
            mu_prime = mu
        else:
            # N = 2
            # noise = T.tensor(np.random.normal(0, 0.2), dtype=T.float).to(self.actor.device)
            # mu_prime = mu + (noise*(1-(episode/self.total_episodes)))
            # mu_prime = mu_prime.clamp(0, 2)

            # # agent has chance = exploration_rate to play randomly 
            # exploration_rate = 1-(episode/self.total_episodes)
            # if random.random() < exploration_rate:
            #     mu_prime = T.tensor([np.random.uniform(0, N)], dtype=T.float).to(self.actor.device)
            # else:
            #     mu_prime = mu
        
            mu_prime = mu.clamp(0, 1)

        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward):
        self.memory.store_transition(state, action, reward)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward = self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        target_actions = self.target_actor.forward(state)
        critic_value_ = self.target_critic.forward(state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)

        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

        return critic_loss.item()
        # return critic_loss.item(), actor_loss.item()

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