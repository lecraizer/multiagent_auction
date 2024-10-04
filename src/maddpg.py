# Module to implement the multi-agent DDPG algorithm

from agent import Agent
from buffer import ReplayBuffer

import torch as T
import torch.nn.functional as F

class MADDPG:
    def __init__(self, alpha=0.000025, beta=0.00025, input_dims=1, 
                 tau=0.001, gamma=0.99, BS=64, fc1=64, fc2=64, 
                 n_actions=1, n_agents=2, total_eps=100000, noise_std=0.2):
        self.agents = []
        self.num_agents = n_agents
        for i in range(n_agents):
            self.agents.append(Agent(alpha=alpha, beta=beta, input_dims=input_dims, 
                                     tau=tau, batch_size=BS, layer1_size=fc1, 
                                     layer2_size=fc2, n_agents=self.num_agents,
                                     n_actions=n_actions, total_eps=total_eps, noise_std=noise_std))
            
        self.batch_size = BS
        self.gamma = gamma
        self.max_size = 1000000
        self.memory = ReplayBuffer(self.max_size, input_dims, n_actions, self.num_agents)
        self.short_memory = ReplayBuffer(1, input_dims, n_actions, self.num_agents)


    def remember(self, state, action, reward, others_states, others_actions):
        self.memory.store_transition(state, action, reward, others_states, others_actions)
        self.short_memory.store_transition(state, action, reward, others_states, others_actions)

    def learn(self, idx):
        if self.memory.mem_cntr < self.batch_size:
            return

        # for agent_idx, agent in enumerate(self.agents):
        #     # For each agent
        agent = self.agents[idx]
        device = agent.critic.device
        
        state, action, reward, others_states, others_actions = self.memory.sample_buffer(self.batch_size)
        state = T.tensor(state, dtype=T.float).to(device)
        action = T.tensor(action, dtype=T.float).to(device)
        reward = T.tensor(reward, dtype=T.float).to(device)

        others_states = T.tensor(others_states, dtype=T.float).to(device)
        others_actions = T.tensor(others_actions, dtype=T.float).to(device)

        agent.target_actor.eval()
        agent.target_critic.eval()
        agent.critic.eval()

        target_actions = agent.target_actor.forward(state)           
        
        num_columns = others_states.shape[1] # get the number of columns in others_states
        indexes = list(range(self.num_agents))
        indexes.remove(idx)

        target_actions_list = [] # initialize an empty list to store the results
        for i in range(num_columns): # loop through each column and apply target_actor.forward()
            other_state_column = others_states[:, i].reshape(-1, 1)
            other_agent_forward = self.agents[indexes[i]].target_actor.forward(other_state_column)
            target_actions_list.append(other_agent_forward)
        others_target_actions = T.cat(target_actions_list, dim=1) # concatenate the results along the second dimension
        
        # Calculate critic value and target critic value
        critic_value_ = agent.target_critic.forward(state, target_actions, others_states, others_target_actions)
        critic_value = agent.critic.forward(state, action, others_states, others_actions)

        # Calculate target q value
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j])
        target = T.tensor(target).to(device)
        target = target.view(self.batch_size, 1)

        agent.critic.train()
        agent.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        agent.critic.optimizer.step()

        agent.critic.eval()
        agent.actor.optimizer.zero_grad()
        mu = agent.actor.forward(state)

        others_mus = [] # initialize an empty list to store the results
        for i in range(num_columns): # loop through each column and apply target_actor.forward()
            other_mu = others_states[:, i].reshape(-1, 1)
            others_mus.append(self.agents[indexes[i]].target_actor.forward(other_mu))
        others_mus = T.cat(others_mus, dim=1)
        
        agent.actor.train()
        actor_loss = -agent.critic.forward(state, mu, others_states, others_mus)

        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        agent.actor.optimizer.step()
        agent.update_network_parameters()

        # short memory
        if self.short_memory.mem_cntr < self.batch_size:
            return
        
        agent = self.agents[idx]
        device = agent.critic.device
        
        state, action, reward, others_states, others_actions = self.short_memory.sample_buffer(self.batch_size)
        state = T.tensor(state, dtype=T.float).to(device)
        action = T.tensor(action, dtype=T.float).to(device)
        reward = T.tensor(reward, dtype=T.float).to(device)

        others_states = T.tensor(others_states, dtype=T.float).to(device)
        others_actions = T.tensor(others_actions, dtype=T.float).to(device)

        agent.target_actor.eval()
        agent.target_critic.eval()
        agent.critic.eval()

        target_actions = agent.target_actor.forward(state)           
        
        num_columns = others_states.shape[1] # get the number of columns in others_states
        indexes = list(range(self.num_agents))
        indexes.remove(idx)

        target_actions_list = [] # initialize an empty list to store the results
        for i in range(num_columns): # loop through each column and apply target_actor.forward()
            other_state_column = others_states[:, i].reshape(-1, 1)
            other_agent_forward = self.agents[indexes[i]].target_actor.forward(other_state_column)
            target_actions_list.append(other_agent_forward)
        others_target_actions = T.cat(target_actions_list, dim=1) # concatenate the results along the second dimension
        
        # Calculate critic value and target critic value
        critic_value_ = agent.target_critic.forward(state, target_actions, others_states, others_target_actions)
        critic_value = agent.critic.forward(state, action, others_states, others_actions)

        # Calculate target q value
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j])
        target = T.tensor(target).to(device)
        target = target.view(self.batch_size, 1)

        agent.critic.train()
        agent.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        agent.critic.optimizer.step()

        agent.critic.eval()
        agent.actor.optimizer.zero_grad()
        mu = agent.actor.forward(state)

        others_mus = [] # initialize an empty list to store the results
        for i in range(num_columns): # loop through each column and apply target_actor.forward()
            other_mu = others_states[:, i].reshape(-1, 1)
            others_mus.append(self.agents[indexes[i]].target_actor.forward(other_mu))
        others_mus = T.cat(others_mus, dim=1)
        
        agent.actor.train()
        actor_loss = -agent.critic.forward(state, mu, others_states, others_mus)

        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        agent.actor.optimizer.step()
        agent.update_network_parameters()
        