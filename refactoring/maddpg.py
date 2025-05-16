import torch as T
import numpy as np
import torch.nn.functional as F

from agent import Agent
from buffer import ReplayBuffer

class MADDPG:
    def __init__(self,
                 alpha=0.000025,
                 beta=0.00025,
                 input_dims=1,
                 tau=0.001,
                 gamma=0.99,
                 BS=64,
                 fc1=64,
                 fc2=64,
                 n_actions=1,
                 n_agents=2,
                 total_eps=100000,
                 noise_std=0.2,
                 tl_flag=False,
                 extra_players=0):
        self.num_agents = n_agents
        self.agents = [Agent(alpha=alpha, beta=beta, input_dims=input_dims, tau=tau, batch_size=BS, 
                             layer1_size=fc1, layer2_size=fc2, n_agents=self.num_agents,
                             n_actions=n_actions, total_eps=total_eps, noise_std=noise_std, 
                             tl_flag=tl_flag, extra_players=extra_players) for _ in range(n_agents)]

            
        self.batch_size = BS
        self.gamma = gamma
        self.max_size = 1000000
        self.memory = ReplayBuffer(self.max_size, input_dims, n_actions, self.num_agents)
        self.short_memory = ReplayBuffer(1, input_dims, n_actions, self.num_agents)

    def remember(self, state, action, reward, others_states, others_actions):
        self.memory.store_transition(state, action, reward, others_states, others_actions)
        self.short_memory.store_transition(state, action, reward, others_states, others_actions)

    def create_tiled_columns(self, others_states, others_actions, num_tiles):
        # Extract first column
        first_column_others_states = np.expand_dims(others_states[:, 0], axis=1)
        first_column_others_actions = np.expand_dims(others_actions[:, 0], axis=1)

        # Create tiles
        tiled_others_states = np.tile(first_column_others_states, (1, num_tiles))
        tiled_others_actions = np.tile(first_column_others_actions, (1, num_tiles))

        # Concatenate with original data
        others_states = np.concatenate([others_states, tiled_others_states], axis=1)
        others_actions = np.concatenate([others_actions, tiled_others_actions], axis=1)

        return others_states, others_actions
    
    def increase_tensor(self, tensor, num_tiles):
        first_column_tensor = tensor[:, 0].unsqueeze(1)
        tiled_tensor = T.cat([first_column_tensor] * num_tiles, dim=1)
        tensor = T.cat([tensor, tiled_tensor], dim=1)

        return tensor
    
    def update_critic_network(self, agent, target, critic_value):
        agent.critic.train()
        agent.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        agent.critic.optimizer.step()
        agent.critic.eval()

    def update_actor_network(self, agent, state, others_states, others_actions_pred):
        agent.actor.train()
        agent.actor.optimizer.zero_grad()
        action_pred  = agent.actor.forward(state)
        actor_loss = -agent.critic.forward(state, action_pred , others_states, others_actions_pred)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        agent.actor.optimizer.step()

    def compute_target_actions(self, idx, others_states):
        num_columns = self.num_agents - 1
        indexes = [i for i in range(self.num_agents) if i != idx]

        target_actions_list = [] # initialize an empty list to store the results
        for i in range(num_columns): # loop through each column and apply target_actor.forward()
            other_state_column = others_states[:, i].reshape(-1, 1)
            other_agent_forward = self.agents[indexes[i]].target_actor.forward(other_state_column)
            target_actions_list.append(other_agent_forward)

        return num_columns, indexes, T.cat(target_actions_list, dim=1)

    def learn_from_memory(self, memory, idx, flag=False, num_tiles=3):
        if memory.mem_cntr < self.batch_size: return

        agent = self.agents[idx]
        device = agent.critic.device
        
        state, action, reward, others_states, others_actions = memory.sample_buffer(self.batch_size)
        state = T.tensor(state, dtype=T.float).to(device)
        action = T.tensor(action, dtype=T.float).to(device)
        reward = T.tensor(reward, dtype=T.float).to(device)

        if flag:
            others_states, others_actions = self.create_tiled_columns(others_states, others_actions, num_tiles)

        others_states = T.tensor(others_states, dtype=T.float).to(device)
        others_actions = T.tensor(others_actions, dtype=T.float).to(device)

        agent.target_actor.eval()
        agent.target_critic.eval()
        agent.critic.eval()

        target_actions = agent.target_actor.forward(state)           
        
        num_columns, indexes, others_target_actions = self.compute_target_actions(idx, others_states)
        
        if flag: others_target_actions = self.increase_tensor(others_target_actions, num_tiles)

        # Calculate critic value and target critic value
        target_critic_value = agent.target_critic.forward(state, target_actions, others_states, others_target_actions)
        critic_value = agent.critic.forward(state, action, others_states, others_actions)

        # Calculate target q value
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*target_critic_value[j])
        target = T.tensor(target).to(device)
        target = target.view(self.batch_size, 1)

        self.update_critic_network(agent, target, critic_value)

        others_mus = [] # initialize an empty list to store the results
        for i in range(num_columns): # loop through each column and apply target_actor.forward()
            other_mu = others_states[:, i].reshape(-1, 1)
            others_mus.append(self.agents[indexes[i]].target_actor.forward(other_mu))
        others_mus = T.cat(others_mus, dim=1)
        
        if flag: others_mus = self.increase_tensor(others_mus, num_tiles)

        self.update_actor_network(agent, state, others_states, others_mus)
        agent.update_network_parameters()

    def learn(self, idx, flag=False, num_tiles=3):
        self.learn_from_memory(self.memory, idx, flag=flag, num_tiles=num_tiles)
        self.learn_from_memory(self.short_memory, idx, flag=flag, num_tiles=num_tiles)