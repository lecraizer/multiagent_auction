from agent import Agent
from buffer import ReplayBuffer
import torch as T
import torch.nn.functional as F

class MADDPG:
    def __init__(self, alpha=0.000025, beta=0.00025, input_dims=1, 
                 tau=0.001, gamma=0.99, BS=64, fc1=64, fc2=64, 
                 n_actions=1, n_agents=2):
        self.agents = []
        for agent_idx in range(n_agents):
            self.agents.append(Agent(alpha=alpha, beta=beta, input_dims=input_dims, 
                                     tau=tau, batch_size=BS, layer1_size=fc1, 
                                     layer2_size=fc2, n_actions=n_actions))
            
        self.batch_size = BS
        self.gamma = gamma
        self.memory = ReplayBuffer(1000000, input_dims, n_actions)


    def remember(self, state, action, reward):
        self.memory.store_transition(state, action, reward)


    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        for agent_idx, agent in enumerate(self.agents):
            # For each agent
            device = self.agents[agent_idx].critic.device
            state, action, reward = self.memory.sample_buffer(self.batch_size)
            reward = T.tensor(reward, dtype=T.float).to(device)
            action = T.tensor(action, dtype=T.float).to(device)
            state = T.tensor(state, dtype=T.float).to(device)

            agent.target_actor.eval()
            agent.target_critic.eval()
            agent.critic.eval()
            target_actions = agent.target_actor.forward(state)

            # Calculate critic value and target critic value
            critic_value_ = agent.target_critic.forward(state, target_actions, state, target_actions)
            critic_value = agent.critic.forward(state, action, state, action)

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
            agent.actor.train()
            actor_loss = -agent.critic.forward(state, mu, state, mu)

            actor_loss = T.mean(actor_loss)
            actor_loss.backward()
            agent.actor.optimizer.step()

            agent.update_network_parameters()