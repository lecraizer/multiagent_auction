import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, 
                 name, n_agents=2, chkpt_dir='models/critic'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir,name)
        sumation = n_agents*(input_dims+n_actions)
        self.fc1 = nn.Linear(sumation, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action, others_states, others_actions):
        conc = T.cat([state, action, others_states, others_actions], dim=1)
        x = F.relu(self.fc1(conc))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q

    def save_checkpoint(self, name):
        # print('... Saving critic model for ' + name + ' ...')
        T.save(self.state_dict(), self.checkpoint_file + '_' + name)

    def load_checkpoint(self, name):
        print('... Loading critic model for ' + name + ' ...')
        self.load_state_dict(T.load(self.checkpoint_file + '_' + name))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, 
                 name, n_agents=2, chkpt_dir='models/actor'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir,name)
        self.n_agents = n_agents
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # auction = 'common_value'
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = T.sigmoid(self.mu(x))
        # if auction == 'common_value':
            # x = x * self.n_agents
        return x

    def save_checkpoint(self, name):
        # print('... Saving actor model for ' + name + ' ...')
        T.save(self.state_dict(), self.checkpoint_file + '_' + name)

    def load_checkpoint(self, name):
        print('... Loading actor model for ' + name + ' ...')
        self.load_state_dict(T.load(self.checkpoint_file + '_' + name))