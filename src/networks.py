# Description: This file contains the implementation of the Actor and Critic networks used in the MADDPG algorithm

import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, 
                 name, n_agents=2, chkpt_dir='models/critic', 
                 flag=False, extra=0):
        super(CriticNetwork, self).__init__()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.checkpoint_file = os.path.join(chkpt_dir, name)

        total_inputs = n_agents * (input_dims + n_actions)
        if flag:
            total_inputs = (n_agents + extra) * (input_dims + n_actions)

        self.fc1 = nn.Linear(total_inputs, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(self.device)

    def forward(self, state, action, others_states, others_actions):
        concat = T.cat([state, action, others_states, others_actions], dim=1)
        x = F.relu(self.fc1(concat))
        x = F.relu(self.fc2(x))
        q_value = self.q(x)
        return q_value

    def save_checkpoint(self, name):
        T.save(self.state_dict(), f'{self.checkpoint_file}_{name}')

    def load_checkpoint(self, name):
        print(f'... Loading critic model for {name} ...')
        self.load_state_dict(
            T.load(f'{self.checkpoint_file}_{name}', map_location=self.device)
        )


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, 
                 name, n_agents=2, chkpt_dir='models/actor'):
        super(ActorNetwork, self).__init__()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.checkpoint_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.mu = nn.Linear(fc2_dims, n_actions)

        self._init_layer(self.fc1)
        self._init_layer(self.fc2)
        self._init_layer(self.mu, scale=0.003)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)

    def _init_layer(self, layer, scale=1.0):
        """Initialize weights uniformly with a given scale."""
        fan_in = layer.weight.data.size()[0]
        limit = scale / np.sqrt(fan_in)
        T.nn.init.uniform_(layer.weight.data, -limit, limit)
        T.nn.init.uniform_(layer.bias.data, -limit, limit)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return T.sigmoid(self.mu(x))

    def save_checkpoint(self, name):
        T.save(self.state_dict(), f'{self.checkpoint_file}_{name}')

    def load_checkpoint(self, name):
        print(f'... Loading actor model for {name} ...')
        self.load_state_dict(
            T.load(f'{self.checkpoint_file}_{name}', map_location=self.device)
        )
