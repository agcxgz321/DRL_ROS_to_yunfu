import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        # I think this breaks if the env has a 2D state representation
        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        input = T.cat([state, action], dim=1)
        x = self.fc1(input)
        layer1 = F.relu(x)
        x = self.fc2(layer1)
        layer2 = F.relu(x)

        q = self.q1(layer2)

        return q

    def save_checkpoint(self, dir):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), dir)

    def load_checkpoint(self, dir):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(dir))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg', maxLinear=0.25, maxAngular=0.5):
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')
        self.maxLinear = maxLinear
        self.maxAngular = maxAngular

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu_linear = nn.Linear(self.fc2_dims, 1)
        self.mu_angular = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)
    def forward(self, state):
        x = self.fc1(state)
        layer1 = F.relu(x)
        x = self.fc2(layer1)
        layer2 = F.relu(x)

        linear_ = T.sigmoid(self.mu_linear(layer2))
        angular_ = T.tanh(self.mu_angular(layer2))
        linear = linear_ * self.maxLinear
        angular = angular_ * self.maxAngular

        return T.cat([linear, angular], dim=1)

    def save_checkpoint(self, dir):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), dir)

    def load_checkpoint(self, dir):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(dir))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)
