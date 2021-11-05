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
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims + self.n_actions, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        
        self.q = nn.Linear(self.fc2_dims, 1)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta,
                                    weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        layer1 = F.relu(self.bn1(state_value))

        state_action_value = T.cat([layer1, action], dim=1)
        
        state_value = self.fc2(state_action_value)
        layer2 = F.relu(self.bn2(state_value))
      
        q_output = self.q(layer2)

        return q_output

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
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')

        self.maxLinear = maxLinear
        self.maxAngular = maxAngular

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        # self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.mu_linear = nn.Linear(self.fc2_dims, 1)
        self.mu_angular = nn.Linear(self.fc2_dims, 1)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.mu_linear.weight.data.uniform_(-f3, f3)
        self.mu_linear.bias.data.uniform_(-f3, f3)
        self.mu_angular.weight.data.uniform_(-f3, f3)
        self.mu_angular.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')

        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        layer1 = F.relu(self.bn1(x))
        x = self.fc2(layer1)
        layer2 = F.relu(self.bn2(x))
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
