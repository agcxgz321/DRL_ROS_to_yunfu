import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# def weights_init_(m):
#     if isinstance(m, nn.Linear):
#         T.nn.init.xavier_uniform_(m.weight, gain=1)
#         T.nn.init.constant_(m.bias, 0)


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
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.fc3 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc4 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q2 = nn.Linear(self.fc2_dims, 1)


        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        input_concate = T.cat([state, action], dim=1)

        layer1 = F.relu(self.fc1(input_concate))
        layer2 = F.relu(self.fc2(layer1))
        q1 = self.q1(layer2)

        layer3 = F.relu(self.fc3(input_concate))
        layer4 = F.relu(self.fc4(layer3))
        q2 = self.q2(layer4)

        return q1, q2

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
        # self.max_action = max_action
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.reparam_noise = 1e-6

        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
        self.maxLinear = maxLinear
        self.maxAngular = maxAngular

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, 2)
        self.sigma = nn.Linear(self.fc2_dims, 2)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        layer2 = F.relu(self.fc2(layer1))

        mu = self.mu(layer2)
        sigma_ = self.sigma(layer2)
        sigma = T.clamp(sigma_, self.LOG_STD_MIN, self.LOG_STD_MAX)

        return mu, sigma


    def sample_normal(self, state):
        mu, sigma = self.forward(state) #应该是<e-8，太小了的缘故
        sigma_exp = sigma.exp()
        dist = T.distributions.Normal(mu, sigma_exp)
        u = dist.rsample() # reparameterizes the policy
        action = u.clone()

        action[:, 0] = T.sigmoid(action[:, 0]) * T.tensor(self.maxLinear)
        action[:, 1] = T.tanh(action[:, 1]) * T.tensor(self.maxAngular)
        action = action.to(self.device)

        log_probs = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
        return action, log_probs


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



