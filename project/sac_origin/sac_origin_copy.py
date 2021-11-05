import imp
import copy
import numpy as np
import torch as T
import torch.nn.functional as F
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork
import os


class Agent:
    def __init__(self, layer_size=(256, 256), 
            state_dims=(16,), action_dims=2, is_training=True,
            replayer_capacity=100000, replayer_initial_transitions=10000, gamma=0.99, batch_size=128, interval=10000, 
            # replayer_capacity=100000, replayer_initial_transitions=15, gamma=0.99, batch_size=16, interval=20, 
            tau=0.005, learning_rate=0.0003, 
            var_linear=0.125, var_angular = 0.5, exploration_decay_start_step = 50000, 
            linear_action_space = (0, 0.25), angular_action_space = (-0.5, 0.5)):
        # self.gamma = gamma
        self.tau = tau  #用于target网络的更新， target网络学习率
        self.memory = ReplayBuffer(replayer_capacity, state_dims, action_dims)  # replayer buffer
        self.batch_size = batch_size
        self.action_dims = action_dims

        self.layer_size = layer_size    #隐藏层大小，输入为元组

        self.linear_action_space = linear_action_space  #线速度范围，为元组or列表
        self.angular_action_space = angular_action_space    #角速度范围，为元组or列表
        self.maxLinear = linear_action_space[1] #线速度的最大值
        self.maxAngular = angular_action_space[1]    #角速度最大值
        self.var_linear = var_linear    #线速度的方差
        self.var_angular = var_angular  #角速度的方差
        self.exploration_decay_start_step = exploration_decay_start_step
        self.interval = interval


        self.real_time_step = 0 #总的时间步数，包括了开始训练前，自由探索10000步的步长
        self.one_episode_step = 0   #一个回合的时间步数，当done==True or arrive==True or one_episode_step>500时，置0
        self.time_step = 0  #开始训练的时间步数
        self.episode = 0    #回合数
        self.pretrain_episode = 0   #开始训练前的时间步长

        self.observation_dim = state_dims   #state空间维度
        self.action_dim = action_dims   #action空间维度

        self.total_reward_10000 = 0 #一万time step的总奖励
        self.avg_reward_10000 = 0   #一万time step的平均奖励
        self.episode_reward = 0 #一个回合的总奖励
        self.avg_episode_reward = 0 #一个回合的平均time step奖励

        self.gamma = gamma  #折扣，防止return趋于无穷大
        self.learning_rate = learning_rate  #网络用于梯度下降的学习率
        self.is_training = is_training  #是否在训练中

        self.update_actor_iteration = 2 # actor 更新频率
        
        self.replayer_initial_transitions = replayer_initial_transitions


        self.actor = ActorNetwork(self.learning_rate, state_dims, self.layer_size[0], self.layer_size[1],
                                n_actions=action_dims, name='actor', maxLinear=self.maxLinear, maxAngular=self.maxAngular)
        self.q_critic = CriticNetwork(self.learning_rate, state_dims, self.layer_size[0], self.layer_size[1],
                                n_actions=action_dims, name='critic')
        self.q_critic_target = copy.deepcopy(self.q_critic)
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        self.alpha = 0.12
        self.adaptive_alpha = True
        device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        if self.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = T.tensor(-self.action_dim, dtype=float, requires_grad=True, device=device)
            # We learn log_alpha instead of alpha to ensure exp(log_alpha)=alpha>0
            self.log_alpha = T.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=device)
            self.alpha_optim = T.optim.Adam([self.log_alpha], lr=self.learning_rate)

        if not self.is_training:
            self.load_models(230000)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)       

    def decide(self, observation):
        if self.is_training:
            if self.is_training and self.memory.mem_cntr < \
                    self.replayer_initial_transitions:
                return np.array([np.random.uniform(self.linear_action_space[0], self.linear_action_space[1]), \
                    np.random.uniform(self.angular_action_space[0], self.angular_action_space[1])])
            
            with T.no_grad():
                state = T.tensor([observation], dtype=T.float).to(self.actor.device)
                mu, _ = self.actor.sample_normal(state)
            mu = mu.cpu().detach().numpy()[0]

            self.write_actions(mu)
        else:
            with T.no_grad():
                state = T.tensor([observation], dtype=T.float).to(self.actor.device)
                mu, _ = self.actor.sample_normal(state)
            mu = mu.cpu().detach().numpy()[0]
        return mu

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.q_critic.device)
        done = T.tensor(done, dtype=T.int).to(self.q_critic.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.q_critic.device)
        state = T.tensor(state, dtype=T.float).to(self.q_critic.device)
        action = T.tensor(action, dtype=T.float).to(self.q_critic.device)

        reward = reward.reshape(-1, 1)
        done = done.reshape(-1, 1)
        # update critic network
        with T.no_grad():
            a_next_re, log_pi_re = self.actor.sample_normal(state_)
            q1_target_re, q2_target_re = self.q_critic_target(state_, a_next_re)
            q_target_re = T.min(q1_target_re, q2_target_re)
            q_target = reward + (1 - done) * self.gamma * (q_target_re - self.alpha * log_pi_re)
        
        q1, q2 = self.q_critic(state, action)

        q_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.q_critic.optimizer.zero_grad()
        q_loss.backward()
        self.q_critic.optimizer.step()

        # update actor network
		# Freeze Q-networks so you don't waste computational effort
		# computing gradients for them during the policy learning step.
        for params in self.q_critic.parameters():
            params.requires_grad = False

        a_re, log_pi_re = self.actor.sample_normal(state)
        q1, q2 = self.q_critic(state, a_re)
        q = T.min(q1, q2)
        a_loss = (self.alpha * log_pi_re - q).mean()
        self.actor.optimizer.zero_grad()
        a_loss.backward()
        self.actor.optimizer.step()

        for params in self.q_critic.parameters():
            params.requires_grad = True

        # update alpha
        if self.adaptive_alpha:
			# we optimize log_alpha instead of aplha, which is aimed to force alpha = exp(log_alpha)> 0
			# if we optimize aplpha directly, alpha might be < 0, which will lead to minimun entropy.
            alpha_loss = -(self.log_alpha * (log_pi_re + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
        
        # update target net
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def perceive(self, observation, action, reward, next_observation,
                done, arrive, limit_step = 500):
                #     done = True
        if self.is_training:
            # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
            self.remember(observation, action, reward, next_observation,
                    done)

            self.real_time_step = self.real_time_step + 1
            self.one_episode_step = self.one_episode_step + 1

            print("real time step: ", self.real_time_step)
            print("time step:",self.time_step)
            if self.memory.mem_cntr == self.replayer_initial_transitions:
                print('\n---------------Start training---------------')
            # Store transitions to replay start size then start training
            if self.memory.mem_cntr > self.replayer_initial_transitions:

                self.time_step += 1
                self.learn()

            self.episode_reward = self.episode_reward + reward

            if self.time_step > 0:
                self.total_reward_10000 = self.total_reward_10000 + reward

            if self.time_step % self.interval == 0 and self.time_step > 0:
                self.save_models(self.time_step)
                self.avg_reward_10000 = self.total_reward_10000 / 10000
                self.write_reture_per_10000()
                self.total_reward_10000 = 0

                
            if self.time_step == 0 and (done == True or self.one_episode_step >= limit_step):
                self.pretrain_episode = self.pretrain_episode + 1
                self.avg_episode_reward = self.episode_reward / self.one_episode_step
                self.write_episode(False, self.pretrain_episode, done, arrive) #写入

            if self.time_step > 0 and (done == True or self.one_episode_step >= limit_step or arrive == True):
                self.episode = self.episode + 1
                self.avg_episode_reward = self.episode_reward / self.one_episode_step
                self.write_episode(True, self.episode, done, arrive)  #写入

            if arrive == True:
                self.episode_reward = 0
                self.one_episode_step = 0

            self.resetEnv = False   # 是否重置环境标志位
            if done == True or self.one_episode_step >= limit_step:
                self.resetEnv = True
                self.episode_reward = 0
                self.one_episode_step = 0
        else:
            self.time_step += 1
            self.one_episode_step += 1
            self.resetEnv = False   # 是否重置环境标志位
            if done == True or self.one_episode_step >= 500:
                self.resetEnv = True
            if done == True or arrive == True or self.resetEnv == True:
                self.episode += 1
                self.one_episode_step = 0

        return self.time_step

    def write_episode(self, start_train, episode, done, arrive):
        print("()()()()()write episode function()()()()()()\n")
        path = os.path.split(os.path.realpath(__file__))[0]
        str_ = "{:<10d} {:<10d} {:<10d} {:<10d} {:<10.2f} {:<10.2f} {:<10} {:<10}\n".format(episode, \
             self.real_time_step, self.time_step, self.one_episode_step, self.episode_reward, \
                 self.avg_episode_reward, done, arrive)
        if start_train == True:
            path = os.path.join(path, 'write_episode.txt')
            with open(path, "a") as file:
                file.write(str_)
        else:
            path = os.path.join(path, 'write_pre_episode.txt')
            with open(path, "a") as file:
                file.write(str_)

    def write_reture_per_10000(self):
        path = os.path.split(os.path.realpath(__file__))[0]
        str_ = "{:<10d} {:<10.2f} {:<10.2f}\n".format(self.time_step, \
            self.total_reward_10000, self.avg_reward_10000)
        path = os.path.join(path, 'write_10000.txt')
        with open(path, "a") as file:
            file.write(str_)
    
    def write_actions(self, action):
        path = os.path.split(os.path.realpath(__file__))[0]
        str_ = "{:<10d} {:<10.2f} {:<10.2f}\n".format(self.episode, action[0], action[1])
        path = os.path.join(path, 'actions.txt')
        with open(path, "a") as file:
                file.write(str_)

    def save_models(self, time_step):
        time_step = str(time_step)
        path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'models_1026', time_step)
        print("path: ", path)
        if not os.path.exists(path):
            os.makedirs(path)

        self.actor.save_checkpoint(os.path.join(path, 'actor.h5'))
        # self.value.save_checkpoint(os.path.join(path, 'value.h5'))
        # self.target_value.save_checkpoint(os.path.join(path, 'target_value.h5'))
        self.q_critic.save_checkpoint(os.path.join(path, 'q_critic.h5'))
        self.q_critic_target.save_checkpoint(os.path.join(path, 'q_critic_target.h5'))

    def load_models(self, time_step):
        time_step = str(time_step)
        path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'models_1026', time_step)

        self.actor.load_checkpoint(os.path.join(path, 'actor.h5'))
        self.q_critic.load_checkpoint(os.path.join(path, 'q_critic.h5'))
        self.q_critic_target.load_checkpoint(os.path.join(path, 'q_critic_target.h5'))