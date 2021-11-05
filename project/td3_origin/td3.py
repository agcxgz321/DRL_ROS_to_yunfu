import imp
import numpy as np
import torch as T
import torch.nn.functional as F
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork
import os


class Agent:
    def __init__(self, layer_size=(400, 300), 
            state_dims=(16,), action_dims=2, is_training=True,
            replayer_capacity=100000, replayer_initial_transitions=10000, gamma=0.99, batch_size=128, interval=10000, 
            # replayer_capacity=100000, replayer_initial_transitions=15, gamma=0.99, batch_size=16, interval=20, 
            tau=0.001, learning_rate=0.0001, 
            # noise_scale=0.1, 
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
        self.critic_1 = CriticNetwork(self.learning_rate, state_dims, self.layer_size[0], self.layer_size[1],
                                n_actions=action_dims, name='critic_1')
        self.critic_2 = CriticNetwork(self.learning_rate, state_dims, self.layer_size[0], self.layer_size[1],
                                n_actions=action_dims, name='critic_2')

        self.target_actor = ActorNetwork(self.learning_rate, state_dims, self.layer_size[0], self.layer_size[1],
                                n_actions=action_dims, name='target_actor', maxLinear=self.maxLinear, maxAngular=self.maxAngular)

        self.target_critic_1 = CriticNetwork(self.learning_rate, state_dims, self.layer_size[0], self.layer_size[1],
                                n_actions=action_dims, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(self.learning_rate, state_dims, self.layer_size[0], self.layer_size[1],
                                n_actions=action_dims, name='target_critic_2')

        if not self.is_training:
            self.load_models(200000)
        else:
            self.update_network_parameters(tau=1)


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + \
                    (1-tau)*target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau*critic_2_state_dict[name].clone() + \
                    (1-tau)*target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                    (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)



    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)       

    def decide(self, observation):
        if self.is_training:
            if self.is_training and self.memory.mem_cntr < \
                    self.replayer_initial_transitions:
                return np.array([np.random.uniform(self.linear_action_space[0], self.linear_action_space[1]), \
                    np.random.uniform(self.angular_action_space[0], self.angular_action_space[1])])
            
            self.actor.eval()
            state = T.tensor([observation], dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)

            mu = mu.cpu().detach().numpy()[0]

            self.write_actions(mu)

            # if self.is_training:
            if self.time_step >= self.exploration_decay_start_step and self.time_step % 5 == 0:
                self.var_linear = self.var_linear * 0.9998
                self.var_angular = self.var_angular * 0.9998

            mu = np.array([np.random.normal(mu[0], self.var_linear), np.random.normal(mu[1], self.var_angular)])

            mu = np.array([np.clip(mu[0], self.linear_action_space[0], self.linear_action_space[1]), 
                            np.clip(mu[1], self.angular_action_space[0], self.angular_action_space[1])])
        else:
            self.actor.eval()
            state = T.tensor([observation], dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)

            mu = mu.cpu().detach().numpy()[0]
            mu = np.array([np.clip(mu[0], self.linear_action_space[0], self.linear_action_space[1]), 
                            np.clip(mu[1], self.angular_action_space[0], self.angular_action_space[1])])
        return mu

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
        done = T.tensor(done).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_1.device)

        target_actions = self.target_actor.forward(state_)

        ''' 
        target_actions = target_actions + \
                T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        # might break if elements of min and max are not all equal
        target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])
        '''

        linear_normal = np.full((self.batch_size,), np.random.normal(self.var_linear))
        limit_linear = (self.linear_action_space[1] - self.linear_action_space[0])/2
        linear_normal = np.clip(linear_normal, -limit_linear, limit_linear)
        
        angular_normal = np.full((self.batch_size,), np.random.normal(self.var_angular))
        limit_angular = (self.angular_action_space[1] - self.angular_action_space[0])/2
        angular_normal = np.clip(angular_normal, -limit_angular, limit_angular)

        linear_normal = linear_normal.reshape(self.batch_size, 1)
        angular_normal = angular_normal.reshape(self.batch_size, 1)
        
        normal_total = np.concatenate((linear_normal, angular_normal), axis=1)
        normal_tensor = T.tensor(normal_total, dtype=T.float).to(self.actor.device)
        target_actions = target_actions + normal_tensor
        



        q1_ = self.target_critic_1.forward(state_, target_actions)
        q2_ = self.target_critic_2.forward(state_, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        # self.critic_1.optimizer.zero_grad()
        # q1_loss = F.mse_loss(target, q1)
        # q1_loss.backward()
        # self.critic_1.optimizer.step()

        # self.critic_2.optimizer.zero_grad()
        # q2_loss = F.mse_loss(target, q2)
        # q2_loss.backward()       
        # self.critic_2.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss # 难受，这里居然是加起来，而不是选择和paper一样较小的来更新！！！
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # self.learn_step_cntr += 1

        if self.time_step % self.update_actor_iteration != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

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
            if done == True or self.one_episode_step >= 1000:
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
        self.target_actor.save_checkpoint(os.path.join(path, 'target_actor.h5'))
        self.critic_1.save_checkpoint(os.path.join(path, 'critic_1.h5'))
        self.target_critic_1.save_checkpoint(os.path.join(path, 'target_critic_1.h5'))
        self.critic_2.save_checkpoint(os.path.join(path, 'critic_2.h5'))
        self.target_critic_2.save_checkpoint(os.path.join(path, 'target_critic_2.h5'))

    def load_models(self, time_step):
        time_step = str(time_step)
        path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'models_1026', time_step)

        self.actor.load_checkpoint(os.path.join(path, 'actor.h5'))
        self.target_actor.load_checkpoint(os.path.join(path, 'target_actor.h5'))
        self.critic_1.load_checkpoint(os.path.join(path, 'critic_1.h5'))
        self.target_critic_1.load_checkpoint(os.path.join(path, 'target_critic_1.h5'))
        self.critic_1.load_checkpoint(os.path.join(path, 'critic_2.h5'))
        self.target_critic_2.load_checkpoint(os.path.join(path, 'target_critic_2.h5'))