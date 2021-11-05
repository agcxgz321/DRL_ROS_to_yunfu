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
            replayer_capacity=100000, replayer_initial_transitions=10000, gamma=0.99, batch_size=128,
            # replayer_capacity=100000, replayer_initial_transitions=130, gamma=0.99, batch_size=128,
            tau = 0.001, learning_rate=0.0001, 
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
        self.var_linear = var_linear    #线速度的衰减率
        self.var_angular = var_angular  #角速度的衰减率
        self.exploration_decay_start_step = exploration_decay_start_step


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
        
        self.replayer_initial_transitions = replayer_initial_transitions

        self.actor = ActorNetwork(self.learning_rate, state_dims, self.layer_size[0], self.layer_size[1],
                                n_actions=action_dims, name='actor', maxLinear=self.maxLinear, maxAngular=self.maxAngular)
        self.critic = CriticNetwork(self.learning_rate, state_dims, self.layer_size[0], self.layer_size[1],
                                n_actions=action_dims, name='critic')

        self.target_actor = ActorNetwork(self.learning_rate, state_dims, self.layer_size[0], self.layer_size[1],
                                n_actions=action_dims, name='target_actor', maxLinear=self.maxLinear, maxAngular=self.maxAngular)

        self.target_critic = CriticNetwork(self.learning_rate, state_dims, self.layer_size[0], self.layer_size[1],
                                n_actions=action_dims, name='target_critic')

        if not self.is_training:
            self.load_models(250000)
        else:
            self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)    #有bug，好像是用strict来记录mean与variance
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)      #但作者也不知道是否有效



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
        # for _ in range(self.batches):
        states, actions, rewards, states_, done = \
                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
    
        self.update_network_parameters()

    def perceive(self, observation, action, reward, next_observation,
                done, arrive, limit_step = 500):
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

            if self.time_step % 10000 == 0 and self.time_step > 0:
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
        self.critic.save_checkpoint(os.path.join(path, 'critic.h5'))
        self.target_critic.save_checkpoint(os.path.join(path, 'target_critic.h5'))

    def load_models(self, time_step):
        time_step = str(time_step)
        path = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'models_1026', time_step)
        self.actor.load_checkpoint(os.path.join(path, 'actor.h5'))
        self.target_actor.load_checkpoint(os.path.join(path, 'target_actor.h5'))
        self.critic.load_checkpoint(os.path.join(path, 'critic.h5'))
        self.target_critic.load_checkpoint(os.path.join(path, 'target_critic.h5'))