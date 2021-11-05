#!/usr/bin/env python3

# from Dqn.src.ddpg_rlbook.ddpg_xzq import DDPGAgent
# from tensorflow.python.framework.ops import device
# from ddpg_book import *
from td3 import Agent
import numpy as np
np.random.seed(0)
import pandas as pd
import matplotlib.pyplot as plt
# import gym
# import tensorflow as tf
# tf.random.set_seed(0)

import rospy
# import gym
# import gym_gazebo
# import numpy as np
# import tensorflow as tf
# from ddpg import *
from environment import Env
import time
import os


def write_informations(episode, time_step, one_episode_step, x, y, action, done, arrive, overtime, target_x, target_y):
    root_path = os.path.split(os.path.realpath(__file__))[0]
    # if not over:
        # str_ = "{:<10d} {:<10d} {:<10d} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}\n".format(episode, time_step, one_episode_step, x, y, action[0], action[1])
    if done:
        str_ = "done\n"
    elif arrive:
        str_ = "arrive\n"
    elif overtime:
        str_ = "overtime\n"
    else:
        str_ = "{:<10d} {:<10d} {:<10d} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}\n".format(episode, time_step, one_episode_step, x, y, action[0], action[1], target_x, target_y)

    path = os.path.join(root_path, 'evaluate_long_1000.txt')
    with open(path, "a") as file:
            file.write(str_)

exploration_decay_start_step = 50000
state_dims = (16,)
action_dims = 2
action_linear_max = 0.25  # m/s
action_angular_max = 0.5  # rad/s
is_training = False

def main():
    rospy.init_node('ddpg_stage_1')
    env = Env(is_training)
    layer_size = (400, 300)

    agent = Agent(layer_size=layer_size, state_dims=state_dims, action_dims=action_dims, is_training=is_training)
            
    past_action = np.array([0., 0.])
    print('State Dimensions: ' + str(state_dims))
    print('Action Dimensions: ' + str(action_dims))
    print('Action Max: ' + str(action_linear_max) + ' m/s and ' + str(action_angular_max) + ' rad/s')

    if is_training:
        print('Training mode')
        avg_reward_his = []
        total_reward = 0
        var = 1.

        # state = env.reset()
        while True:
            state = env.reset()
            # one_round_step = 0

            while True:
                print("start step")
                a = agent.decide(state)
                print("after deciede")

                state_, r, done, arrive = env.step(a, past_action)
                time_step = agent.perceive(state, a, r, state_, done, arrive)


                print("one round stepe: ", agent.one_episode_step)
                if arrive:
                    result = 'Success'
                else:
                    result = 'Fail'

                if time_step > 0:
                    total_reward += r
                    print("total_reward: ", total_reward)

                if time_step % 10000 == 0 and time_step > 0:
                    print('---------------------------------------------------')
                    
                    avg_reward = total_reward / 10000
                    print('Average_reward = ', avg_reward)
                    avg_reward_his.append(round(avg_reward, 2))
                    print('Average Reward:',avg_reward_his)
                    total_reward = 0

                if time_step % 5 == 0 and time_step > exploration_decay_start_step:
                    var *= 0.9999

                past_action = a
                state = state_
                # one_round_step += 1

                if arrive:
                    print('Step: %3i' % agent.one_episode_step, '| Var: %.2f' % var, '| Time step: %i' % time_step, '|', result)
                    # one_round_step = 0
                # if done or one_round_step >= 500:
                if agent.resetEnv == True:
                    print('Step: %3i' % agent.one_episode_step, '| Var: %.2f' % var, '| Time step: %i' % time_step, '|', result)
                    break

    else:
        print('Testing mode')
        arrive_time = 0
        done_time = 0
        state = env.reset()
        t = time.time()
        last_t = t
        while agent.episode < 100:
            one_round_step = 0
            while True:
                a = agent.decide(state)
                # a[0] = np.clip(a[0], 0., 1.)
                # a[1] = np.clip(a[1], -0.5, 0.5)
                state_, r, done, arrive = env.step(a, past_action)
                past_action = a
                state = state_
                one_round_step += 1
                agent.perceive(state, a, r, state_, done, arrive)
                x1, y1 = env.get_position()
                xT, yT = env.get_target()
                write_informations(agent.episode, agent.time_step, agent.one_episode_step, x1, y1, a, done, arrive, agent.resetEnv, xT, yT)
                if arrive:
                    arrive_time += 1
                    t = time.time()
                    delta_t = t - last_t
                    last_t = t
                    print('episode: ', agent.episode, 'Step: %3i' % one_round_step, '| Arrive!!!', 'arrive_time:', arrive_time, 'duation_time: %.3f'%delta_t)
                    one_round_step = 0
                    break
                if done:
                    done_time += 1
                    t = time.time()
                    delta_t = t - last_t
                    last_t = t
                    print('episode: ', agent.episode, 'Step: %3i' % one_round_step, '| done!!!', 'done_time: ', done_time, 'duation_time: %.3f'%delta_t)
                    one_round_step = 0
                    state = env.reset()
                    break
                if agent.resetEnv:
                    t = time.time()
                    delta_t = t - last_t
                    last_t = t
                    print('episode: ', agent.episode, 'Step: %3i' % one_round_step, '| overtime!!!', 'duation_time: %.3f'%delta_t)
                    one_round_step = 0
                    state = env.reset()
                    break


if __name__ == '__main__':
    main()