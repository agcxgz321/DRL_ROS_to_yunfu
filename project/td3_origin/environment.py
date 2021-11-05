#!/usr/bin/env python3
import os
import rospy
import numpy as np
import math
from math import pi
import random

from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel

# diagonal_dis = math.sqrt(2) * (3.6 + 3.8)
# 1. 增大距离归一化系数
diagonal_dis = math.sqrt(2) * 17    
goal_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', 'turtlebot3_simulations',
                              'turtlebot3_gazebo', 'models', 'Target', 'model.sdf')

random.seed(0)

class Env():
    def __init__(self, is_training):
        self.position = Pose()
        self.goal_position = Pose()
        self.goal_position.position.x = 0.
        self.goal_position.position.y = 0.
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.past_distance = 0.
        if is_training:
            self.threshold_arrive = 0.2
        else:
            self.threshold_arrive = 0.7

    def getGoalDistace(self):
        goal_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        self.past_distance = goal_distance

        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
        yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))

        if yaw >= 0:
             yaw = yaw
        else:
             yaw = yaw + 360

        rel_dis_x = round(self.goal_position.position.x - self.position.x, 1)
        rel_dis_y = round(self.goal_position.position.y - self.position.y, 1)

        # Calculate the angle between robot and target
        if rel_dis_x > 0 and rel_dis_y > 0:
            theta = math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x > 0 and rel_dis_y < 0:
            theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y > 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0:
            theta = 1 / 2 * math.pi
        elif rel_dis_x == 0 and rel_dis_y < 0:
            theta = 3 / 2 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0:
            theta = 0
        else:
            theta = math.pi
        rel_theta = round(math.degrees(theta), 2)

        diff_angle = abs(rel_theta - yaw)

        if diff_angle <= 180:
            diff_angle = round(diff_angle, 2)
        else:
            diff_angle = round(360 - diff_angle, 2)

        self.rel_theta = rel_theta
        self.yaw = yaw
        self.diff_angle = diff_angle

    def getState(self, scan):
        global scan_range
        scan_range = []
        yaw = self.yaw
        rel_theta = self.rel_theta
        diff_angle = self.diff_angle
        min_range = 0.2
        done = False
        arrive = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        if min_range > min(scan_range) > 0:
            done = True

        current_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        if current_distance <= self.threshold_arrive:
            # done = True
            arrive = True

        return scan_range, current_distance, yaw, rel_theta, diff_angle, done, arrive

    def setReward(self, done, arrive):
        current_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        distance_rate = (self.past_distance - current_distance)

        # reward = 500.*distance_rate
        reward = 500.*distance_rate - 1 \
                    - 3*(np.e**(0.4*((1/min(scan_range)) - 1)) - 1) \
                    + 10*(linear_vel_diff - 0.125)   #鼓励线速度提高，增加探索
                    # - 0.5*(np.e**(3*(abs(record_ang_vel))) - 1) \
 
        self.past_distance = current_distance

        if done:
            reward = -100.
            self.pub_cmd_vel.publish(Twist())

        if arrive:
            reward = 120.
            self.pub_cmd_vel.publish(Twist())
            rospy.wait_for_service('/gazebo/delete_model')
            self.del_model('target')

            # Build the target
            rospy.wait_for_service('/gazebo/spawn_sdf_model')
            try:
                goal_urdf = open(goal_model_dir, "r").read()
                target = SpawnModel
                target.model_name = 'target'  # the same with sdf name
                target.model_xml = goal_urdf
                # self.goal_position.position.x = random.uniform(-3.6, 3.6)
                # self.goal_position.position.y = random.uniform(-3.6, 3.6)
                self.goal_position.position.x = random.uniform(-8, 8)
                self.goal_position.position.y = random.uniform(-8, 8)
                while(self.getEulaDistance(self.goal_position.position.x, self.goal_position.position.y, self.lastGoal_x, self.lastGoal_y) < 6):
                    self.goal_position.position.x = random.uniform(-8, 8)
                    self.goal_position.position.y = random.uniform(-8, 8)
                self.lastGoal_x = self.goal_position.position.x
                self.lastGoal_y = self.goal_position.position.y

                # if  self.goal_position.position.x > -2.2 and self.goal_position.position.x < -0.8 and self.goal_position.position.y > -2.2 and self.goal_position.position.y < -0.8:
                #     self.goal_position.position.x += 1.4
                #     self.goal_position.position.y += 1.4
                # if  self.goal_position.position.x > -2.2 and self.goal_position.position.x < -0.8 and self.goal_position.position.y > 0.8 and self.goal_position.position.y < 2.2:
                #     self.goal_position.position.x += 1.4
                #     self.goal_position.position.y += 1.4
                # if  self.goal_position.position.x > 0.8 and self.goal_position.position.x < 2.2 and self.goal_position.position.y > -3.2 and self.goal_position.position.y < -1.8:
                #     self.goal_position.position.x += 1.4
                #     self.goal_position.position.y += 1.4
                # if  self.goal_position.position.x > 0.8 and self.goal_position.position.x < 2.2 and self.goal_position.position.y > -0.2 and self.goal_position.position.y < 1.2:
                #     self.goal_position.position.x += 1.4
                #     self.goal_position.position.y += 1.4
                self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
            except (rospy.ServiceException) as e:
                print("/gazebo/failed to build the target")
            rospy.wait_for_service('/gazebo/unpause_physics')
            self.goal_distance = self.getGoalDistace()
            arrive = False

        return reward

    def step(self, action, past_action):
        linear_vel = action[0]
        ang_vel = action[1]
        global linear_vel_diff
        linear_vel_diff = linear_vel
        vel_cmd = Twist()
        # vel_cmd.linear.x = linear_vel / 4
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
        state = [i / 3.5 for i in state]
        # print('X: %.3f' % self.position.x, 'Y: %.3f' % self.position.y, 'linear: %.3f' % action[0], 'steer: %.3f' % action[1])#############记录控制量和运动轨迹

        for pa in past_action:
            state.append(pa)

        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]
        reward = self.setReward(done, arrive)

        return np.asarray(state), reward, done, arrive

    def reset(self):
        # Reset the env #
        rospy.wait_for_service('/gazebo/delete_model')
        self.del_model('target')

        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        # Build the targetz
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            goal_urdf = open(goal_model_dir, "r").read()
            target = SpawnModel
            target.model_name = 'target'  # the same with sdf name
            target.model_xml = goal_urdf
            # self.goal_position.position.x = random.uniform(-3.6, 3.6)
            # self.goal_position.position.y = random.uniform(-3.6, 3.6)
            self.goal_position.position.x = random.uniform(-8, 8)
            self.goal_position.position.y = random.uniform(-8, 8)
            while(self.getEulaDistance(self.goal_position.position.x, self.goal_position.position.y, 0, 0) < 6):
                self.goal_position.position.x = random.uniform(-8, 8)
                self.goal_position.position.y = random.uniform(-8, 8)
            self.lastGoal_x = self.goal_position.position.x
            self.lastGoal_y = self.goal_position.position.y

            # if  self.goal_position.position.x > -2.2 and self.goal_position.position.x < -0.8 and self.goal_position.position.y > -2.2 and self.goal_position.position.y < -0.8:
            #     self.goal_position.position.x += 1.4
            #     self.goal_position.position.y += 1.4
            # if  self.goal_position.position.x > -2.2 and self.goal_position.position.x < -0.8 and self.goal_position.position.y > 0.8 and self.goal_position.position.y < 2.2:
            #     self.goal_position.position.x += 1.4
            #     self.goal_position.position.y += 1.4
            # if  self.goal_position.position.x > 0.8 and self.goal_position.position.x < 2.2 and self.goal_position.position.y > -3.2 and self.goal_position.position.y < -1.8:
            #     self.goal_position.position.x += 1.4
            #     self.goal_position.position.y += 1.4
            # if  self.goal_position.position.x > 0.8 and self.goal_position.position.x < 2.2 and self.goal_position.position.y > -0.2 and self.goal_position.position.y < 1.2:
            #     self.goal_position.position.x += 1.4
            #     self.goal_position.position.y += 1.4

            self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target")
        rospy.wait_for_service('/gazebo/unpause_physics')
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        self.goal_distance = self.getGoalDistace()
        state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
        state = [i / 3.5 for i in state]

        state.append(0)
        state.append(0)

        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]

        return np.asarray(state)
    def get_position(self):
        return self.position.x, self.position.y

    def get_target(self):
        return self.goal_position.position.x, self.goal_position.position.y

    def getEulaDistance(self, x, y, x2, y2):
        delta_x = x - x2
        delta_y = y - y2
        return math.sqrt(delta_x**2 + delta_y**2)

