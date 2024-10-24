import gymnasium as gym
import numpy
import time
from gymnasium import wrappers
# ROS packages required
import rospy
import rospkg
# import our training environment
from openai_ros.task_envs.tiago import tiago_navigation

if __name__ == '__main__':

    rospy.init_node('tiago_DDPG', anonymous=True, log_level=rospy.DEBUG)

    env = gym.make('TiagoNavigation-v0')
    rospy.loginfo("Gym env done!")
