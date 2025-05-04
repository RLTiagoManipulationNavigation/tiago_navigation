import gymnasium as gym
import numpy
import time
from gymnasium import wrappers
# ROS packages required
import rospy
import rospkg
from nav_msgs.srv import GetPlan
from geometry_msgs.msg import PoseStamped
import tf2_ros
# import our training environment
from openai_ros.task_envs.tiago import tiago_arm_motion

if __name__ == '__main__':

    rospy.init_node('tiago_DDPG', anonymous=True, log_level=rospy.DEBUG)

    env = gym.make('TiagoArmMotion-v0')