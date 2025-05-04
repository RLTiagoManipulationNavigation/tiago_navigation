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
from openai_ros.task_envs.tiago import tiago_navigation

if __name__ == '__main__':

    rospy.init_node('tiago_DDPG', anonymous=True, log_level=rospy.DEBUG)

    env = gym.make('TiagoNavigation-v0')
    rospy.loginfo("Gym env done!")
    # Initialize TF listener
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    transform_laser = tf_buffer.lookup_transform(
                    'base_laser_link',
                    'base_footprint',
                    rospy.Time(0),
                    rospy.Duration(1.0)
    )
            # Print current position
    rospy.loginfo(f"Current position: x={transform_laser.transform.translation.x}, "
                         f"y={transform_laser.transform.translation.y}, "
                         f"z={transform_laser.transform.translation.z}")

    """
    # Define start_pose
    start_pose = PoseStamped()
    start_pose.header.frame_id = "map"  # Use "map" or "odom" as the frame of reference
    start_pose.header.stamp = rospy.Time.now()  # Set the current time

    # Set position for start_pose
    start_pose.pose.position.x = 0.0  # Starting x-coordinate
    start_pose.pose.position.y = 0.0  # Starting y-coordinate
    start_pose.pose.position.z = 0.0  # Often 0 for a 2D navigation

    # Set orientation for start_pose (quaternion values for no rotation)
    start_pose.pose.orientation.x = 0.0
    start_pose.pose.orientation.y = 0.0
    start_pose.pose.orientation.z = 0.0
    start_pose.pose.orientation.w = 1.0  # w = 1 means no rotation

    # Define goal_pose
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = "map"
    goal_pose.header.stamp = rospy.Time.now()

    # Set position for goal_pose
    goal_pose.pose.position.x = 1.2  # Goal x-coordinate
    goal_pose.pose.position.y = -4.4  # Goal y-coordinate
    goal_pose.pose.position.z = 0.0

    # Set orientation for goal_pose (optional; here, same as start)
    goal_pose.pose.orientation.x = 0.0
    goal_pose.pose.orientation.y = 0.0
    goal_pose.pose.orientation.z = 0.0
    goal_pose.pose.orientation.w = 1.0

    make_plan = rospy.ServiceProxy("/move_base/make_plan", GetPlan)
    try:
        response = make_plan(start_pose, goal_pose, 0.5)
        rospy.loginfo(str(response))
    except rospy.ServiceException as e:
        rospy.logerr("Failed to make plan: %s" % e)
    """