import gymnasium as gym
import numpy as np
import time
from gymnasium import wrappers
from tiago_navigation.ddpg import DDPG
from tiago_navigation.TD3 import TD3
import rospy
import rospkg
import torch

# import our training environment
from openai_ros.task_envs.tiago import tiago_navigation

if __name__ == '__main__':

    rospy.init_node('tiago_test_DDPG', anonymous=True, log_level=rospy.INFO)

    env = gym.make('TiagoNavigation-v0')
    rospy.loginfo("Gym env done!")
    ddpg_flag = True
    
    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('tiago_navigation') + '/train_results'
    rospy.loginfo('Moder are savend into folder : ' + pkg_path)
    outdir = '/training_results/DDPG'
    filename = 'DDPG300'

    last_time_steps = np.ndarray(0)

    #init training parameters
    nepisodes = rospy.get_param("/Training/nepisodes")
    nsteps = rospy.get_param("/Training/nsteps")

    rospy.logdebug("observation dim :" + str(env.observation_space.shape[0]))
    rospy.logdebug("action dim :" + str(env.action_space.shape[0]))

    #initialize the algorithm that we are going to use for learning
    if ddpg_flag:
        #initialize DDPG net
        ddpg_net = DDPG(env)
    else:
        ddpg_net = TD3(env)
    highest_reward = -float('inf')
    mean_reward = 0

    #upload network weight
    ddpg_net.load(filename , pkg_path)
    # Starts the main training loop: the one about the episodes to do
    for x in range(1 , 10):
        rospy.logdebug("############### START EPISODE=>" + str(x))

        cumulated_reward = 0
        # Initialize the environment and get first state of the robot
        curr_observation , info = env.reset()
        for i in range(nsteps):
            #generate current state from current observation
            state = torch.from_numpy(np.array(curr_observation)).float()
            waypoints = torch.from_numpy(np.array(info["waypoints"]).flatten()).float()
            #rospy.loginfo(str(state))
            #rospy.logwarn("############### Start Step=>" + str(i))
            # Run networj for obtain the action 
            action = ddpg_net(state ,waypoints , False).detach().numpy()
            #action = action.squeeze(0) 

            rospy.loginfo("Next action is:%d" + str(action) + " with linear velocity : " + str(action[0]) + " and angular velocity : " + str(action[1]))

            #obtain the results of the action obtained
            observation, reward, terminated , truncated , info = env.step(action)
            cumulated_reward += reward
            
            #terminate episode if Tiago crash 
            if terminated or truncated:
                break
    
            curr_observation = observation
            

        #weight update
        if truncated :
            rospy.logwarn("\n Robot reach the objective with the following reward : " + str(cumulated_reward) )
            # Open the file in append mode ('a')
            with open(rospack.get_path('tiago_navigation') + "/train_results/reward_goal.txt", 'a') as file:
                # Append the new data to the end of the file
                file.write("At episode : " + str(x) + " , obtain reward : " + str(cumulated_reward) + "\n")
        mean_reward += cumulated_reward
        rospy.loginfo("Reward of episode is : " + str(cumulated_reward) )

    #close environment
    env.close()
    