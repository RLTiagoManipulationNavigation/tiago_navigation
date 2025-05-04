import gymnasium as gym
import numpy as np
import time
from gymnasium import wrappers
from tiago_navigation.ddpg import DDPG
from tiago_navigation.TD3 import TD3
import rospy
import rospkg
import torch
import os

# import our training environment
from openai_ros.task_envs.tiago import tiago_navigation


def remove_file(folder_path , filename):
     # Create the full path to the file
    file_path = os.path.join(folder_path, filename)
    
    # Check if the file exists
    if os.path.isfile(file_path):
        try:
            # Delete the file
            os.remove(file_path)
            rospy.loginfo(f"File '{filename}' has been deleted successfully.")
        except Exception as e:
            rospy.loginfo(f"An error occurred while deleting the file: {e}")
    else:
        rospy.loginfo(f"File '{filename}' not found in the folder '{folder_path}'.")

if __name__ == '__main__':

    rospy.init_node('tiago_DDPG', anonymous=True, log_level=rospy.INFO)

    env = gym.make('TiagoNavigation-v0')
    rospy.logdebug("Gym env done!")

    #flag used for dermine if use DDPG or TD3 algorithm 
    ddpg_flag = True
    
    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('tiago_navigation') + '/train_results'
    rospy.loginfo('Moder are savend into folder : ' + pkg_path)
    filename = 'DDPG'
    
    remove_file(rospack.get_path('tiago_navigation') + "/train_results/" , "reward_goal.txt")
    remove_file(rospack.get_path('tiago_navigation') + "/train_results/" , "mean_reward.txt")
    remove_file(rospack.get_path('tiago_navigation') + "/train_results/" , "eval_reward.txt")

    #variable used into training phase
    update_step = 0
    init_update = 0
    mean_reward = 0

    #init training parameters
    nepisodes = rospy.get_param("/Training/nepisodes")
    nsteps = rospy.get_param("/Training/nsteps")
    if ddpg_flag:
        #initialize DDPG net
        ddpg_net = DDPG(env)
    else:
        ddpg_net = TD3(env)

    # Starts the main training loop: the one about the episodes to do
    for x in range(1 , nepisodes + 1):
        rospy.logwarn("############### START EPISODE=>" + str(x))

        cumulated_reward = 0
        # Init new episode with reset the simulation
        curr_observation , info = env.reset()
        for i in range(nsteps):
            #generate current state from current observation
            state = torch.from_numpy(np.array(curr_observation)).float()
            waypoints = torch.from_numpy(np.array(info["waypoints"]).flatten()).float()
            #rospy.loginfo(str(state))
            #rospy.logwarn("############### Start Step=>" + str(i))
            # Run networj for obtain the action
            if x%10 == 0 : 
                action = ddpg_net(state ,waypoints , False).detach().numpy()
                rospy.loginfo(str(action))
            else:     
                action = ddpg_net(state ,waypoints).detach().numpy()
            #action = action.squeeze(0) 

            rospy.logdebug("Next action is:%d" + str(action) + " with linear velocity : " + str(action[0]) + " and angular velocity : " + str(action[1]))

            #obtain the results of the action obtained
            observation, reward, terminated , truncated , info = env.step(action)
            cumulated_reward += reward
            
            #terminate episode if Tiago crash 
            if terminated or truncated:
                break
            #add the data obtained from current step into replay buffer 
            if x%10 != 0 : 
                ddpg_net.update_buffer(curr_observation , observation , reward , action , [element for tuple in info["waypoints"] for element in tuple])
            ##if x%20 == 0 :
            #    ddpg_net.print_param()
            #weight update
            #if x%10 != 0:
            #    if i%20 == 0:
            #        ddpg_net.update() 
          
            """
            if i%update_step == 0 :
                critic_loss , actor_loss = ddpg_net.update()
                rospy.loginfo("\n Update of stepss : " + str(i) 
                      + " , critic loss of current episode is : " + str(critic_loss)
                      + " , actor loss of current episode is : " + str(actor_loss))
            """
    
            curr_observation = observation
            
        #weight update
        #if x%10 != 0 : 
        #    ddpg_net.update(True)
        #rospy.logwarn("\n Result of episodes " + str(x) )
                      #+ " , episode reward : " + str(cumulated_reward) 
        #              + " , critic loss of current episode is : " + str(critic_loss)
        #              + " , actor loss of current episode is : " + str(actor_loss))
        if truncated and info.get("goal_reach" , False):
            rospy.loginfo(str(info))
            rospy.logwarn("\n Robot reach the objective with the following reward : " + str(cumulated_reward) )
            # Open the file in append mode ('a')
            with open(rospack.get_path('tiago_navigation') + "/train_results/reward_goal.txt", 'a') as file:
                # Append the new data to the end of the file
                file.write("At episode : " + str(x) + " , obtain reward : " + str(cumulated_reward) + "\n")
        mean_reward += cumulated_reward
        if x%11 == 0 :
            with open(rospack.get_path('tiago_navigation') + "/train_results/mean_reward.txt", 'a') as file:
                # Append the new data to the end of the file
                file.write(str(mean_reward/10) + "\n")
            mean_reward = 0  
        if x%10 == 0:
             with open(rospack.get_path('tiago_navigation') + "/train_results/eval_reward.txt", 'a') as file:
                # Append the new data to the end of the file
                file.write(str(cumulated_reward) + "\n")
        if x%10 != 0 :
            ddpg_net.update()   
        if x%100 == 0:     
            ddpg_net.save(filename + str(x), pkg_path)
                 

    #save network weight
    ddpg_net.save(filename , pkg_path)
   
    #close environment
    env.close()
    
   