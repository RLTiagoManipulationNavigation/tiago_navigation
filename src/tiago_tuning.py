import gymnasium as gym
import numpy as np
import time
from gymnasium import wrappers
from tiago_navigation.ddpg_finetun import DDPG
from sklearn.model_selection import ParameterGrid
import rospy
import rospkg
import torch
import os
import keyboard

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

def eval_policy(ddpg , env):
    # Starts the main training loop: the one about the episodes to do
    for x in range(1 , 10):

        cumulated_reward = 0
        # Init new episode with reset the simulation
        curr_observation , info = env.reset()

        for i in range(1000):
            #generate current state from current observation
            state = torch.from_numpy(np.array(curr_observation)).float()
            waypoints = torch.from_numpy(np.array(info["waypoints"]).flatten()).float()
            action = ddpg_net(state ,waypoints , False).detach().numpy()

            rospy.loginfo("Next action is:%d" + str(action) + " with linear velocity : " + str(action[0]) + " and angular velocity : " + str(action[1]))
            #with open(rospack.get_path('tiago_navigation') + "/tuning_results/action/action_value"+ str(actor_params) +"_"+ str(critic_params)+".txt", 'a') as file:
                # Append the new data to the end of the file
                #file.write("At episode : " + str(x) + "Next action is:%d" + str(action) + "\n")  

            #obtain the results of the action obtained
            observation, reward, terminated , truncated , info = env.step(action)
            cumulated_reward += reward
                    
            #terminate episode if Tiago crash 
            if terminated or truncated:
                break

            curr_observation = observation
            eval_mean += cumulated_reward
            rospy.loginfo("actor value : " + str(actor_params) + " , critic value : " + str(critic_params) + " , reward is : " + str(cumulated_reward))
            # Open the file in append mode ('a')
            #with open(rospack.get_path('tiago_navigation') + "/tuning_results/eval/eval_value"+ str(actor_params) +"_"+ str(critic_params)+".txt", 'a') as file:
                # Append the new data to the end of the file
                #file.write("At episode : " + str(x) + " , obtain reward : " + str(cumulated_reward) + "\n")  
                #if truncated and info.get("goal_reach" , False):
                    #file.write("Reach goal \n")


if __name__ == '__main__':

    spatial_flag = False
    lr_range = [0.01 , 0.0001]
    weight_decay = []
    batch_size = [64 , 128 , 256]

    rospy.init_node('tuning_DDPG', anonymous=True, log_level=rospy.INFO)

    env = gym.make('TiagoNavigation-v0')
    rospy.logdebug("Gym env done!")
    
    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('tiago_navigation') + '/tuning_results'
    rospy.loginfo('Moder are savend into folder : ' + pkg_path)
    filename = 'DDPG'
    actor_lr = lr_range[]
    for actor_params in ParameterGrid(actor_param_grid):
        for critic_params in ParameterGrid(critic_param_grid):

            #variable used into training phase
            update_step = 0
            init_update = 0
            mean_reward = 0

            rospy.logerr("Start new set of actor param : " + str(actor_params) + " , and critic param: " + str(critic_params))

            #initialize DDPG net
            ddpg_net = DDPG(env , actor_params , critic_params , spatial_flag)

            # Starts the main training loop: the one about the episodes to do
            for x in range(1 , 100):

                rospy.logwarn("############### START EPISODE=>" + str(x))

                cumulated_reward = 0
                # Init new episode with reset the simulation
                curr_observation , info = env.reset()

                for i in range(1000):
                    #generate current state from current observation
                    state = torch.from_numpy(np.array(curr_observation)).float()
                    waypoints = torch.from_numpy(np.array(info["waypoints"]).flatten()).float()
                    action = ddpg_net(state ,waypoints).detach().numpy()

                    rospy.logdebug("Next action is:%d" + str(action) + " with linear velocity : " + str(action[0]) + " and angular velocity : " + str(action[1]))

                    #obtain the results of the action obtained
                    observation, reward, terminated , truncated , info = env.step(action)
                    cumulated_reward += reward
                    
                    #terminate episode if Tiago crash 
                    if terminated or truncated:
                        break
                    #add the data obtained from current step into replay buffer 
                    ddpg_net.update_buffer(curr_observation , observation , reward , action , [element for tuple in info["waypoints"] for element in tuple])
            
                    curr_observation = observation
                    
                if truncated and info.get("goal_reach" , False):
                    rospy.loginfo(str(info))
                    rospy.logwarn("\n Robot reach the objective with the following reward : " + str(cumulated_reward) )
                    # Open the file in append mode ('a')
                    #with open(rospack.get_path('tiago_navigation') + "/tuning_results/goal/value_goal"+ str(actor_params) +"_"+ str(critic_params)+".txt", 'a') as file:
                        # Append the new data to the end of the file
                        #file.write("At episode : " + str(x) + " , obtain reward : " + str(cumulated_reward) + "\n")
                mean_reward += cumulated_reward
                #if x%11 == 0 :
                    #with open(rospack.get_path('tiago_navigation') + "/tuning_results/mean_reward/mean_value.txt", 'a') as file:
                        # Append the new data to the end of the file
                        #file.write(str(mean_reward/10) + "\n")
                    #mean_reward = 0  
                
                ddpg_net.update(pkg_path)        
                        

            #save network weight
            #ddpg_net.save((filename + "_" + str(actor_params) + "_" + str(critic_params)), pkg_path)
            rospy.logerr("Start eval of the net")
            print("Program paused. Press 'Enter' to continue...")
            keyboard.wait('enter')
            print("Continuing program...")
            #ddpg_net.load((filename + "_" + str(actor_params) + "_" + str(critic_params)), pkg_path)
            #start evaluate
            eval_mean = 0
            

   
    #close environment
    env.close()
    
   