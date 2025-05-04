import gymnasium as gym
import numpy as np
import time
from gymnasium import wrappers
from RL_Algorithms.ddpg import DDPG
from RL_Algorithms.TD3 import TD3
from RL_Algorithms.PPO import PPO
from RL_Algorithms.SAC import SAC
from tiago_navigation.utils import *
import rospy
import rospkg
import torch
import os
import math

# import our training environment
from openai_ros.task_envs.tiago import tiago_navigation


def policy(env , model , nsteps , algorithm_name , att_flag , evaluation = False ):
    curr_observation , info = env.reset()
    #rospy.loginfo(str(info['curr_pos']))
    #rospy.loginfo(str(info['curr_pos'][0][0]))
    cumulated_reward = 0
    curr_observation = gen_bounded_scan(curr_observation)
   
    if att_flag :
        spatial_input , _ , _ = generate_rays(curr_observation.copy() , n_discard_scan , initial_angle  , angle_increment)
    else:
        #spatial_input = curr_observation.copy()
        _ , _ , spatial_input = generate_rays(curr_observation.copy() , n_discard_scan , initial_angle  , angle_increment)
    debug_flag = rospy.get_param("/Training/debug")

    

    for n in range(nsteps):
        waypoints = [element for tuple in info["waypoints"] for element in tuple]
        goal_pos = [element for tuple in info["final_pos"] for element in tuple]   
        #plot(spatial_input)
        
        if evaluation:
            action = model(spatial_input.copy() , waypoints.copy() , goal_pos.copy() , False).detach().numpy()

            if debug_flag:
                with open(rospack.get_path('tiago_navigation') + "/data/" + str(algorithm_name) + "_linear_vel.txt", 'a') as file:
                    # Append the new data to the end of the file
                    file.write(str(action[0]))  
                with open(rospack.get_path('tiago_navigation') + "/data/" + str(algorithm_name) + "_angular_vel.txt", 'a') as file:
                    # Append the new data to the end of the file
                    file.write(str(action[1])) 
                with open(rospack.get_path('tiago_navigation') + "/data/" + str(algorithm_name) + "_position.txt", 'a') as file:
                    # Append the new data to the end of the file
                    file.write(str(info['curr_pos'][0]))  
        else:
            action = model(spatial_input.copy() , waypoints.copy() , goal_pos.copy()).detach().numpy()
    
        observation, reward , terminated , _ , info = env.step(action) 
        truncated = info["truncated"]
        cumulated_reward += reward
        observation = gen_bounded_scan(observation)
        prev_spatial_input = spatial_input.copy()
        
        if att_flag:
            spatial_input , _ , _ = generate_rays(observation.copy() , n_discard_scan , initial_angle  , angle_increment)
        else:
            #spatial_input = observation.copy()
            _ , _ , spatial_input = generate_rays(curr_observation.copy() , n_discard_scan , initial_angle  , angle_increment)
        if evaluation is False : 
            if terminated:
                model.update_buffer( prev_spatial_input, spatial_input , reward , action , waypoints , [element for tuple in info["waypoints"] for element in tuple] , goal_pos , [element for tuple in info["final_pos"] for element in tuple] , 0)
            else:
                model.update_buffer( prev_spatial_input, spatial_input , reward , action , waypoints , [element for tuple in info["waypoints"] for element in tuple] , goal_pos , [element for tuple in info["final_pos"] for element in tuple] , 1)
            #method call if want use original reading
            #ddpg_net.update_buffer( curr_observation , observation , reward , action , waypoints , [element for tuple in info["waypoints"] for element in tuple] , goal_pos , [element for tuple in info["final_pos"] for element in tuple])

        #terminate episode if Tiago crash 
        if terminated or truncated:
            if evaluation is False:
                with open(rospack.get_path('tiago_navigation') + "/data/reward.txt", 'a') as file:
                    # Append the new data to the end of the file
                    file.write(str(cumulated_reward/n) + "\n")  
            break
        else:   
            if debug_flag and evaluation :
                with open(rospack.get_path('tiago_navigation') + "/data/" + str(algorithm_name) + "_linear_vel.txt", 'a') as file:
                    # Append the new data to the end of the file
                    file.write(",")  
                with open(rospack.get_path('tiago_navigation') + "/data/" + str(algorithm_name) + "_angular_vel.txt", 'a') as file:
                    # Append the new data to the end of the file
                    file.write(",")   
                with open(rospack.get_path('tiago_navigation') + "/data/" + str(algorithm_name) + "_position.txt", 'a') as file:
                    # Append the new data to the end of the file
                    file.write(",")  

    #curr_observation = observation

    if evaluation is False :
        if truncated and info.get("goal_reach" , False):
            rospy.loginfo(str(info))
            rospy.logwarn("\n Robot reach the objective with the following reward : " + str(cumulated_reward) )
            # Open the file in append mode ('a')
            with open(rospack.get_path('tiago_navigation') + "/data/reward_goal.txt", 'a') as file:
                # Append the new data to the end of the file
                file.write("At episode : " + str(x) + " , obtain reward : " + str(cumulated_reward) + "\n")
    else:
        if debug_flag:
            with open(rospack.get_path('tiago_navigation') + "/data/" + str(algorithm_name) + "_linear_vel.txt", 'a') as file:
                # Append the new data to the end of the file
                file.write("\n")  
            with open(rospack.get_path('tiago_navigation') + "/data/" + str(algorithm_name) + "_angular_vel.txt", 'a') as file:
                # Append the new data to the end of the file
                file.write("\n")  
            with open(rospack.get_path('tiago_navigation') + "/data/" + str(algorithm_name) + "_position.txt", 'a') as file:
                # Append the new data to the end of the file
                file.write("\n") 
            with open(rospack.get_path('tiago_navigation') + "/data/" + str(algorithm_name) + "_attention_score.txt", 'a') as file:
                # Append the new data to the end of the file
                file.write("---\n")  

    if truncated and info.get("goal_reach" , False):
        return 1
    else:
        return 0

if __name__ == '__main__':

    rospy.init_node('tiago_DDPG', anonymous=True, log_level=rospy.INFO)

    env = gym.make('TiagoNavigation-v0')
    rospy.logdebug("Gym env done!")
    #flag used for dermine if use DDPG or TD3 algorithm 
    upload_weight = False
    upload_partial_weight = False
    spatial_att_flag = rospy.get_param("/Training/attention_module_flag")
    scan_type = rospy.get_param("/Training/scan_type") # differnet version of scan type : raw for raw scan , polar for polar version and cartesian
    if scan_type == 'raw' or scan_type == 'polar' or scan_type == 'cartesian' :
        rospy.logwarn(" Scan type is : " + str(scan_type) )
    else :
        rospy.logerr(" Error scan type not valid !")
        scan_type = 'raw'
    if rospy.get_param("/Training/debug"):
        n_evaluation = 5
        success_limit = 1.0
        n_eval_iter = 1
    else:
        n_evaluation = 5
        success_limit = 0.7
        n_eval_iter = 1
    test_execution = rospy.get_param("/Training/test")
    algorithm_name = rospy.get_param("/Training/algorithm")
    
    # Set the logging system
    rospack = rospkg.RosPack()
    train_result = rospack.get_path('tiago_navigation') + '/train_results'
    data_folder = rospack.get_path('tiago_navigation') + '/data'
    rospy.loginfo('Moder are savend into folder : ' + train_result)
    filename = 'DDPG_CURR'

    #set algorithm class
    if algorithm_name == "DDPG":
        model = DDPG(env)
    elif algorithm_name == "TD3":
        model = TD3(env)
    elif algorithm_name == "PPO":
        model = PPO(env)
    elif algorithm_name == "SAC":
        model = SAC(env)
    else:
        rospy.logerr("Algorithm not valid !")
    
    remove_file(rospack.get_path('tiago_navigation') + "/data/" , "reward_goal.txt")
    if upload_weight is False :
        remove_file(rospack.get_path('tiago_navigation') + "/data/" , "reward.txt")
    remove_file(rospack.get_path('tiago_navigation') + "/data/" , "mean_reward.txt")
    remove_file(rospack.get_path('tiago_navigation') + "/data/" , "critic_loss.txt")
    remove_file(rospack.get_path('tiago_navigation') + "/data/" , "actor_loss.txt")
    remove_file(rospack.get_path('tiago_navigation') + "/data/" , "critic_diff.txt")
    remove_file(rospack.get_path('tiago_navigation') + "/data/" , "critic_q.txt")
    remove_file(rospack.get_path('tiago_navigation') + "/data/" , str(algorithm_name) + "_angular_vel.txt")
    remove_file(rospack.get_path('tiago_navigation') + "/data/" , str(algorithm_name) + "_linear_vel.txt")
    remove_file(rospack.get_path('tiago_navigation') + "/data/" , str(algorithm_name) + "_attention_score.txt")
    remove_file(rospack.get_path('tiago_navigation') + "/data/" , str(algorithm_name) + "_position.txt")

    #value for generate the correct version of laser scan 
    initial_angle = rospy.get_param("/Tiago/initial_angle")
    angle_increment = rospy.get_param("/Tiago/angle_increment")
    n_discard_scan = rospy.get_param("/Tiago/remove_scan")
    bounded_ray = rospy.get_param("/Training/max_ray_value")
    #variable used into training phase
    update_step = 0
    init_update = 0
    mean_reward = 0

    #init training parameters
    nepisodes = rospy.get_param("/Training/nepisodes")
    nsteps = rospy.get_param("/Training/nsteps")

    #tagd = TAGD()
    if upload_weight:
        #upload network weight
        model.load(filename , train_result)   
    if upload_partial_weight:
        model.load("DDPG_PARTIAL" , train_result)

    #make training    
    if not test_execution:
        # Starts the main training loop: the one about the episodes to do
        for x in range(1 , nepisodes + 1):
        
            if x%n_evaluation == 0:
                rospy.logwarn("############## START POLICY EVAL")
                success_rate = 0
                for i in range(n_eval_iter):
                    rospy.logwarn("start policy evaluation episode n : " + str(i))
                    success_rate += policy(env , model , nsteps , algorithm_name , spatial_att_flag , True )
                model.save("DDPG_PARTIAL" , train_result)
                success_rate = success_rate/n_eval_iter
                rospy.loginfo("Success Rate obtained : " + str(success_rate))
                if success_rate >= success_limit:
                    rospy.logwarn("Success !")
                    #save network weight
                    model.save(filename , train_result)
                    break
            else:   
                rospy.logwarn("############### START EPISODE=>" + str(x))
                policy(env , model , nsteps , algorithm_name ,spatial_att_flag)
                model.update(data_folder)   
    else :
        policy(env , model , nsteps , spatial_att_flag , True )
   
    #close environment
    env.close()    