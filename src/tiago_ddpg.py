import gymnasium as gym
import numpy as np
import time
from gymnasium import wrappers
from ddpg.ddpg import DDPG
import rospy
import rospkg
import torch

# import our training environment
from openai_ros.task_envs.tiago import tiago_navigation

if __name__ == '__main__':

    rospy.init_node('tiago_DDPG', anonymous=True, log_level=rospy.DEBUG)

    env = gym.make('TiagoNavigation-v0')
    rospy.loginfo("Gym env done!")
    
    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('ddpg') + '/train_results'
    rospy.loginfo('Moder are savend into folder : ' + pkg_path)
    outdir = '/training_results/DDPG'
    filename = 'DDPG'
    #env = wrappers.Monitor(env, outdir, force=True)
    rospy.loginfo("Monitor Wrapper started")

    update_step = 100

    last_time_steps = np.ndarray(0)

    #init training parameters
    nepisodes = rospy.get_param("/Training/nepisodes")
    nsteps = rospy.get_param("/Training/nsteps")

    rospy.logdebug("observation dim :" + str(env.observation_space.shape[0]))
    rospy.logdebug("action dim :" + str(env.action_space.shape[0]))

    #initialize the algorithm that we are going to use for learning
    ddpg_net = DDPG(env)

    highest_reward = -float('inf')

    # Starts the main training loop: the one about the episodes to do
    for x in range(nepisodes):
        rospy.logdebug("############### START EPISODE=>" + str(x))

        cumulated_reward = 0
        # Initialize the environment and get first state of the robot
        curr_observation , _ = env.reset()
        for i in range(nsteps):
            #generate current state from current observation
            state = torch.from_numpy(np.array(curr_observation)).float()
            #rospy.logwarn("############### Start Step=>" + str(i))
            # Run networj for obtain the action 
            action = ddpg_net(state).detach().numpy()
            action = action.squeeze(0) 

            rospy.logdebug("Next action is:%d" + str(action) + " with linear velocity : " + str(action[0]) + " and angular velocity : " + str(action[1]))

            #obtain the results of the action obtained
            observation, reward, terminated , truncated , info = env.step(action)

            cumulated_reward += reward
            
            #terminate episode if Tiago crash 
            if terminated :
                break
            #add the data obtained from current step into replay buffer 
            
            ddpg_net.update_buffer(curr_observation , observation , reward , action)
            """
            if i%update_step == 0 :
                critic_loss , actor_loss = ddpg_net.update()
                rospy.loginfo("\n Update of stepss : " + str(i) 
                      + " , critic loss of current episode is : " + str(critic_loss)
                      + " , actor loss of current episode is : " + str(actor_loss))
            """

            curr_observation = observation
            

        #weight update
        
        critic_loss , actor_loss = ddpg_net.update()
        rospy.loginfo("\n Result of episodes " + str(x) 
                      #+ " , episode reward : " + str(cumulated_reward) 
                      + " , critic loss of current episode is : " + str(critic_loss)
                      + " , actor loss of current episode is : " + str(actor_loss))
    #save network weight
    ddpg_net.save(filename , pkg_path)
    #close environment
    env.close()
    

   