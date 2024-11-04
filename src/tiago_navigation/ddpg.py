import torch 

import numpy as np
from tiago_navigation.model import (ActorNet , CriticNet)
from tiago_navigation.replay_buffer import Replay_Buffer
from tiago_navigation.attention_module import Spatial_Attention
import rospy
from tiago_navigation.OUNoise import OUNoise
from tiago_navigation.utils import clone_model , target_weight_update
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os

class DDPG(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.name = 'DDPG' # name for uploading results
        self.environment = env
        self.ddpg_input_dim = rospy.get_param("/Spatial_Attention/spatial_att_ourdim")
        self.spatial_input_size = rospy.get_param("/Spatial_Attention/input_spatial_size")

        #Setting environment param
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        #initialize Spatial Attention Network
        self.spatial_attention = Spatial_Attention(self.spatial_input_size)

        #initialize critic and actor network , giving in input dimension of input and output layer
        #self.actor_network = ActorNet(self.ddpg_input_dim)
        #self.critic_network = CriticNet(self.ddpg_input_dim + self.action_dim)

        self.actor_network = ActorNet(312)
        self.critic_network = CriticNet( 312 + self.action_dim)

        #initialize target networks 
        self.target_actor = clone_model(self.actor_network)
        self.target_critic = clone_model(self.critic_network)
        
        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)

        # initialize replay buffer
        self.replay_buffer = Replay_Buffer(rospy.get_param('/Training/buffer_size'))

        #initialize the optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor_network.parameters(), lr = 0.0001)#lr=rospy.get_param("/ADAM_param/actor"))
        self.critic_optimizer = torch.optim.Adam(self.critic_network.parameters() , lr = 0.0001)#lr=rospy.get_param("/ADAM_param/critic"))
        self.attention_optimizer = torch.optim.Adam(self.spatial_attention.parameters() , lr = 0.001)#lr=rospy.get_param("/ADAM_param/spatial_attention"))
        
        #initialize the target network update parameter
        self.target_update = rospy.get_param("/DDPG/soft_target_update")

        self.discount_factor = rospy.get_param("/DDPG/discount_factor")

        self.batch_size = rospy.get_param("/Training/batch_size")

    def forward(self , input ):
        """
        #rospy.logdebug("INPUT DDPG : " + str(input))
        attention_process = self.spatial_attention(input)
        #rospy.logdebug("ATTENTION OUTPUT : " + str(attention_process))
        action = self.actor_network(attention_process)
        #generate OUNoise 
        noise = torch.tensor(self.exploration_noise.noise()).view(1, 2)
        #rospy.logdebug("noise : " + str(noise) + " dimension : " + str(noise.shape))
        
        action = action + noise
        #rospy.logdebug("ACTION : " + str(action))
        """
        # Ensure input is a tensor (convert if necessary)
        if isinstance(input, list):
            input = torch.tensor(input, dtype=torch.float32)
    
        # Check if input is 1D (single sample); if so, add batch dimension
        if len(input.shape) == 1:
            input = input.unsqueeze(0)  # Shape becomes (1, input_dim)

        action = self.actor_network(input)
        #generate OUNoise 
        noise = torch.tensor(self.exploration_noise.noise()).view(1, 2)
        #rospy.logdebug("noise : " + str(noise) + " dimension : " + str(noise.shape))
        
        action = action + noise    

        return action 
    
    def update(self):
        if int(self.replay_buffer.count()) < int(self.batch_size) :
            return 'null' , 'null'
        #generate minibatch from replay buffer
        buffer_batch = self.replay_buffer.get_batch(self.batch_size)
        spatial_states, actions, rewards, next_spatial_states = zip(*buffer_batch)
        
        spatial_state_batch = torch.FloatTensor(np.array(spatial_states))
        action_batch = torch.FloatTensor(np.array(actions))
        reward_batch = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_spatial_state_batch = torch.FloatTensor(np.array(next_spatial_states))
        """
        # Process states through spatial attention
        state_batch = self.spatial_attention(spatial_state_batch)
        #rospy.loginfo("State batch :" + str(state_batch))
        next_state_batch = self.spatial_attention(next_spatial_state_batch)
        #rospy.loginfo("State batch next :" + str(next_state_batch))
        # Compute the target Q value
        with torch.no_grad():
            target_action = self.target_actor(next_state_batch)
            #rospy.loginfo("Target action :" + str(target_action))
            target_q = self.target_critic(next_state_batch, target_action)
            #rospy.loginfo("Target_q :" + str(target_q))
            target_q = reward_batch + self.discount_factor * target_q
        
        #rospy.loginfo("Error target_q :" + str(target_q))

        self.critic_optimizer.zero_grad()

        # Update Critic
        current_q = self.critic_network(state_batch, action_batch)
        #rospy.loginfo("Current q  :" + str(current_q))
        critic_loss = F.mse_loss(current_q, target_q)
        #rospy.loginfo("Critic loss :" + str(critic_loss))

        
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        #self.check_gradients(self.critic_network)
        

        actor_action = self.actor_network(state_batch)
        #rospy.loginfo("Actor action :" + str(actor_action))
        q_values = self.critic_network(state_batch, actor_action)
        #rospy.loginfo("Q values :" + str(q_values))
        
        #rospy.logdebug("Q values shape: %s, dtype: %s", q_values.shape, q_values.dtype)
        
        actor_loss = -q_values.mean()
        #rospy.loginfo("Actor loss  :" + str(actor_loss))
        
        if not isinstance(actor_loss, torch.Tensor):
            rospy.logerr("Actor loss is not a tensor. Type: %s", type(actor_loss))
            return
        
        if not actor_loss.requires_grad:
            rospy.logerr("Actor loss does not require grad. This will prevent backpropagation.")
            return

        #rospy.logdebug("Actor loss: %f", actor_loss.item())
        # Update Actor
        self.actor_optimizer.zero_grad()
        #self.actor_optimizer.zero_grad()
        self.attention_optimizer.zero_grad()
        
        
        try:
            actor_loss.backward()
        except RuntimeError as e:
            rospy.logerr("Error in backward pass: %s", str(e))
            return
        """
        """
        # Check the gradient flow through the actor's parameters
        for param in self.actor_network.parameters():
            if param.grad is not None:
                rospy.loginfo(str(param.grad.norm()))  # This should not be zero if gradients are flowing
        """
        
        # Compute the target Q value
        with torch.no_grad():
            target_action = self.target_actor(next_spatial_state_batch)
            #rospy.loginfo("Target action :" + str(target_action))
            target_q = self.target_critic(next_spatial_state_batch, target_action)
            #rospy.loginfo("Target_q :" + str(target_q))
            tot_target_q = reward_batch + self.discount_factor * target_q
        
        #rospy.loginfo("Error target_q :" + str(target_q))

        self.critic_optimizer.zero_grad()

        # Update Critic
        current_q = self.critic_network(spatial_state_batch, action_batch)
        #rospy.loginfo("Current q  :" + str(current_q))
        critic_loss = F.mse_loss(current_q, tot_target_q)
        #rospy.loginfo("Critic loss :" + str(critic_loss))

        
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        #self.check_gradients(self.critic_network)
        

        actor_action = self.actor_network(spatial_state_batch)
        #rospy.loginfo("Actor action :" + str(actor_action))
        q_values = self.critic_network(spatial_state_batch, actor_action)
        #rospy.loginfo("Q values :" + str(q_values))
        
        #rospy.logdebug("Q values shape: %s, dtype: %s", q_values.shape, q_values.dtype)
        
        actor_loss = -q_values.mean()
        #rospy.loginfo("Actor loss  :" + str(actor_loss))
        
        if not isinstance(actor_loss, torch.Tensor):
            rospy.logerr("Actor loss is not a tensor. Type: %s", type(actor_loss))
            return
        
        if not actor_loss.requires_grad:
            rospy.logerr("Actor loss does not require grad. This will prevent backpropagation.")
            return

        #rospy.logdebug("Actor loss: %f", actor_loss.item())
        # Update Actor
        self.actor_optimizer.zero_grad()
        #self.actor_optimizer.zero_grad()


        #self.attention_optimizer.zero_grad()
        
        
        try:
            actor_loss.backward()
        except RuntimeError as e:
            rospy.logerr("Error in backward pass: %s", str(e))
            return
        
        self.actor_optimizer.step()
        #self.attention_optimizer.step()


        #rospy.loginfo("Actor network")
        #self.check_gradients(self.actor_network)
        #rospy.loginfo("Attention network")
        #self.check_gradients(self.spatial_attention)

        # Update target networks
        target_weight_update(self.target_actor, self.actor_network, self.target_update)
        target_weight_update(self.target_critic, self.critic_network, self.target_update)

        return critic_loss , actor_loss
    
    # Diagnostic function
    def check_gradients(self , model):
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name}: grad norm = {param.grad.norm().item()}")
            else:
                print(f"{name}: No gradient")

    def update_buffer(self , state , new_state , reward , action ):
        self.replay_buffer.add( state, action , reward, new_state )   

    def save(self, filename , folder_path):
        torch.save(self.critic_network.state_dict(),  os.path.join(folder_path, filename + "_critic"))
        torch.save(self.critic_optimizer.state_dict(), os.path.join(folder_path,filename + "_critic_optimizer"))
        
        torch.save(self.actor_network.state_dict(), os.path.join(folder_path,filename + "_actor"))
        torch.save(self.actor_optimizer.state_dict(), os.path.join(folder_path,filename + "_actor_optimizer"))

        torch.save(self.spatial_attention.state_dict(), os.path.join(folder_path,filename + "_spatial_attention"))
        torch.save(self.attention_optimizer.state_dict(), os.path.join(folder_path,filename + "_spatial_attention_optimizer"))
        
    def load(self, filename):
        self.critic_network.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = clone_model(self.critic_network)
        
        self.actor_network.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = clone_model(self.actor_network)

        self.spatial_attention.load_state_dict(torch.load(filename + "_spatial_attention"))
        self.attention_optimizer.load_state_dict(torch.load(filename + "_spatial_attention_optimizer"))
     




