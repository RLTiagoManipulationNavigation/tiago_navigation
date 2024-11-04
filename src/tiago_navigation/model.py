import torch
import torch.nn as nn
import torch.nn.functional as F
import rospy

#######################################
#                                     #
#       DDPG ARCHITECTURE             #
#                                     #
#######################################

class ActorNet(nn.Module):

  def __init__(self , input_size):
    super().__init__()

    self.input_size = input_size 

    #input layer
    self.mlp_input = MLP(input_size  , 128)
    #hidden layer
    self.mlp_hid1 = MLP(128 , 64)
    self.mlp_hid2 = MLP(64 , 64)
    #output layer
    self.mlp_output = MLP(64 , 2 , final_layer=True)


  def forward(self  , input):
        
    out =  self.mlp_input(input)
    out = self.mlp_hid1(out)
    out = self.mlp_hid2(out)  
    out = self.mlp_output(out)
    
    # Bound the outputs
    output1 = 0.1*torch.sigmoid(out[:, 0]).unsqueeze(1)  # Bound to (0, 1)
    output2 = 0.3 * torch.tanh(out[:, 1]).unsqueeze(1)  # Bound to (-3, 3)
        
    return torch.cat([output1, output2], dim=1)
    #return out

class CriticNet(nn.Module):
   
  def __init__(self , input_size):
    super().__init__()

    self.input_size = input_size 

    #input layer
    self.mlp_input = MLP(input_size  , 128)
    #hidden layer
    self.mlp_hid1 = MLP(128 , 64)
    self.mlp_hid2 = MLP(64 , 64)
    #output layer
    self.mlp_output = MLP(64 , 1 , final_layer=True)


  def forward(self  , state , action):
    input = torch.cat([state, action], dim=1)
    #rospy.logdebug("critic input dim : " + str(input.shape))
    out =  self.mlp_input(input)
    out = self.mlp_hid1(out)
    out = self.mlp_hid2(out)  
    out = self.mlp_output(out)

    return out
    

#######################################
#                                     #
#       ATTENTION ARCHITECTURE        #
#                                     #
#######################################    

#Embedding Network
class Embedding(nn.Module):
  def __init__(self , input_size):
    super().__init__()

    self.input_size = input_size 

    #input layer
    self.mlp_input = MLP(input_size  , 256)
    #hidden layer
    self.mlp_hid = MLP(256 , 128)
    #output layer
    self.mlp_output = MLP(128 , 64)


  def forward(self  , input):
    out =  self.mlp_input(input)
    out = self.mlp_hid(out)  
    out = self.mlp_output(out)

    return out
  
  
#Feature Network
class Feature(nn.Module):
  def __init__(self , input_size):
    super().__init__()

    self.input_size = input_size 

    #input layer
    self.mlp_input = MLP(input_size  , 80)
    #hidden layer
    self.mlp_hid = MLP(80 , 50)
    #output layer
    self.mlp_output = MLP(50 , 30)


  def forward(self  , input):
        
    out =  self.mlp_input(input)
    out = self.mlp_hid(out)  
    out = self.mlp_output(out)

    return out
  
#Score Network
class Score(nn.Module):
  def __init__(self , input_size):
    super().__init__()

    self.input_size = input_size 

    #input layer
    self.mlp_input = MLP(input_size  , 60)
    #hidden layer
    self.mlp_hid = MLP(60 , 50)
    #output layer
    self.mlp_output = MLP(50 , 1)


  def forward(self  , input):
        
    out =  self.mlp_input(input)
    out = self.mlp_hid(out)  
    out = self.mlp_output(out)

    return out
  

class MLP(nn.Module):
    def __init__(self, input_size, output_size, final_layer=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Linear(self.input_size, self.output_size)

        # Apply different initialization depending on whether it's the final layer
        if final_layer:
            # Apply the DDPG paper's specific initialization for the final layers
            if output_size == 1:  # Critic's final layer
                nn.init.uniform_(self.fc.weight, -3e-4, 3e-4)
                nn.init.uniform_(self.fc.bias, -3e-4, 3e-4)
            else:  # Actor's final layer
                nn.init.uniform_(self.fc.weight, -3e-3, 3e-3)
                nn.init.uniform_(self.fc.bias, -3e-3, 3e-3)
        else:
            # Xavier initialization for hidden layers
            nn.init.xavier_uniform_(self.fc.weight)
            #nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='leaky_relu')
            nn.init.zeros_(self.fc.bias)

    def forward(self, input):
        #return F.leaky_relu(self.fc(input))
        return F.leaky_relu(self.fc(input))

"""            
# Multi Layer Perceptron
class MLP(nn.Module):
  def __init__(self, input_size, output_size):
    super().__init__()
    self.input_size   = input_size
    self.output_size  = output_size
    self.fc = nn.Linear(self.input_size, self.output_size)
  def forward(self, input):
    return F.relu(self.fc(input))
"""
  