import torch
import torch.nn as nn
import copy
import rospy
from std_msgs.msg import String

def clone_model(original_model):
    # Create a deep copy of the model
    clone_model = copy.deepcopy(original_model)
    
    # Ensure the clone has the same device as the original
    device = next(original_model.parameters()).device
    clone_model.to(device)

    # Verify that the clone has the same structure and weights
    for (name1, param1), (name2, param2) in zip(original_model.named_parameters(), clone_model.named_parameters()):
      if not torch.all(param1.eq(param2)):
        rospy.logerr(f"Parameters are different !")
    
    
    return clone_model
def target_weight_update( target_network , network , update_coeff):
   
   for target_weight , weight in zip(target_network.parameters(), network.parameters()):
            # Update the weights of network B
            target_weight.data = update_coeff * weight.data + (1 - update_coeff) * target_weight.data





