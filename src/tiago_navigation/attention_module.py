import torch 

import numpy as np
import rospy
import torch.nn as nn
from tiago_navigation.model import Embedding , Feature , Score
import torch.nn.functional as F

class Spatial_Attention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.name = 'Spatial Attention'
        self.n_section = rospy.get_param("/Spatial_Attention/n_sector_spatialatt")
        self.output_dim = rospy.get_param("/Spatial_Attention/embedding_output_size")
        self.Embedding = Embedding(input_dim)
        self.Score = Score(self.output_dim)
        self.Feature = Feature(self.output_dim)

    def forward(self, input):

        # Ensure input is a tensor (convert if necessary)
        if isinstance(input, list):
            input = torch.tensor(input, dtype=torch.float32)
    
        # Check if input is 1D (single sample); if so, add batch dimension
        if len(input.shape) == 1:
            input = input.unsqueeze(0)  # Shape becomes (1, input_dim)
        
        input_dim = input.shape[1]    # input dimension (e.g., lidar readings per sample)
        
        # Compute section size (each sample is divided into n_section parts)
        section_size = input_dim // self.n_section
        
        # Split input for each sample into sections
        # We need to split along the dimension 1 (input_dim), so use torch.split in that dimension
        sections = torch.split(input, section_size, dim=1)
        
        embeddings = []
        scores = []
        features = []
        
        for section in sections:
            #rospy.loginfo(" input sector : " + str(section))
            # Process each section in the batch
            ei = self.Embedding(section)  # Apply the embedding layer
            embeddings.append(ei)
            #rospy.loginfo(" embedding : " + str(ei))
            si = self.Score(ei)  # Apply the score layer
            scores.append(si)
            #rospy.loginfo(" score : " + str(si))
            fi = self.Feature(ei)  # Apply the feature layer
            features.append(fi)
            #rospy.loginfo(" feature : " + str(fi))
             #raw_input("Next Step...PRESS KEY")
            #rospy.sleep(100.0)
        
        # Stack along a new dimension (representing the sections) for each of embeddings, scores, features
        embeddings = torch.stack(embeddings, dim=1)  # Shape: (batch_size, n_section, output_dim)
        scores = torch.stack(scores, dim=1)  # Shape: (batch_size, n_section, 1)
        features = torch.stack(features, dim=1)  # Shape: (batch_size, n_section, output_dim)
        
        # Softmax normalization of scores across sections
        attention_weights = F.softmax(scores, dim=1)  # Normalize across sections, Shape: (batch_size, n_section, 1)
        
        # Weighted sum of features across sections
        weighted_features = features * attention_weights  # Element-wise multiplication, broadcasting over output_dim
        #rospy.logdebug("weighted features: " + str(weighted_features) + " tensor size: " + str(weighted_features.shape))
        
        # Sum across the sections (dim=1), leaving (batch_size, output_dim)
        output = torch.sum(weighted_features, dim=1)
        #rospy.logdebug("output: " + str(output) + " tensor size: " + str(output.shape))
        
        return output