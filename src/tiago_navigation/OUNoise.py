# Ornstein-Uhlenbeck Noise
# Author: Flood Sung
# Date: 2016.5.4
# Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
# --------------------------------------

import numpy as np
import numpy.random as nr
import rospy

class OUNoise:
    """docstring for OUNoise"""
    def __init__(self,action_dimension):
        self.action_dimension = action_dimension
        self.mu = rospy.get_param('/Ornstein_Uhlenbeck/mu')
        self.theta = rospy.get_param('/Ornstein_Uhlenbeck/teta')
        self.sigma = rospy.get_param('/Ornstein_Uhlenbeck/sigma')
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state

if __name__ == '__main__':
    ou = OUNoise(3)
    states = []
    for i in range(1000):
        states.append(ou.noise())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()