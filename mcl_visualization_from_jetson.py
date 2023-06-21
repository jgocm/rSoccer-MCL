import gym
import numpy as np
from rsoccer_gym.Tracking.particle_filter_helpers import *
from rsoccer_gym.Utils.mcl_communication import *

if __name__ == "__main__":
    # Init Communication and Positions
    UDP = ParticlesReceiver()
    particles = []
    while True:
        has_msg, particles = UDP.recvMCLMessage()
        if has_msg: break

    # Using VSS Single Agent env
    env = gym.make('SSLVisionBlackout-v0',
                   n_particles = 100,
                   initial_position = [0, 0, 0])
    env.reset()

    while True:
        has_msg, particles = UDP.recvMCLMessage()
        if has_msg:        
            # update visualization:    
            env.update([0, 0, 0], 
                       particles, 
                       [0, 0, 0],
                       [0, 0, 0], 
                       1)
        
        env.render()
