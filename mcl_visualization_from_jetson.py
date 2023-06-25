import gym
import numpy as np
from rsoccer_gym.Tracking.particle_filter_helpers import *
from rsoccer_gym.Utils.mcl_communication import *

if __name__ == "__main__":
    # Init Communication and Positions
    UDP = ParticlesReceiver()
    while True:
        has_msg, robot_position, particles, mcl_position, odometry_position, time_steps = UDP.recvMCLMessage()
        if has_msg: break

    # Initialize Environment
    env = gym.make('SSLVisionBlackout-v0',
                   n_particles = 100,
                   initial_position = robot_position)
    env.reset()

    while True:
        has_msg, robot_position, particles, mcl_position, odometry_position, time_steps = UDP.recvMCLMessage()
        if has_msg:        
            # update visualization:    
            env.update(robot_position, 
                       particles, 
                       mcl_position,
                       odometry_position, 
                       time_steps)
            print(f"Step nr: {time_steps}")
            env.render()
