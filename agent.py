import gym
import numpy as np
import rsoccer_gym
from rsoccer_gym.Controller import controllers

def get_action_from_desired_local_velocity(env, vx, vy, w):
    vx = vx/env.max_v
    vy = vy/env.max_v
    w = w/env.max_w
    return np.array([vx, vy, w], dtype=np.float32)

def get_action_from_desired_global_position(env, x, y):
    for robot in env.frame.robots_blue:
        v_x, v_y = controller_mpc.optimize(robot = env.robot_mpc, goal_xy = [x, y])

    return np.array([v_x, v_y, 0], dtype=np.float32)

if __name__ ==  "__main__":
    # Init MPC controller
    controller_mpc = controllers.MPC(horizon = 5)

    # Using SSL Single Agent env
    env = gym.make('SSLGoToBall-v0', 
                n_robots_blue=1, 
                n_robots_yellow=0,
                mm_deviation=0,
                angle_deviation=0)

    env.reset()

    # Run for 1 episode and print reward at the end
    for i in range(1):
        done = False
        while not done:
            # Step using random actions
            action = get_action_from_desired_global_position(env, 0, 0)
            next_state, reward, done, _ = env.step(action)
            env.render()
