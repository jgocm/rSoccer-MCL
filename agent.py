import gym
import numpy as np
import math
import rsoccer_gym
from rsoccer_gym.Controller import controllers

def get_action_from_desired_local_velocity(env, vx, vy, w):
    vx = vx/env.max_v
    vy = vy/env.max_v
    w = w/env.max_w
    return np.array([vx, vy, w], dtype=np.float32)

def get_action_from_desired_global_position(env, x, y, angle):
    KP_ANLGE = 0.2
    MIN_ANGLE_TO_ROTATE = 3

    for robot in env.frame.robots_blue:
        v_x, v_y = controller_mpc.optimize(robot = env.robot_mpc, goal_xy = [x, y])

        angle_to_rotate = smallest_angle_diff(angle, env.robot_mpc.x[2])
        if abs(angle_to_rotate) < MIN_ANGLE_TO_ROTATE:
            v_w = 0
        else:
            v_w = -KP_ANLGE * angle_to_rotate
    return np.array([v_x/env.max_v, v_y/env.max_v, v_w/env.max_w], dtype=np.float32)

def smallest_angle_diff(angle_a: float, angle_b: float) -> float:
    """Returns the smallest angle difference between two angles"""
    angle: float = math_modularize(angle_b - angle_a, 2 * 180)
    if angle >= 180:
        angle -= 2 * 180
    elif angle < -180:
        angle += 2 * 180
    return angle

def math_modularize(value: float, mod: float) -> float:
    """Make a value modular between 0 and mod"""
    if not -mod <= value <= mod:
        value = math.fmod(value, mod)

    if value < 0:
        value += mod

    return value

if __name__ ==  "__main__":
    # Init  controller
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
            action = get_action_from_desired_global_position(env, 1.2, -1.1, 30)
            next_state, reward, done, _ = env.step(action)
            env.render()
