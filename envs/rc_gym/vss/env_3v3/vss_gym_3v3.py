import math
import random

import gym
import numpy as np
from rc_gym.Entities import Frame, Robot
from rc_gym.vss.vss_gym_base import VSSBaseEnv
from rc_gym.Utils import distance, normVt, normVx, normX


class VSS3v3Env(VSSBaseEnv):
    """
    Description:
        This environment controls a single robot soccer in VSS League 3v3 match
    Observation:
        Type: Box(53)
        Num     Observation normalized
        0        Ball X
        1        Ball Y
        2        Ball Vx
        3        Ball Vy
        x        id i Blue Robot target_x
        x        id i Blue Robot target_y
        x        id i Blue Robot X
        x        id i Blue Robot Y
        x        id i Blue Robot sin(theta)
        x        id i Blue Robot cos(theta)
        x        id i Blue Robot Vx
        x        id i Blue Robot Vy
        x        id i Blue Robot v_theta
        x        id i Yellow Robot X
        x        id i Yellow Robot Y
        x        id i Yellow Robot sin(theta)
        x        id i Yellow Robot cos(theta)
        x        id i Yellow Robot Vx
        x        id i Yellow Robot Vy
        x        id i Yellow Robot v_theta
    Actions:
        Type: Box(2, )
        Num     Action
        0       id 0 Blue Angular Speed (%)
        1       id 0 Blue Linear Speed (%)
    Reward:
    Starting State:
        TODO
    Episode Termination:
        Match time or goal
    """

    def __init__(self):
        super().__init__(field_type=0, n_robots_blue=3, n_robots_yellow=3,
                         time_step=0.032)

        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-0, high=1, shape=(53, ), dtype=np.float32)
        self.matches_played = 0
        print('Environment initialized')

    def _frame_to_observations(self):

        observation = []

        width = 1.3/2.0
        lenght = (1.5/2.0) + 0.1
        observation.append(1 - (self.frame.time / 300))
        observation.append(normX(lenght + self.frame.ball.x))
        observation.append(normX(width - self.frame.ball.y))
        observation.append(normVx(self.frame.ball.v_x))
        observation.append(normVx(-self.frame.ball.v_y))

        for i in range(self.n_robots_blue):
            if self.actions != None:
                target_theta = (np.deg2rad(self.frame.robots_blue[i].theta) + (
                    self.actions[i][0] * self.simulator.angular_speed_range
                    * self.time_step * 5.3))

                target_x = self.frame.robots_blue[i].x + (
                    self.actions[i][1] * self.simulator.linear_speed_range
                    * np.cos(target_theta)) * 0.52  # * self.time_step

                target_y = self.frame.robots_blue[i].y + (
                    self.actions[i][1] * self.simulator.linear_speed_range
                    * np.sin(target_theta)) * 0.52  # * self.time_step
            else:
                target_x = self.frame.robots_blue[i].x
                target_y = self.frame.robots_blue[i].y
            observation.append(normX(lenght + target_x))
            observation.append(normX(width - target_y))
            observation.append(normX(lenght + self.frame.robots_blue[i].x))
            observation.append(normX(width - self.frame.robots_blue[i].y))
            observation.append(
                np.sin(np.deg2rad(-self.frame.robots_blue[i].theta)))
            observation.append(
                np.cos(np.deg2rad(-self.frame.robots_blue[i].theta)))
            observation.append(normVx(self.frame.robots_blue[i].v_x))
            observation.append(normVx(-self.frame.robots_blue[i].v_y))
            observation.append(normVt(self.frame.robots_blue[i].v_theta))

        for i in range(self.n_robots_yellow):
            observation.append(normX(lenght + self.frame.robots_yellow[i].x))
            observation.append(normX(width - self.frame.robots_yellow[i].y))
            observation.append(
                np.sin(np.deg2rad(-self.frame.robots_yellow[i].theta)))
            observation.append(
                np.cos(np.deg2rad(-self.frame.robots_yellow[i].theta)))
            observation.append(normVx(self.frame.robots_yellow[i].v_x))
            observation.append(normVx(-self.frame.robots_yellow[i].v_y))
            observation.append(normVt(self.frame.robots_yellow[i].v_theta))

        return np.array(observation)

    def _get_commands(self, actions):
        commands = []
        self.actions = {}

        self.actions[0] = actions
        v_wheel1, v_wheel2 = self._actions_to_v_wheels(actions)
        commands.append(Robot(yellow=False, id=0, v_wheel1=v_wheel1,
                              v_wheel2=v_wheel2))

        # Send random commands to the other robots
        for i in range(1, 3):
            # actions = self.action_space.sample()
            actions = [0., 0.]
            self.actions[i] = actions
            v_wheel1, v_wheel2 = self._actions_to_v_wheels(actions)
            commands.append(Robot(yellow=False, id=i, v_wheel1=v_wheel1,
                                  v_wheel2=v_wheel2))
        for i in range(3):
            # actions = self.action_space.sample()
            actions = [0., 0.]
            v_wheel1, v_wheel2 = self._actions_to_v_wheels(actions)
            commands.append(Robot(yellow=True, id=i, v_wheel1=v_wheel1,
                                  v_wheel2=v_wheel2))

        return commands

    def _calculate_reward_and_done(self):
        reward = 0
        goal = False
        w_move = 0.2
        w_ball_grad = 0.8
        w_energy = 2e-4
        if self.reward_shaping_total == None:
            self.reward_shaping_total = {'goal_score': 0, 'move': 0,
                                         'ball_grad': 0, 'energy': 0,
                                         'goals_blue': 0, 'goals_yellow': 0}

        # Check if goal ocurred
        if self.frame.ball.x > (self.field_params['field_length'] / 2):
            # print('GOAL BLUE')
            self.reward_shaping_total['goal_score'] += 10
            self.reward_shaping_total['goals_blue'] += 1
            reward = 10
            goal = True
        elif self.frame.ball.x < -(self.field_params['field_length'] / 2):
            # print('GOAL YELLOW')
            self.reward_shaping_total['goal_score'] -= 10
            self.reward_shaping_total['goals_yellow'] += 1
            reward = -10
            goal = True
        else:
            # Calculate ball potential
            width = 1.3/2.0
            lenght = (1.5/2.0) + 0.1
            dx_d = 0 - (lenght + self.frame.ball.x) * \
                100  # distance to defence
            dx_a = 170.0 - (lenght + self.frame.ball.x) * \
                100  # distance to attack
            dy = 65.0 - (width - self.frame.ball.y) * 100
            ball_potential = ((-math.sqrt(dx_a ** 2 + 2 * dy ** 2)
                               + math.sqrt(dx_d ** 2 + 2 * dy ** 2)) / 170 - 1) / 2

            if self.last_frame is not None:
                if self.previous_ball_potential is not None:
                    grad_ball_potential = np.clip(((ball_potential
                                                   - self.previous_ball_potential) * 3/ self.time_step),
                                                  -1.0, 1.0)
                else:
                    grad_ball_potential = 0

                self.previous_ball_potential = ball_potential

                # Move Reward : Reward the robot for moving in ball direction
                prev_dist_robot_ball = distance(
                    (self.last_frame.ball.x, self.last_frame.ball.y),
                    (self.last_frame.robots_blue[0].x,
                     self.last_frame.robots_blue[0].y)
                ) * 100
                dist_robot_ball = distance(
                    (self.frame.ball.x, self.frame.ball.y),
                    (self.frame.robots_blue[0].x, self.frame.robots_blue[0].y)
                ) * 100

                move_reward = prev_dist_robot_ball - dist_robot_ball
                move_reward = np.clip(move_reward / (40 * self.time_step),
                                      -1.0, 1.0)

                energy_penalty = - \
                    (abs(self.sent_commands[0].v_wheel1 / 0.026) +
                     abs(self.sent_commands[0].v_wheel2 / 0.026))

                reward = w_move * move_reward + \
                    w_ball_grad * grad_ball_potential + \
                    w_energy * energy_penalty
                
                self.reward_shaping_total['move'] += w_move * move_reward
                self.reward_shaping_total['ball_grad'] +=\
                    w_ball_grad * grad_ball_potential
                self.reward_shaping_total['energy'] += w_energy * energy_penalty
                

            self.last_frame = self.frame

        if goal:
            initial_pos_frame: Frame = self._get_initial_positions_frame()
            self.simulator.reset(initial_pos_frame)
            self.frame = self.simulator.get_frame()
            self.last_frame = None

        done = self.steps * self.time_step >= 300

        if done and self.summary_writer != None:
            self.summary_writer.add_scalar("rw/goal_score", self.reward_shaping_total['goal_score'], self.matches_played)
            self.summary_writer.add_scalar("rw/move", self.reward_shaping_total['move'], self.matches_played)
            self.summary_writer.add_scalar("rw/ball_grad", self.reward_shaping_total['ball_grad'], self.matches_played)
            self.summary_writer.add_scalar("rw/energy", self.reward_shaping_total['energy'], self.matches_played)
            self.summary_writer.add_scalar("rw/goals_blue", self.reward_shaping_total['goals_blue'], self.matches_played)
            self.summary_writer.add_scalar("rw/goals_yellow", self.reward_shaping_total['goals_yellow'], self.matches_played)

        return reward, done

    def _get_initial_positions_frame(self):
        field_half_length = self.field_params['field_length'] / 2
        field_half_width = self.field_params['field_width'] / 2

        def x(): return random.uniform(-field_half_length + 0.1,
                                       field_half_length - 0.1)

        def y(): return random.uniform(-field_half_width + 0.1,
                                       field_half_width - 0.1)

        def theta(): return random.uniform(-180, 180)

        pos_frame: Frame = Frame()

        pos_frame.ball.x = x()
        pos_frame.ball.y = y()
        pos_frame.ball.v_x = 0.
        pos_frame.ball.v_y = 0.

        pos_frame.robots_blue[0] = Robot(x=x(), y=y(), theta=theta())
        pos_frame.robots_blue[1] = Robot(x=x(), y=y(), theta=theta())
        pos_frame.robots_blue[2] = Robot(x=x(), y=y(), theta=theta())

        pos_frame.robots_yellow[0] = Robot(x=x(), y=y(), theta=theta())
        pos_frame.robots_yellow[1] = Robot(x=x(), y=y(), theta=theta())
        pos_frame.robots_yellow[2] = Robot(x=x(), y=y(), theta=theta())

        return pos_frame

    def _actions_to_v_wheels(self, actions):
        linear_speed_desired = actions[1] * self.simulator.linear_speed_range
        angular_speed_desired = actions[0] * self.simulator.angular_speed_range
        robot_radius = 0.038

        left_wheel_speed = linear_speed_desired - \
            (robot_radius * angular_speed_desired)
        right_wheel_speed = linear_speed_desired + \
            (robot_radius * angular_speed_desired)

        # Deadzone
        if -0.208 < left_wheel_speed < 0.208:
            left_wheel_speed = 0

        if -0.208 < right_wheel_speed < 0.208:
            right_wheel_speed = 0

        left_wheel_speed, right_wheel_speed = np.clip(
            (left_wheel_speed, right_wheel_speed), -2.6, 2.6)

        return left_wheel_speed, right_wheel_speed

    def set_writer(self, writer):
        self.summary_writer = writer
        
    def set_matches_played(self, matches):
        self.matches_played = matches
