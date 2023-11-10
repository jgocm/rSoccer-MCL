import numpy as np
from rsoccer_gym.Controller.parameters import *
from rsoccer_gym.Kinematics.Kinematics import Robot

M_TO_MM = 1000

class OmnidirectionalRobot:
	def __init__(self, intial_x, intial_y, initial_theta):
		self.x = np.array([
							intial_x,
							intial_y,
							initial_theta
						  ])

		self.x_dot = np.array([
							0,
							0,
							0
						  ])

		self.wheel_speed = np.array([
										0,
										0,
										0,
										0
									])

		self.l = 0.081*M_TO_MM
		self.r = 0.02475*M_TO_MM
		self.kinematics = Robot(number_of_wheels = 4,
								wheel_radius = self.l,
								axis_length = self.r,
								wheels_alphas = [60, 135, -135, -60],
								wheels_betas = [0, 0, 0, 0],
								mm_deviation = 0,
								angle_deviation = 0)

	def set_wheel_velocity(self, w1, w2, w3, w4):
		self.wheel_speed = np.array([
										w1,
										w2,
										w3, 
										w4
									])
		self.x_dot = self.forward_kinematics()

	def set_robot_velocity(self, v_x, v_y, v_w):
		self.x_dot = np.array([v_x,
							   v_y,
							   v_w])
		self.wheel_speed = self.inverse_kinematics()

	def set_robot_position(self, x, y, theta):
		self.x = np.array([x,
						   y,
						   theta])
		
	def set_robot_state(self, robot):
		self.set_robot_position(robot.x, robot.y, robot.theta)
		self.set_robot_velocity(robot.v_x, robot.v_y, robot.v_theta)

	def forward_kinematics(self):
		return self.kinematics.get_forward_kinematics(self.wheel_speed)

	def inverse_kinematics(self):
		return self.kinematics.get_inverse_kinematics(self.x_dot)

	def update_state(self, dt):
		A = np.array([
						[1, 0, 0],
						[0, 1, 0],
						[0, 0, 1]
					])
		B = np.array([
						[dt,  0, 0],
						[ 0, dt, 0],
						[ 0,  0, dt]
					])

		vel = np.array([
						  self.x_dot[0],
						  self.x_dot[1],
						  self.x_dot[2]
						])
		
		global_vel = self.rotate_to_global(vel)

		self.x = A@self.x + B@global_vel

	def update(self, dt):
		self.wheel_speed[self.wheel_speed>MAX_WHEEL_ROT_SPEED_RAD] = MAX_WHEEL_ROT_SPEED_RAD
		self.wheel_speed[self.wheel_speed<MIN_WHEEL_ROT_SPEED_RAD] = MIN_WHEEL_ROT_SPEED_RAD
		self.x_dot = self.forward_kinematics()
		self.update_state(dt)
		self.wheel_speed = self.inverse_kinematics()

	def get_state(self):
		return self.x, self.x_dot
	
	def rotate_to_global(self, local_vector):
		theta = np.deg2rad(self.x[2])
		global_vector = np.array([
								 np.cos(theta)*local_vector[0] - np.sin(theta)*local_vector[1],
								 np.sin(theta)*local_vector[0] + np.cos(theta)*local_vector[1],
								 local_vector[2]
								])
		return global_vector

	def rotate_to_local(self, global_vector):
		theta = np.deg2rad(self.x[2])
		local_vector = np.array([
								 np.cos(theta)*global_vector[0] + np.sin(theta)*global_vector[1],
								 -np.sin(theta)*global_vector[0] + np.cos(theta)*global_vector[1],
								 -global_vector[2]
								])
		return local_vector