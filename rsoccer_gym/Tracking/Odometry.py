import math
import numpy as np

class Odometry:
    '''
    Defines calculations for inertial odometry tracking
    '''
    def __init__(
                self,
                initial_position = np.array([0,0,0])
                ):
       self.last_position = initial_position
       self.current_position = initial_position
       self.movement = np.array([0,0,0])

    def rotate_vector(self, x, y, angle):
        '''
        Rotates x, y vector through angle.
        x: x coordinate
        y: y coordinate
        angle: rotation angle in radians
        '''
        rotated_x = x*math.cos(angle) - y*math.sin(angle)
        rotated_y = x*math.sin(angle) + y*math.cos(angle)
        return np.array([rotated_x, rotated_y])

    def rotate_to_local(self, x, y, robot_w):
        return self.rotate_vector(x, y, -robot_w)

    def rotate_to_global(self, x, y, robot_w):
        return self.rotate_vector(x, y, robot_w)
    
    def compute_local_movement(self, global_movement):
        dx, dy, dtheta = global_movement
        local_dx, local_dy = self.rotate_to_local(dx, dy, self.current_position[2])
        return np.array([local_dx, local_dy, dtheta])

    def update(self, new_pose):
        '''
        Updates state and movement from most recent odometry tracking.

        new_pose: 3-dimensional numpy array with x, y and angle (radians)
        '''
        global_movement = new_pose-self.current_position
        self.movement = self.compute_local_movement(global_movement)
        self.last_position = self.current_position
        self.current_position = new_pose

    def deg2rad(self, pose):
        '''
        Converts a pose's angle from degree to radians
        '''
        pose[2] = math.radians(pose[2])
        return pose
    
    def rad2deg(self, pose):
        '''
        Converts a pose's angle from radians to degrees
        '''
        pose[2] = math.degrees(pose[2])
        return pose
    
if __name__ == "__main__":

    odometry = Odometry()
    
