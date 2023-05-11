import numpy as np

def is_out_of_field(particle, x_min, x_max, y_min, y_max):
    '''
    Check if particle is out of field boundaries
    
    param: current field configurations
    return: True if particle is out of field boundaries
    '''
    if particle[1] < x_min or particle[1] > x_max or particle[2] < y_min or particle[2] > y_max:
        return True
    else:
        return False

def rotate_to_global(local_x, local_y, robot_w):
    theta = np.deg2rad(robot_w)
    global_x = local_x*np.cos(theta) - local_y*np.sin(theta)
    global_y = local_x*np.sin(theta) + local_y*np.cos(theta)
    return global_x, global_y

def add_move_noise(movement, deviation):
    standard_deviation_vector = np.abs(movement)*deviation
    return np.random.normal(movement, standard_deviation_vector, 3)

def limit_angle_degrees(angle):
    while angle > 180:
        angle -= 2*180
    while angle < -180:
        angle += 2*180
    return angle

class Pose3D:
    x: float
    y: float
    theta: float