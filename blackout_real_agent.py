import gym
import numpy as np
import cv2
from rsoccer_gym.Tracking.ParticleFilterBase import ParticleFilter, Particle
from rsoccer_gym.Tracking import ResamplingAlgorithms
from rsoccer_gym.Plotter.Plotter import RealTimePlotter
from rsoccer_gym.Perception.jetson_vision import JetsonVision
from rsoccer_gym.Tracking.Odometry import Odometry

def get_image_from_frame_nr(path_to_images_folder, frame_nr):
    dir = path_to_images_folder+f'/cam/{frame_nr}.png'
    img = cv2.imread(dir)
    return img

if __name__ == "__main__":
    import os
    from rsoccer_gym.Utils.load_localization_data import Read
    cwd = os.getcwd()

    n_particles = 100
    vertical_lines_nr = 1

    # CHOOSE SCENARIO
    scenario = 'sqr'
    lap = 2

    # LOAD DATA
    path = f'/home/rc-blackout/ssl-navigation-dataset/data/{scenario}_0{lap}'
    path_to_log = path+'/logs/processed.csv'
    data = Read(path_to_log, is_raw=False, degrees=False)
    time_steps = data.get_timesteps()
    frames = data.get_frames()
    has_goals = data.get_has_goals(remove_false_positives=True)
    goals = data.get_goals()

    # LOAD POSITION DATA
    position = data.get_position()
    odometry = data.get_odometry()

    # SET INITIAL ROBOT POSITION AND SEED
    initial_position = position[0]
    seed_radius = 1
    initial_position[2] = np.degrees(initial_position[2])

    # Using VSS Single Agent env
    env = gym.make('SSLVisionBlackout-v0',
                   n_particles = n_particles,
                   initial_position = [initial_position[0], initial_position[1], initial_position[2]],
                   time_step=data.get_timesteps_average())
    env.reset()

    # Init Particle Filter
    robot_tracker = ParticleFilter(number_of_particles=n_particles, 
                                   field=env.field,
                                   motion_noise=[0.1, 0.1, 0.01],
                                   measurement_weights=[5, 0, 0],
                                   vertical_lines_nr=vertical_lines_nr,
                                   resampling_algorithm=ResamplingAlgorithms.SYSTEMATIC,
                                   initial_odometry=odometry[0])
    robot_tracker.initialize_particles_from_seed_position(initial_position[0], initial_position[1], seed_radius)

    # Init Embedded Vision
    jetson_vision = JetsonVision(vertical_lines_nr=vertical_lines_nr, 
                                        enable_field_detection=True,
                                        enable_randomized_observations=True)
    jetson_vision.jetson_cam.setPoseFrom3DModel(170, 106.8)
    #self.embedded_vision.jetson_cam.setPoseFrom3DModel(170, 107.2)

    # Init Odometry
    odometry_particle = Particle(initial_state=initial_position,
                                movement_deviation=[0, 0, 0])
    for frame_nr in data.frames:      
        # update odometry:
        robot_tracker.odometry.update(odometry[env.steps])
        movement = robot_tracker.odometry.rad2deg(robot_tracker.odometry.movement)

        # capture frame:
        img, has_goal, goal_bbox = get_image_from_frame_nr(path, frame_nr), has_goals[env.steps], goals[env.steps]

        # make observations:    
        _, _, _, _, particle_filter_observations = jetson_vision.process_from_log(src = img,
                                                                                  timestamp = data.timestamps[env.steps],
                                                                                  has_goal = has_goal,
                                                                                  goal_bounding_box = goal_bbox)

        # compute particle filter tracking:    
        robot_tracker.update(movement, particle_filter_observations)
        odometry_particle.move(movement)

        # update visualization:    
        env.update(position[env.steps], 
                   robot_tracker.particles, 
                   robot_tracker.get_average_state(),
                   odometry_particle.state, 
                   time_steps[env.steps])
        env.render()