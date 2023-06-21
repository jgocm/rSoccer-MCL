import numpy as np
import cv2
from rsoccer_gym.Tracking.ParticleFilterBase import ParticleFilter, Particle
from rsoccer_gym.Tracking import ResamplingAlgorithms
from rsoccer_gym.Perception.jetson_vision import JetsonVision
from rsoccer_gym.Perception.entities import *
from rsoccer_gym.Tracking.particle_filter_helpers import *
from rsoccer_gym.Utils.mcl_communication import *


def get_image_from_frame_nr(path_to_images_folder, frame_nr):
    dir = path_to_images_folder+f'/cam/{frame_nr}.png'
    img = cv2.imread(dir)
    return img

if __name__ == "__main__":
    import os
    import time
    from rsoccer_gym.Utils.load_localization_data import Read
    cwd = os.getcwd()

    debug = True
    n_particles = 100
    vertical_lines_nr = 1

    # CHOOSE SCENARIO
    scenarios = ['rnd_01', 'sqr_02', 'igs_03']

    # LOAD DATA
    path = f'/home/vision-blackout/ssl-navigation-dataset-jetson/data/{scenarios[0]}'
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

    # Init Particle Filter
    robot_tracker = ParticleFilter(number_of_particles=n_particles, 
                                   field=Field(),
                                   motion_noise=[0.2, 0.2, 0.05],
                                   measurement_weights=[1],
                                   vertical_lines_nr=vertical_lines_nr,
                                   resampling_algorithm=ResamplingAlgorithms.SYSTEMATIC,
                                   initial_odometry=odometry[0],
                                   data_type=np.float16)
    robot_tracker.initialize_particles_from_seed_position(initial_position[0], initial_position[1], seed_radius)
    #robot_tracker.initialize_particles_uniform()
    
    # Init Embedded Vision
    jetson_vision = JetsonVision(vertical_lines_nr=vertical_lines_nr, 
                                 enable_field_detection=True,
                                 enable_randomized_observations=True,
                                 score_threshold=0.2,
                                 draw=debug,
                                 debug=debug)
    jetson_vision.jetson_cam.setPoseFrom3DModel(170, 106.8)
    #self.embedded_vision.jetson_cam.setPoseFrom3DModel(170, 107.2)

    # Init Odometry
    odometry_particle = Particle(initial_state=initial_position,
                                 movement_deviation=[0, 0, 0])
    
    # Send Particles
    UDP = ParticlesSender()
    
    avg_fps = 0
    steps = 0

    for frame_nr in data.frames:    
        # start_time
        start_time = time.time()

        # update odometry:
        robot_tracker.odometry.update(odometry[steps])
        movement = limit_angle_from_pose(robot_tracker.odometry.rad2deg(robot_tracker.odometry.movement))

        # capture frame:
        img, has_goal, goal_bbox = get_image_from_frame_nr(path, frame_nr), has_goals[steps], goals[steps]

        # make observations:
        _, _, _, _, particle_filter_observations = jetson_vision.process(src = img,
                                                                         timestamp = data.timestamps[steps])

        # compute particle filter tracking:    
        robot_tracker.update(movement, particle_filter_observations, steps)
        odometry_particle.move(movement)

        # update step:    
        final_time = time.time()
        dt = final_time-start_time
        avg_fps = 0.5*avg_fps + 0.5*(1/dt)
        avg_particle = robot_tracker.get_average_state()
        #print(f'Nr Particles: {robot_tracker.n_particles} | Current processing time: {dt} | Avg FPS: {avg_fps}')
        print(f'avg particle: {robot_tracker.get_average_state()} | ground-truth: {position[steps]}')

        # debug
        if debug:
            cv2.imshow('debug', img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            UDP.setMCLMessage(robot_tracker.particles.astype(float))
            UDP.sendMCLMessage()
        
        steps += 1

    cv2.destroyAllWindows()