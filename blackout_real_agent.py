import numpy as np
import cv2
from rsoccer_gym.Tracking.ParticleFilterBase import ParticleFilter, Particle
from rsoccer_gym.Tracking import ResamplingAlgorithms
from rsoccer_gym.Perception.jetson_vision import JetsonVision
from rsoccer_gym.Perception.entities import *
from rsoccer_gym.Tracking.particle_filter_helpers import *
from rsoccer_gym.Utils.mcl_communication import *

def serialize_processing_times_to_log(frame_nr, n_particles, total, vision, localization):
    data = [frame_nr, n_particles, total, \
            vision[0], vision[1], vision[2], vision[3], \
            localization[0], localization[1], localization[2], localization[3], localization[4], localization[5]]
    return data

def get_image_from_frame_nr(path_to_images_folder, frame_nr):
    dir = path_to_images_folder+f'/cam/{frame_nr}.png'
    img = cv2.imread(dir)
    return img

if __name__ == "__main__":
    import os
    import time
    import csv
    from rsoccer_gym.Utils.load_localization_data import Read
    cwd = os.getcwd()

    debug = True
    n_particles = 200
    vertical_lines_nr = 1

    # CHOOSE SCENARIO
    scenarios = ['rnd_01', 'sqr_02', 'igs_03', 'tst_01']
    scenario = scenarios[2]

    # LOAD DATA
    path = f'/home/vision-blackout/ssl-navigation-dataset-jetson/data/{scenario}'
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

    # SET FIELD WITH RC LIMITS
    rc_field = Field()
    rc_field.redefineFieldLimits(x_max=4.2, y_max=3, x_min=-0.3, y_min=-3)

    # Init Particle Filter
    robot_tracker = ParticleFilter(number_of_particles=n_particles, 
                                   field=rc_field,
                                   motion_noise=[0.5, 0.5, 0.01],
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
                                 score_threshold=0.3,
                                 draw=debug,
                                 debug=debug)
    jetson_vision.jetson_cam.setPoseFrom3DModel(170, 106.8)
    #self.embedded_vision.jetson_cam.setPoseFrom3DModel(170, 107.2)

    # Init Odometry
    odometry_particle = Particle(initial_state=[0, 0, 0],
                                 movement_deviation=[0, 0, 0])
    
    # Send Particles
    UDP = ParticlesSender(receiver_address='192.168.1.52')
    
    # Evaluation metrics    
    avg_fps = 0
    steps = 0
    log = []

    for frame_nr in data.frames:    
        # start_time
        start_time = time.time()

        # update odometry:
        robot_tracker.odometry.update(odometry[steps])
        movement = limit_angle_from_pose(robot_tracker.odometry.rad2deg(robot_tracker.odometry.movement))

        # capture frame:
        img, has_goal, goal_bbox = get_image_from_frame_nr(path, frame_nr), has_goals[steps], goals[steps]

        # make observations:    
        _, _, _, _, particle_filter_observations, vision_dt = jetson_vision.process(src = img,
                                                                                    timestamp = data.timestamps[steps])

        # compute particle filter tracking:    
        failure_flag, relocalization_flag, avg_particle_state, localization_dt = robot_tracker.update(movement, 
                                                                                                      particle_filter_observations, 
                                                                                                      steps)
        if relocalization_flag: 
            odometry_particle = Particle(initial_state = avg_particle_state,
                                         movement_deviation = [0, 0, 0])
        odometry_particle.move(movement)

        # update step:    
        final_time = time.time()
        dt = final_time-start_time
        avg_fps = 0.5*avg_fps + 0.5*(1/dt)
        print(f'Nr Particles: {robot_tracker.n_particles} | Current processing time: {dt:.3f} | Avg FPS: {avg_fps:.3f} | Frame nr: {frame_nr}')

        # save processing times
        log.append(serialize_processing_times_to_log(frame_nr, 
                                                     robot_tracker.n_particles, 
                                                     dt,
                                                     vision_dt,
                                                     localization_dt))

        # debug
        if debug:
            cv2.imshow('debug', img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p') or failure_flag:
                import pdb;pdb.set_trace()

        UDP.setMCLMessage(position[steps].astype(float),
                          robot_tracker.particles.astype(float),
                          avg_particle_state.astype(float),
                          odometry_particle.state,
                          steps)
        UDP.sendMCLMessage()
        
        steps += 1

    cv2.destroyAllWindows()

    print("FINISHED")
