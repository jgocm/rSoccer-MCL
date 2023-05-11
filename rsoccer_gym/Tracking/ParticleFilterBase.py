import numpy as np
import math
from rsoccer_gym.Perception.ParticleVision import ParticleVision
from rsoccer_gym.Tracking.Resampler import Resampler
from rsoccer_gym.Perception.entities import Field
from rsoccer_gym.Tracking.Odometry import Odometry
from rsoccer_gym.Utils.Utils import *

class Particle:
    '''
    Particle pose has 3 degrees of freedom:
        x: particle position towards the global X axis (m)
        y: particle position towards the global Y axis (m)
        theta: particle orientation towards the field axis (degrees)     

    State: 
        (x, y, theta)
    
    Constraints:
        is_out_of_field: returns if the particle is out-of-field boundaries
    '''
    def __init__(
                self,
                initial_state = [0, 0, 0],
                weight = 1,
                movement_deviation = [0.1, 0.1, 0.01]
                ):
        self.state = initial_state
        self.x = self.state[0]
        self.y = self.state[1]
        self.theta = ((self.state[2] + 180) % 360) - 180
        self.weight = weight

        # standard deviation for movements
        self.movement_deviation = movement_deviation

    def from_weighted_sample(self, sample):
        self.__init__(weight=sample[0], initial_state=sample[1])

    def as_weighted_sample(self):
        return [self.weight,[self.x, self.y, self.theta]]

    def is_out_of_field(self, x_min, x_max, y_min, y_max):
        '''
        Check if particle is out of field boundaries
        
        param: current field configurations
        return: True if particle is out of field boundaries
        '''
        if self.x < x_min:
            return True
        elif self.x > x_max:
            return True
        elif self.y < y_min:
            return True
        elif self.y > y_max:
            return True
        else:
            return False

    def rotate_to_global(self, local_x, local_y, robot_w):
        theta = np.deg2rad(self.theta)
        global_x = local_x*np.cos(theta) - local_y*np.sin(theta)
        global_y = local_x*np.sin(theta) + local_y*np.cos(theta)
        return global_x, global_y, robot_w

    def add_move_noise(self, movement):
        movement_abs = np.array([np.abs(movement[0]), np.abs(movement[1]), np.abs(movement[2])])
        standard_deviation_vector = self.movement_deviation*movement_abs

        return np.random.normal(movement, standard_deviation_vector, 3).tolist()

    def limit_theta_degrees(self, theta):
        while theta > 180:
            theta -= 2*180
        while theta < -180:
            theta += 2*180
        return theta

    def move(self, movement):
        movement = self.add_move_noise(movement)
        movement = self.rotate_to_global(movement[0], movement[1], movement[2])
        self.x = self.state[0] + movement[0]
        self.y = self.state[1] + movement[1]
        self.theta = self.state[2] + movement[2] # TODO: add limit theta
        self.state = [self.x, self.y, self.theta]

class ParticleFilter:
    def __init__(self,
                 number_of_particles,
                 field,
                 motion_noise,
                 measurement_weights,
                 vertical_lines_nr,
                 resampling_algorithm,
                 initial_odometry,
                 data_type):

        if number_of_particles < 1:
            print(f"Warning: initializing particle filter with number of particles < 1: {number_of_particles}")
        
        # State related settings
        self.state_dimension = len(motion_noise)
        self.set_field_limits(field)
        self.data_type = data_type

        # Initialize filter settings
        self.n_particles = number_of_particles
        self.n_active_particles = number_of_particles
        self.particles = self.reset_particles()

        # Metrics for evaluating the particles' quality
        self.prior_weights_sum = 0
        self.average_particle_weight = 0

        # Particle sensors
        self.vision = ParticleVision(vertical_lines_nr=vertical_lines_nr)

        # Set noise
        self.motion_noise = motion_noise
        self.measurement_weights = measurement_weights

        # Resampling
        self.resampling_algorithm = resampling_algorithm
        self.resampler = Resampler()
        self.displacement = np.zeros(self.state_dimension, dtype=self.data_type)

        # Trackings
        self.odometry = Odometry(initial_position=initial_odometry)
        self.particles_average = self.set_particle()

        # Detect algorithm failures -> reset
        self.failure = False

        # Sets minimal value
        self.SMALL_VALUE = 1e-6

    def reset_particles(self):
        states = np.zeros((self.n_particles, self.state_dimension), dtype=self.data_type)
        weights = np.ones(self.n_particles, dtype=self.data_type) / self.n_particles
        particles = np.column_stack((weights, states))
        return particles

    def set_particle(self, weight=0, x=0, y=0, theta=0):
        return np.array([weight, x, y, theta], dtype=self.data_type)

    def move_particle(self, particle, movement, deviation):
        particle_state = particle[1:]
        noisy_movement = add_move_noise(movement, deviation)
        global_movement = rotate_to_global(noisy_movement[0], noisy_movement[1], particle_state[2])
        new_x = particle_state[0] + global_movement[0]
        new_y = particle_state[1] + global_movement[1]
        new_theta = limit_angle_degrees(particle_state[2] + movement[2])
        return self.set_particle(particle[0], new_x, new_y, new_theta)

    def initialize_particles_from_seed_position(self, position_x, position_y, max_distance):
        """
        Initialize the particles uniformly around a seed position (x, y, orientation). 
        """

        radius = np.random.uniform(0, max_distance)
        direction = np.random.uniform(0, 360)
        seed_x = position_x + radius*math.cos(direction)
        seed_y = position_y + radius*math.sin(direction)

        weight = 1.0/self.n_particles
        for i in range(self.n_particles):
            radius = np.random.uniform(0, max_distance)
            direction = np.random.uniform(0, 360)
            orientation = np.random.uniform(0, 360)
            x = seed_x + radius*math.cos(direction)
            y = seed_y + radius*math.sin(direction)
            self.particles[i] = self.set_particle(weight, x, y, orientation)
                
    def initialize_particles_uniform(self):
        """
        Initialize the particles uniformly over the world assuming a 3D state (x, y, orientation). 
        No arguments are required and function always succeeds hence no return value.
        """

        # Initialize particles with uniform weight distribution
        weight = 1.0 / self.n_particles
        for i in range(self.n_particles):
            x = np.random.uniform(self.x_min, self.x_max)
            y = np.random.uniform(self.y_min, self.y_max)
            orientation = np.random.uniform(-180, 180)
            self.particles[i] = self.set_particle(weight, x, y, orientation)
        
    def set_field_limits(self, field = Field()):
        self.field = field
        self.x_min = field.x_min
        self.x_max = field.x_max
        self.y_min = field.y_min
        self.y_max = field.y_max

    def get_average_state(self):
        """
        Compute average state according to all weighted particles

        :return: Average x-position, y-position and orientation
        """

        # Compute weighted average
        weighted_average = np.average(self.particles[:, 1:], axis=0, weights=self.particles[:, 0])
        weighted_average[2] = limit_angle_degrees(weighted_average[2])

        return weighted_average

    def get_max_weight(self):
        """
        Find maximum weight in particle filter.

        :return: Maximum particle weight
        """
        return max(self.particles[:, 0])

    def normalize_weights(self, prior_weights = []):
        """
        Normalize all particle weights.

        Receives weights as a list.
        """

        # Check if weights are non-zero
        # TODO: reinitialize particles => initialize from seed and add odometry
        if self.prior_weights_sum < self.SMALL_VALUE:
            print(f"Weight normalization failed: sum of all weights is {self.prior_weights_sum} (weights will be reinitialized)")
            self.failure = True
            # Set uniform weights
            return np.ones(self.n_particles, dtype=self.data_type) / self.n_particles

        # Return normalized weights
        return np.array(prior_weights, dtype=self.data_type) / self.prior_weights_sum

    def propagate_particles(self, movement):
        """
        Propagate particles from odometry movement measurements. 
        Return the propagated particle.

        :param movement: [forward motion, side motion and rotation] in meters and degrees
        """
        self.displacement = self.displacement + movement
        
        # Move particles
        for particle in self.particles:
            particle.move(movement)

            # Remove Particles Out of Field Boundaries
            if particle.is_out_of_field(x_min=self.x_min, x_max=self.x_max, y_min=self.y_min, y_max=self.y_max):
                particle.weight = 0

    def compute_observation(self, particle):
        goal = self.vision.track_positive_goal_center(                                    
                                    particle[1], 
                                    particle[2], 
                                    particle[3], 
                                    self.field)
        boundary_points = self.vision.detect_boundary_points(
                                    particle[1], 
                                    particle[2], 
                                    particle[3], 
                                    self.field)
        
        return goal, boundary_points

    def compute_boundary_points_similarity(self, alpha=10, robot_observations=[], particle_observations=[]):
        # returns 1 if there are no robot observations
        if len(robot_observations)<1:
            return 1
        
        # initial value
        likelihood_sample = 1

        # Compute difference between real measurements and sample observations
        differences = np.array(robot_observations) - particle_observations
        # Loop over all observations for current particle
        for diff in differences:
            # Map difference true and expected angle measurement to probability
            p_z_given_distance = \
                np.exp(-alpha * (diff[0]) * (diff[0]) /
                    (robot_observations[0][0] * robot_observations[0][0]))

            # Incorporate likelihoods current landmark
            likelihood_sample *= p_z_given_distance
            #if likelihood_sample<self.SMALL_VALUE:
            #    return 0

        return likelihood_sample

    def compute_goal_similarity(self, robot_observation=[], particle_observation=[]):
        # Returns 1 if robot does not see the goal
        if not robot_observation: return 1

        # Returns 0 if particle's angle to goal is too high
        else: return particle_observation[0]

    def compute_likelihood(self, observations, particle):
        """
        Compute likelihood p(z|sample) for a specific measurement given sample observations.

        :param robot_field_points: Current robot_field_points
        :param observations: Detected wall relative positions from the sample vision
        :return Likelihood
        """
        # Check if particle is out of field boundaries
        if is_out_of_field(particle=particle, x_min=self.x_min, x_max=self.x_max, y_min=self.y_min, y_max=self.y_max):
            return 0

        else:
            # Parse particle filter observations
            robot_goal, robot_boundary_points, robot_line_points = observations

            # Initialize measurement likelihood
            likelihood_sample = 1.0
            
            # Compute particle observations
            particle_goal, particle_boundary_points = self.compute_observation(particle)
            
            # Compute similarity from field boundary points
            likelihood_sample *= self.compute_boundary_points_similarity(self.measurement_weights[0], robot_boundary_points, particle_boundary_points)

            # Compute similarity from goal center
            likelihood_sample *= self.compute_goal_similarity(robot_goal, particle_goal)

            # Return importance weight based on all landmarks
            return likelihood_sample

    def compute_covariance(self, avg_particle):
        # TODO: fix for numpy array particles
        Pxx = 0
        ux = np.array(avg_particle)
        for particle in self.particles:
            diff = particle.state - ux
            diff[0] = diff[0]/self.x_max
            diff[1] = diff[1]/self.x_max
            diff[2] = self.compute_normalized_angle_diff(diff[2])
            Pxx += particle.weight*diff@diff

        return Pxx      

    def needs_resampling(self, prior_weights = []):
        '''
        Checks if resampling is needed
        '''
        # Check if weights are non-zero
        if self.prior_weights_sum < self.SMALL_VALUE:
            # TODO: reinitialize particles => initialize from seed and add total odometry
            self.particles[:, 0] = self.normalize_weights(prior_weights)
            return True

        # Check if distance traveled from last resampling is sufficient for lowering odometry confidence
        distance = math.sqrt(self.displacement[0]**2 + self.displacement[1]**2)
        rotation = self.displacement[2]
        if distance>0.5 or rotation>45:
            return True

        # Check if the distribution has converged to one particle
        for weight in prior_weights:
            if weight/self.prior_weights_sum>0.7:
                return True

        return False

    def update(self, movement, observations):
        """
        Process a measurement given the measured robot displacement and resample if needed.

        :param robot_forward_motion: Measured forward robot motion in meters.
        :param robot_angular_motion: Measured angular robot motion in radians.
        :param field_points: field_points.
        :param landmarks: Landmark positions.
        """

        if len(observations[1])>0:
            self.vision.set_detection_angles_from_list([observations[1][0][1]])

        prior_weights = []
        self.prior_weights_sum = 0
        self.displacement = self.displacement + movement
        for i in range(self.n_particles):
            # Propagate the particle's state according to the current movements
            self.particles[i] = self.move_particle(self.particles[i], movement, self.motion_noise)

            # Compute current particle's weight based on likelihood
            prior_probability = self.compute_likelihood(observations, self.particles[i])

            # Update to unnormalized weight
            self.particles[i][0] *= prior_probability

            # Store weight for normalization
            prior_weights.append(self.particles[i][0])

            # Update unnormalized weights' sum
            self.prior_weights_sum += self.particles[i][0]      

        # Resample if needed
        if self.needs_resampling(prior_weights):
            self.displacement = np.zeros(self.state_dimension, dtype=self.data_type)
            new_samples = self.resampler.resample(self.particles,
                                                  self.n_particles, 
                                                  self.resampling_algorithm)
            self.particles = new_samples
        
        # Update to normalized weights
        self.particles[:, 0]  = self.normalize_weights(prior_weights)


if __name__=="__main__":
    from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv

    env = SSLBaseEnv(
        field_type=1,
        n_robots_blue=0,
        n_robots_yellow=0,
        time_step=0.025)
        
    env.field.boundary_width = 0.3

    particle_filter = ParticleFilter(
        number_of_particles = 3,
        field = env.field,
        motion_noise = [1, 1, 1],
        measurement_noise = [1, 1]
    )