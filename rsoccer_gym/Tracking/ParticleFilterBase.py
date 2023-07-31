import numpy as np
import math
from rsoccer_gym.Perception.ParticleVision import ParticleVision
from rsoccer_gym.Tracking.Resampler import Resampler
from rsoccer_gym.Perception.entities import Field
from rsoccer_gym.Tracking.Odometry import Odometry
from .particle_filter_helpers import *

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
                movement_deviation = [1, 1, 0.1]
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

    def from_numpy_array(self, sample):
        sample = sample.tolist()
        self.__init__(weight=sample[0], initial_state=sample[1:])

    def as_numpy_array(self):
        return np.array([self.weight, self.x, self.y, self.theta])

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
        noisy_movement = self.add_move_noise(movement)
        global_movement = self.rotate_to_global(noisy_movement[0], noisy_movement[1], noisy_movement[2])
        self.x = self.state[0] + global_movement[0]
        self.y = self.state[1] + global_movement[1]
        self.theta = self.state[2] + global_movement[2]
        self.state = [self.x, self.y, self.theta]

class ParticleFilter:
    def __init__(self,
                 number_of_particles,
                 is_adaptive,
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

        # Sets precision
        self.SMALL_VALUE = 1e-7
        self.data_type = data_type

        # Initialize filter settings
        self.is_adaptive = is_adaptive
        self.n_max_particles = number_of_particles
        self.n_min_particles = 20
        self.n_particles = number_of_particles
        self.n_active_particles = number_of_particles
        self.particles = self.reset_particles()
        self.numpy_particles = self.reset_particles()

        # Metrics for evaluating the particles' quality
        self.prior_weights_sum = 0
        self.average_particle_weight = 1/number_of_particles

        # Particle sensors
        self.vision = ParticleVision(vertical_lines_nr=vertical_lines_nr)

        # Set noise
        self.motion_noise = motion_noise
        self.measurement_weights = measurement_weights

        # Resampling
        self.resampling_algorithm = resampling_algorithm
        self.resampler = Resampler()
        self.displacement = np.zeros(self.state_dimension)

        # Trackings
        self.odometry = Odometry(initial_position=initial_odometry)
        self.self_localization = self.set_particle()
        self.total_movement = np.zeros(self.state_dimension)

        # Detect algorithm failures -> reset
        self.failure = False

    def reset_particles(self):
        states = np.zeros((self.n_particles, self.state_dimension), dtype=self.data_type)
        weights = np.ones(self.n_particles, dtype=self.data_type) / self.n_particles
        particles = np.column_stack((weights, states))
        return particles
    
    def set_particle(self, weight=0, x=0, y=0, theta=0):
        return np.array([weight, x, y, theta], dtype=self.data_type)

    def propagate_particles_as_matrix(self, movement, motion_noise):
        # adds gaussian noise to movement vector
        movement_abs = np.abs(movement)
        standard_deviation_vector = motion_noise*movement_abs
        noisy_movement = np.random.normal(movement, standard_deviation_vector, (self.n_particles, self.state_dimension))

        # rotates movement vector to global axis
        thetas = np.deg2rad(self.particles[:, 3])
        try:
            global_xs = noisy_movement[:, 0]*np.cos(thetas) - noisy_movement[:, 1]*np.sin(thetas)
        except: 
            import pdb;pdb.set_trace()
        global_ys = noisy_movement[:, 0]*np.sin(thetas) + noisy_movement[:, 1]*np.cos(thetas)
        global_movement = np.column_stack([global_xs, global_ys, noisy_movement[:, 2]])

        # sums movements
        self.particles[:, 1:] += global_movement

    def propagate_particle(self, particle_state, movement, motion_noise):
        noisy_movement = add_move_noise(movement, motion_noise)
        global_movement = rotate_to_global(particle_state[2], noisy_movement[0], noisy_movement[1], noisy_movement[2])
        particle_state += global_movement
        return particle_state

    def initialize_particles_from_seed_position(self, position_x, position_y, max_distance):
        """
        Initialize the particles uniformly around a seed position (x, y, orientation). 
        """
        radius = np.random.uniform(0, max_distance)
        direction = np.random.uniform(-180, 180)
        seed_x = position_x + radius*math.cos(np.deg2rad(direction))
        seed_y = position_y + radius*math.sin(np.deg2rad(direction))

        particles = []
        weight = 1.0/self.n_particles
        for i in range(self.n_particles):
            radius = np.random.uniform(0, max_distance)
            direction = np.random.uniform(-180, 180)
            orientation = np.random.uniform(-180, 180)
            x = seed_x + radius*math.cos(np.deg2rad(direction))
            y = seed_y + radius*math.sin(np.deg2rad(direction))
            particle = self.set_particle(weight, x, y, orientation)
            particles.append(particle)
        
        self.particles = np.array(particles)

        robot_area = np.pi*max_distance*max_distance
        total_area = (self.x_max-self.x_min)*(self.y_max-self.y_min)
        self.average_particle_weight = 1 - robot_area/total_area
        
    def initialize_particles_uniform(self):
        """
        Initialize the particles uniformly over the world assuming a 3D state (x, y, orientation). 
        No arguments are required and function always succeeds hence no return value.
        """

        # Initialize particles with uniform weight distribution
        particles = []
        weight = 1.0 / self.n_particles
        for i in range(self.n_particles):
            x = np.random.uniform(self.x_min, self.x_max)
            y = np.random.uniform(self.y_min, self.y_max)
            orientation = np.random.uniform(-180, 180)
            particle = self.set_particle(weight, x, y, orientation)
            particles.append(particle)
        
        self.particles = np.array(particles)

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

        # Compute sum of all weights
        sum_weights = np.sum(self.particles[:, 0])

        # Compute weighted average
        self_localization = np.average(self.particles[:, 1:], 
                                       axis=0, 
                                       weights=self.particles[:, 0]/sum_weights)
        
        return limit_angle_from_pose(self_localization)

    def get_max_weight(self):
        """
        Find maximum weight in particle filter.

        :return: Maximum particle weight
        """
        return max(self.particles[:, 0])

    def normalize_weights(self):
        """
        Normalize all particle weights.

        Receives weights as a list
        """
        weights = self.particles[:, 0]

        # Total number of weights
        n_weights = len(weights)

        # Update prior weights sum
        self.prior_weights_sum = np.sum(weights)

        # Check if weights are non-zero
        if self.prior_weights_sum < self.SMALL_VALUE:
            print(f"Weight normalization failed: sum of all weights is {self.SMALL_VALUE} (weights will be reinitialized)")
            self.failure = True

            # Set uniform weights
            return np.ones(n_weights, dtype=self.data_type) / n_weights

        # Return normalized weights
        return weights / self.prior_weights_sum

    def compute_observation(self, particle_state):
        goal = self.vision.track_positive_goal_center(                                    
                                    particle_state[0], 
                                    particle_state[1], 
                                    particle_state[2], 
                                    self.field)
        boundary_points = self.vision.detect_boundary_points(
                                    particle_state[0], 
                                    particle_state[1], 
                                    particle_state[2], 
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

        return likelihood_sample

    def compute_normalized_angle_diff(self, diff):
        while diff>180:
            diff -= 2*180
        while diff<-180:
            diff += 2*180
        d = np.abs(diff)/180
        return d

    def compute_goal_similarity(self, robot_observation=[], particle_observation=[]):
        # Returns 1 if robot does not see the goal
        if not robot_observation: return 1

        # Returns 0 if particle's angle to goal is too high
        else: return particle_observation[0]

    def compute_likelihood(self, observations, particle_state):
        """
        Compute likelihood p(z|sample) for a specific measurement given sample observations.

        :param robot_field_points: Current robot_field_points
        :param observations: Detected wall relative positions from the sample vision
        :return Likelihood
        """
        # Check if particle is out of field boundaries
        if is_out_of_field(particle_state=particle_state, x_min=self.x_min, x_max=self.x_max, y_min=self.y_min, y_max=self.y_max):
            return 0

        else:
            # Parse particle filter observations
            robot_goal, robot_boundary_points, _ = observations

            # Initialize measurement likelihood
            likelihood_sample = 1.0
            
            # Compute particle observations
            particle_goal, particle_boundary_points = self.compute_observation(particle_state)
            
            # Compute similarity from field boundary points
            likelihood_sample *= self.compute_boundary_points_similarity(self.measurement_weights[0], robot_boundary_points, particle_boundary_points)

            # Compute similarity from goal center
            likelihood_sample *= self.compute_goal_similarity(robot_goal, particle_goal)

            # Return importance weight based on all landmarks
            return likelihood_sample

    def compute_covariance(self, avg_particle):
        Pxx = 0
        ux = avg_particle
        for particle in self.particles:
            diff = particle[1:] - ux
            diff[0] = diff[0]/self.x_max
            diff[1] = diff[1]/self.x_max
            diff[2] = self.compute_normalized_angle_diff(diff[2])
            Pxx += particle[0]*diff@diff

        return Pxx      

    def needs_resampling(self):
        '''
        Checks if resampling is needed
        '''
        
        if self.is_adaptive:
            self.n_active_particles = int(map(1-self.average_particle_weight,
                                            in_min=0.01,
                                            in_max=0.6,
                                            out_min=self.n_min_particles, 
                                            out_max=self.n_max_particles))
        if abs(self.n_active_particles-self.n_particles)>15:
            self.n_particles = self.n_active_particles
            return True
        
        if np.sum(self.particles[:, 0])<self.SMALL_VALUE:
            # Update to normalized weights        
            self.particles[:, 0] = np.ones(self.n_particles, dtype=self.data_type) / self.n_particles
            return True

        distance = math.sqrt(self.displacement[0]**2 + self.displacement[1]**2)
        dtheta = self.displacement[2]
        if distance>0.5 or dtheta>45:
            return True

        for particle in self.particles:
            if particle[0]>0.9:
                return True

        else: return False

    def update(self, movement, observations, step):
        """
        Updates particles states and weights, resamples if needed, 
            and evaluates the current distribuition.

        :param movement: Measured local robot motion in meters and degrees.
        :param observations: Observed goal and field boundary point.
        :param step: Current execution iteration.
        """
        
        if len(observations[1])>0:
            self.vision.set_detection_angles_from_list([observations[1][0][1]])

        # Propagate the states according to the current movement
        self.displacement += movement
        self.total_movement += movement
        self.propagate_particles_as_matrix(movement, self.motion_noise)

        for particle in self.particles:
            # Compute current particle's weight based on likelihood
            particle[0] *= self.compute_likelihood(observations, particle[1:])

        # Update to normalized weights        
        self.particles[:, 0] = self.normalize_weights()

        # Computes average for evaluating current state
        alpha = 0.95
        self.average_particle_weight = alpha*self.average_particle_weight + (1-alpha)*self.compute_likelihood(observations, self.get_average_state())

        # Resample if needed
        if self.needs_resampling():
            self.displacement = np.zeros(self.state_dimension)
            self.particles = self.resampler.resample(self.particles, 
                                                     self.n_particles, 
                                                     self.resampling_algorithm,
                                                     self.data_type,
                                                     self.average_particle_weight)

            # Update to normalized weights        
            self.particles[:, 0] = self.normalize_weights()

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