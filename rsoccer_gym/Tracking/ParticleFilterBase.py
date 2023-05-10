import numpy as np
import math
from rsoccer_gym.Perception.ParticleVision import ParticleVision
from rsoccer_gym.Tracking.Resampler import Resampler
from rsoccer_gym.Perception.entities import Field
from rsoccer_gym.Tracking.Odometry import Odometry

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
        self.theta = self.state[2] + movement[2]
        self.state = [self.x, self.y, self.theta]

class ParticleFilter:
    def __init__(self,
                 number_of_particles,
                 field,
                 motion_noise,
                 measurement_weights,
                 vertical_lines_nr,
                 resampling_algorithm,
                 initial_odometry):

        if number_of_particles < 1:
            print(f"Warning: initializing particle filter with number of particles < 1: {number_of_particles}")
        
        # State related settings
        self.state_dimension = len(motion_noise)
        self.set_field_limits(field)

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
        self.displacement = np.zeros(self.state_dimension)

        # Trackings
        self.odometry = Odometry(initial_position=initial_odometry)
        self.self_localization = Particle()

        # Detect algorithm failures -> reset
        self.failure = False

        # Sets minimal value
        self.SMALL_VALUE = 1e-13

    def reset_particles(self):
        states = np.zeros((self.n_particles, self.state_dimension), dtype=np.float16)
        weights = np.ones(self.n_particles, dtype=np.float16) / self.n_particles
        particles = np.column_stack((weights, states))
        return particles

    def initialize_particles_from_seed_position(self, position_x, position_y, max_distance):
        """
        Initialize the particles uniformly around a seed position (x, y, orientation). 
        """

        radius = np.random.uniform(0, max_distance)
        direction = np.random.uniform(0, 360)
        seed_x = position_x + radius*math.cos(direction)
        seed_y = position_y + radius*math.sin(direction)

        particles = []
        weight = 1.0/self.n_particles
        for i in range(self.n_particles):
            radius = np.random.uniform(0, max_distance)
            direction = np.random.uniform(0, 360)
            orientation = np.random.uniform(0, 360)
            x = seed_x + radius*math.cos(direction)
            y = seed_y + radius*math.sin(direction)
            particle = Particle(initial_state=[x, y, orientation], 
                                weight=weight, 
                                movement_deviation=self.motion_noise)
            particles.append(particle)
        
        self.particles = particles
        
    def initialize_particles_uniform(self):
        """
        Initialize the particles uniformly over the world assuming a 3D state (x, y, orientation). 
        No arguments are required and function always succeeds hence no return value.
        """

        # Initialize particles with uniform weight distribution
        particles = []
        weight = 1.0 / self.n_particles
        for i in range(self.n_particles):
            particle = Particle(
                initial_state=[
                    np.random.uniform(self.x_min, self.x_max),
                    np.random.uniform(self.y_min, self.y_max),
                    np.random.uniform(-180, 180)],
                    weight=weight,
                    movement_deviation = self.motion_noise)

            particles.append(particle)
        
        self.particles = particles

    def set_field_limits(self, field = Field()):
        self.field = field
        self.x_min = field.x_min
        self.x_max = field.x_max
        self.y_min = field.y_min
        self.y_max = field.y_max

    def particles_as_weigthed_samples(self):
        samples = []
        for particle in self.particles:
            samples.append(particle.as_weighted_sample())
        return samples

    def get_average_state(self):
        """
        Compute average state according to all weighted particles

        :return: Average x-position, y-position and orientation
        """

        # Compute sum of all weights
        sum_weights = 0.0
        for particle in self.particles:
            sum_weights += particle.weight

        # Compute weighted average
        avg_x = 0.0
        avg_y = 0.0
        avg_theta = 0.0
        for particle in self.particles:
            avg_x += particle.x / sum_weights * particle.weight
            avg_y += particle.y / sum_weights * particle.weight
            avg_theta += particle.theta / sum_weights * particle.weight

        return [avg_x, avg_y, avg_theta]

    def get_max_weight(self):
        """
        Find maximum weight in particle filter.

        :return: Maximum particle weight
        """
        return max([particle.as_weigthed_sample()[0] for particle in self.particles])

    def normalize_weights(self, weights = []):
        """
        Normalize all particle weights.

        Receives weights as a list
        """

        # Check if weights are non-zero
        if self.prior_weights_sum < self.SMALL_VALUE:
            print(f"Weight normalization failed: sum of all weights is {self.SMALL_VALUE} (weights will be reinitialized)")
            self.failure = True

            # Set uniform weights
            return [(1.0 / len(weights)) for i in weights]

        # Return normalized weights
        return [weight / self.prior_weights_sum for weight in weights]

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
                                    particle.x, 
                                    particle.y, 
                                    particle.theta, 
                                    self.field)
        boundary_points = self.vision.detect_boundary_points(
                                    particle.x, 
                                    particle.y, 
                                    particle.theta, 
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
            if likelihood_sample<self.SMALL_VALUE:
                return 0

        return likelihood_sample

    def compute_normalized_angle_diff(self, diff):
        while diff>180:
            diff -= 2*180
        while diff<-180:
            diff += 2*180
        d = np.abs(diff)/180
        return d

    def compute_goal_similarity(self, alpha_distance=5, alpha_angle=10, robot_observation=[], particle_observation=[]):
        # Returns 1 if robot does not see the goal
        if not robot_observation: return 1

        # Returns 0 if particle's angle to goal is too high
        else: return particle_observation[0]

        # initial value
        likelihood_sample = 1

        # Compute difference between real measurements and sample observations
        differences = np.array(robot_observation) - particle_observation
        differences[2] = self.compute_normalized_angle_diff(differences[2])
        
        # Map difference true and expected angle measurement to probability
        p_z_given_distance = \
            np.exp(-alpha_distance * (differences[1]) * (differences[1]) /
                (robot_observation[1] * robot_observation[1]))
        p_z_given_angle = \
            np.exp(-alpha_angle * (differences[2]) * (differences[2]) /
                (robot_observation[1] * robot_observation[1]))
            
        # Incorporate likelihoods current landmark
        likelihood_sample *= p_z_given_distance*p_z_given_angle
        if likelihood_sample<1e-15:
            return 0

        return likelihood_sample

    def compute_likelihood(self, observations, particle):
        """
        Compute likelihood p(z|sample) for a specific measurement given sample observations.

        :param robot_field_points: Current robot_field_points
        :param observations: Detected wall relative positions from the sample vision
        :return Likelihood
        """
        # Check if particle is out of field boundaries
        if particle.is_out_of_field(x_min=self.x_min, x_max=self.x_max, y_min=self.y_min, y_max=self.y_max):
            return 0

        else:
            # Parse particle filter observations
            robot_goal, robot_boundary_points, robot_field_points = observations

            # Initialize measurement likelihood
            likelihood_sample = 1.0
            
            # Compute particle observations
            particle_goal, particle_boundary_points = self.compute_observation(particle)
            
            # Compute similarity from field boundary points
            likelihood_sample *= self.compute_boundary_points_similarity(self.measurement_weights[0], robot_boundary_points, particle_boundary_points)

            # Compute similarity from goal center
            likelihood_sample *= self.compute_goal_similarity(self.measurement_weights[1], self.measurement_weights[2], robot_goal, particle_goal)

            # Return importance weight based on all landmarks
            return likelihood_sample

    def compute_covariance(self, avg_particle):
        Pxx = 0
        ux = np.array(avg_particle)
        for particle in self.particles:
            diff = particle.state - ux
            diff[0] = diff[0]/self.x_max
            diff[1] = diff[1]/self.x_max
            diff[2] = self.compute_normalized_angle_diff(diff[2])
            Pxx += particle.weight*diff@diff

        return Pxx      

    def needs_resampling(self, observations):
        '''
        Checks if resampling is needed
        '''
        # computes average for evaluating current state
        avg_particle = Particle(self.get_average_state(), 1)
        weight = self.compute_likelihood(observations, avg_particle)
        self.average_particle_weight = weight
        pxx = self.compute_covariance(avg_particle.state)
        #print(f'pxx: {pxx}')
        #if pxx>0.005:
        #    return True
        
        if weight<0.5:
            return True

        distance = math.sqrt(self.displacement[0]**2 + self.displacement[1]**2)
        dtheta = self.displacement[2]
        if distance>1 or dtheta>90:
            return True

        for particle in self.particles:
            if particle.weight>0.9:
                return True

        else: return False

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

        weights = []
        self.prior_weights_sum = 0
        for particle in self.particles:
            # Propagate the particle's state according to the current movements
            particle.move(movement)

            # Compute current particle's weight based on likelihood
            weight = particle.weight * self.compute_likelihood(observations, particle)

            # Store weight for normalization
            weights.append(weight)

            # Update prior weights' sum
            self.prior_weights_sum += weight

        # Update to normalized weights
        weights = self.normalize_weights(weights)
        self.n_active_particles = self.n_particles
        for i in range(self.n_particles):
            if weights[i]<self.SMALL_VALUE:
                self.n_active_particles = self.n_active_particles-1
            self.particles[i].weight = weights[i]

        # Resample if needed
        if self.needs_resampling(observations):
            self.displacement = [0, 0, 0]
            samples = self.resampler.resample(
                            self.particles_as_weigthed_samples(), 
                            self.n_particles, 
                            self.resampling_algorithm)
            for i in range(self.n_particles):
                self.particles[i].from_weighted_sample(samples[i])
                weights[i] = self.particles[i].weight
            self.normalize_weights(weights)
            for i in range(self.n_particles):
                if weights[i]<self.SMALL_VALUE:
                    self.n_active_particles = self.n_active_particles-1
                self.particles[i].weight = weights[i]

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