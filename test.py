import numpy as np

n_particles = 3
state_dimension = 3

def reset_particles():
    states = np.zeros((n_particles, state_dimension), dtype=np.float16)
    weights = np.ones(n_particles, dtype=np.float16) / n_particles
    particles = np.column_stack((weights, states))
    return particles

def set_particle(weight, x, y, theta):
    return np.array([weight, x, y, theta], dtype=np.float16)

samples = reset_particles()

samples[1] = set_particle(0.5, 1, 2, 3)

weighted_average = np.average(samples[:, 1:], axis=0, weights=samples[:, 0])
print(max(samples[:, 0]))

samples[:, 0] = np.ones(n_particles, dtype=np.float16) / n_particles

print(max(samples[:, 0]))

movement = np.array([-1, -2, -3])
print(np.abs(movement))
