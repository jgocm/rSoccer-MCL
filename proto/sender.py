import socket
from particles_pb2 import Particle, ParticleList

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Define the destination address and port
address = ('192.168.0.123', 1000)

# Create a ParticleList message
particle_list = ParticleList()

# Add particles to the ParticleList
particles = [
    [1.0, 2.0, 3.0, 0.5],
    [2.0, 3.0, 4.0, 1.0],
    [3.0, 4.0, 5.0, 1.5]
]

for particle in particles:
    p = particle_list.particles.add()
    p.w, p.x, p.y, p.theta = particle

# Serialize the ParticleList message
serialized_message = particle_list.SerializeToString()

# Send the serialized message over UDP
sock.sendto(serialized_message, address)

# Close the socket
sock.close()
