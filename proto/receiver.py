import socket
from particles_pb2 import *

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Define the receiving address and port
address = ('', 9600)  # Empty string for the local IP address

# Bind the socket to the address and port
sock.bind(address)

while True:
    # Receive the UDP packet
    data, addr = sock.recvfrom(1024)  # Adjust the buffer size if needed
    
    # Deserialize the received message
    particle_list = ParticleList()
    particle_list.ParseFromString(data)
    
    # Access the list of particles
    particles = []
    for particle in particle_list.particles:
        p = [particle.w, particle.x, particle.y, particle.theta]
        particles.append(p)
    
    # Print the list of particles
    for particle in particles:
        print(particle)
    
# Close the socket
sock.close()