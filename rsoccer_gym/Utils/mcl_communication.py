import socket
import numpy as np
from particles_pb2 import *

class SocketUDP():
    def __init__(
                self,
                host_address='192.168.0.193',
                host_port=9601,
                device_address='192.168.0.123',
                device_port=9600                
                ):
        super(SocketUDP, self).__init__()
        self.host_address = host_address
        self.host_port = host_port
        self.device_address = device_address
        self.device_port = device_port
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.bind(('', self.host_port))
        self.udp_sock.settimeout(0)
        self.msg = ParticleList()

    def setHostUDP(self, address, port):
        self.host_address=address
        self.host_port=port
    
    def setDeviceUDP(self, address,port):
        self.device_address=address
        self.device_port=port

    def setMCLMessage(self, particles):
        self.msg = ParticleList()
        for particle in particles:
            p = self.msg.particles.add()
            p.w, p.x, p.y, p.theta = particle

    def sendMCLMessage(self):
        self.udp_sock.sendto(self.msg.SerializeToString(), (self.device_address, self.device_port))
    
    def recvMCLMessage(self):
        msg = ParticleList()

        # multiple messages are received so read until messages are no longer available
        has_msg = False
        while True:
            try:
                data, _ = self.udp_sock.recvfrom(1024)
                msg.ParseFromString(data)
                has_msg = True
            except:
                break 

        particles = []
        for particle in msg.particles:
            p = [particle.w, particle.x, particle.y, particle.theta]
            particles.append(p)
        
        return np.array(particles)

if __name__ == "__main__":

    host_address = "192.168.0.193"
    host_port = 9601
    device_address = "192.168.0.123"
    device_port = 9600

    UDP = SocketUDP(
        host_address=host_address,
        host_port=host_port,
        device_address=device_address,
        device_port=device_port
    )

    is_receiver = True

    # RECEIVER:
    while is_receiver:
        particles = UDP.recvMCLMessage()
        # Print the list of particles
        for particle in particles:
            print(particle)
    
    # SENDER:
    particles = np.array([
        [1.0, 2.0, 3.0, 0.5],
        [2.0, 3.0, 4.0, 1.0],
        [3.0, 4.0, 5.0, 1.5]
    ])
    UDP.setMCLMessage(particles)
    while not is_receiver:
        UDP.sendMCLMessage()



