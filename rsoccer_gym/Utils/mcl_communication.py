import socket
import numpy as np

class ParticlesReceiver():
    def __init__(self,
                 sender_address = '192.168.0.193',
                 port = 9600): 
        self.address = sender_address
        self.port = port
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.bind(('', self.port))
        self.udp_sock.settimeout(0)
        self.msg = pb.ParticleList()

    def recvMCLMessage(self):
        msg = pb.ParticleList()

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
    
class ParticlesSender():
    def __init__(self,
                 receiver_address='192.168.0.123',
                 port=9600):
        self.address = receiver_address
        self.port = port
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.msg = pb.ParticleList()

    def setMCLMessage(self, particles):
        self.msg = pb.ParticleList()
        for particle in particles:
            p = self.msg.particles.add()
            p.w, p.x, p.y, p.theta = particle

    def sendMCLMessage(self):
        self.udp_sock.sendto(self.msg.SerializeToString(), (self.address, self.port))
    

def run_receiver():
    UDP = ParticlesReceiver()

    while True:
        particles = UDP.recvMCLMessage()
        # Print the list of particles
        for particle in particles:
            print(particle)  

def run_sender():
    UDP = ParticlesSender()

    particles = np.array([
        [1.0, 2.0, 3.0, 0.5],
        [2.0, 3.0, 4.0, 1.0],
        [3.0, 4.0, 5.0, 1.5]
    ])
    UDP.setMCLMessage(particles)
    while True:
        UDP.sendMCLMessage()

if __name__ == "__main__":
    import particles_pb2 as pb


    is_receiver = False

    if is_receiver: run_receiver()
    else: run_sender()

else:
    from rsoccer_gym.Utils import particles_pb2 as pb
