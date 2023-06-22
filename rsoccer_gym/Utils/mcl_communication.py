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

        robot_position = np.array(msg.robot_position)
        mcl_position = np.array(msg.mcl_position)
        odometry_position = np.array(msg.odometry_position)
        time_steps = msg.time_steps

        particles = []
        for particle in msg.particles:
            p = [particle.w, particle.x, particle.y, particle.theta]
            particles.append(p)
        
        return has_msg, robot_position, np.array(particles), mcl_position, odometry_position, time_steps
    
class ParticlesSender():
    def __init__(self,
                 receiver_address='192.168.0.123',
                 port=9600):
        self.address = receiver_address
        self.port = port
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.msg = pb.ParticleList()

    def setRobotPosition(self, robot_position):
        self.msg.robot_position.x = robot_position[0]
        self.msg.robot_position.y = robot_position[1]
        self.msg.robot_position.theta = robot_position[2]

    def setMCLPosition(self, mcl_position):
        self.msg.mcl_position.x = mcl_position[0]
        self.msg.mcl_position.y = mcl_position[1]
        self.msg.mcl_position.theta = mcl_position[2]
    
    def setOdometryPosition(self, odometry_position):
        self.msg.odometry_position.x = odometry_position[0]
        self.msg.odometry_position.y = odometry_position[1]
        self.msg.odometry_position.theta = odometry_position[2]
    
    def setParticles(self, particles):
        for particle in particles:
            p = self.msg.particles.add()
            p.w, p.x, p.y, p.theta = particle

    def setMCLMessage(self, robot_position, particles, mcl_position, odometry_position, time_steps):
        self.msg = pb.ParticleList()
        self.setRobotPosition(robot_position)
        self.setParticles(particles)
        self.setMCLPosition(mcl_position)
        self.setOdometryPosition(odometry_position)
        self.msg.time_steps = time_steps

    def sendMCLMessage(self):
        self.udp_sock.sendto(self.msg.SerializeToString(), (self.address, self.port))
    

def run_receiver():
    UDP = ParticlesReceiver()

    while True:
        has_msg, robot_position, particles, mcl_position, odometry_position, time_steps = UDP.recvMCLMessage()
        print(has_msg, robot_position, particles, mcl_position, odometry_position, time_steps)

def run_sender():
    UDP = ParticlesSender()

    particles = np.array([
        [1.0, 2.0, 3.0, 0.5],
        [2.0, 3.0, 4.0, 1.0],
        [3.0, 4.0, 5.0, 1.5]
    ])
    UDP.setMCLMessage([1, 1, 1], 
                      particles, 
                      [1, 1, 1], 
                      [1, 1, 1], 
                      1)
    while True:
        UDP.sendMCLMessage()

if __name__ == "__main__":
    import particles_pb2 as pb


    is_receiver = False

    if is_receiver: run_receiver()
    else: run_sender()

else:
    from rsoccer_gym.Utils import particles_pb2 as pb
