import csv
import numpy as np

class Read:
    def __init__(self, 
                 path):
        self.path = path
        self.fields = []
        self.set_size = []
        self.ground_truth = []
        self.mcl = []
        self.odometry = []
        self.timestamps = []
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    self.fields = row
                    line_count += 1
                else:
                    self.set_size.append(int(row[1]))
                    self.ground_truth.append([float(row[2]), float(row[3]), float(row[4])])
                    self.mcl.append([float(row[5]), float(row[6]), float(row[7])])
                    self.odometry.append([float(row[8]), float(row[9]), float(row[10])])
                    self.timestamps.append(float(row[11]))
                    line_count += 1
    
    def get_fields(self):
        return self.fields

    def get_set_sizes(self):
        return np.array(self.set_size)

    def get_ground_truth(self):
        return np.array(self.ground_truth)

    def get_mcl(self):
        return np.array(self.mcl)
    
    def get_odometry(self):
        return np.array(self.odometry)
    
    def get_timestamps(self):
        return np.array(self.timestamps)

if __name__ == "__main__":
    import os

    cwd = os.getcwd()

    # Choose scenario
    scenarios = ['rnd_01', 'sqr_02', 'igs_03']
    rounds = [1, 2, 3]

    path = cwd+f'/msc_experiments/logs/27jun/seed/localization/{scenarios[1]}_{rounds[0]}.csv'

    data = Read(path)

    print(data.odometry[1000])