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

    def pointsDistance(self, truth, predict):
        diff = np.subtract(truth, predict) # Points difference x2 - x1, y2 - y1
        return np.sum(np.square(diff), axis=1, keepdims=True) #  Components sum (x^2 + y^2) 

    def MSE(self, truth, predict):
        return self.pointsDistance(truth, predict).mean() # Square supressed by missing distance sqrt

    def RMSE(self, truth, predict):
        return np.sqrt(self.MSE(truth, predict))
    
    def get_trajectory_RMSEs(self):
        ground_truth = self.get_ground_truth()[:, :2]
        mcl = self.get_mcl()[:, :2]
        odometry = self.get_odometry()[:, :2]

        mcl_RMSE = self.RMSE(ground_truth, mcl)
        odometry_RMSE = self.RMSE(ground_truth, odometry)

        return mcl_RMSE, odometry_RMSE

    def get_avg_fps(self):
        timestamps = self.get_timestamps()
        avg_processing_time = np.mean(timestamps[1:] - timestamps[:-1])
        fps = 1/avg_processing_time
        return fps

if __name__ == "__main__":
    import os

    cwd = os.getcwd()

    # Choose scenario
    scenarios = ['rnd_01', 'sqr_02', 'igs_03']
    rounds = [1, 2, 3]

    path = cwd+f'/msc_experiments/logs/27jun/seed/localization/{scenarios[1]}_{rounds[0]}.csv'

    data = Read(path)

    print(data.get_avg_fps())