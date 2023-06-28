import os
import csv
import numpy as np

class Read:
    def __init__(self, 
                 path):
        self.path = path
        self.fields = []
        self.set_size = []
        self.total = []
        self.vision = []
        self.localization = []
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    self.fields = row
                    line_count += 1
                else:
                    self.set_size.append(int(row[1]))
                    self.total.append(float(row[2]))
                    self.vision.append([float(row[3]), float(row[4]), float(row[5]), float(row[6])])
                    self.localization.append([float(row[7]), float(row[8]), float(row[9]), float(row[10]), float(row[11]), float(row[12])])
                    line_count += 1
    
    def get_fields(self):
        return self.fields

    def get_set_sizes(self):
        return np.array(self.set_size)

    def get_totals(self):
        return np.array(self.total)

    def get_vision(self):
        return np.array(self.vision)
    
    def get_localization(self):
        return np.array(self.localization)
    
if __name__ == "__main__":

    # CHOOSE SCENARIO
    scenarios = ['rnd_01', 'sqr_02', 'igs_03', 'tst_01']
    scenario = scenarios[2]

    # SET PATH
    cwd = os.getcwd()
    path_to_log = cwd+f"/msc_experiments/logs/23jun/{scenario}.csv"

    data = Read(path_to_log)

    print(data.get_vision()[:, 0])
