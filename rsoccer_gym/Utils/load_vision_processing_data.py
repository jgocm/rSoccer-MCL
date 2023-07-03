import os
import csv
import numpy as np

class Read:
    def __init__(self, 
                 path):
        self.path = path
        self.fields = []
        self.frames = []
        self.total = []
        self.objects_detection = []
        self.scan_arrangment = []
        self.boundary_detection = []
        self.line_detection = []
        self.camera_transformation = []

        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    self.fields = row
                    line_count += 1
                else:
                    self.frames.append(int(row[0]))
                    self.total.append(float(row[1]))
                    self.objects_detection.append(float(row[2]))
                    self.scan_arrangment.append(float(row[3]))
                    self.boundary_detection.append(float(row[4]))
                    self.line_detection.append(float(row[5]))
                    self.camera_transformation.append(float(row[6]))
                    line_count += 1
    
    def get_fields(self):
        return self.fields

    def get_frames(self):
        return np.array(self.frames)

    def get_total_ms(self):
        return np.array(self.total)

    def get_objects_detection_ms(self):
        return np.array(self.objects_detection)

    def get_scan_arrangment_ms(self):
        return np.array(self.scan_arrangment)
    
    def get_boundary_detection_ms(self):
        return np.array(self.boundary_detection)

    def get_line_detection_ms(self):
        return np.array(self.line_detection)
    
    def get_camera_transformation_ms(self):
        return np.array(self.camera_transformation)

if __name__ == "__main__":

    # CHOOSE SCENARIO
    scenarios = ['rnd_01', 'sqr_02', 'igs_03', 'tst_01']
    scenario = scenarios[2]

    # SET PATH
    cwd = os.getcwd()
    path_to_log = cwd+f"/msc_experiments/logs/03jul/processing_times/{scenario}.csv"

    data = Read(path_to_log)

    print(np.mean(data.get_boundary_detection_ms()))