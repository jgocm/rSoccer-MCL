import time
import os
import cv2
import csv
import numpy as np
from rsoccer_gym.Perception.jetson_vision import *
from rsoccer_gym.Utils import load_localization_data

def save_processing_times_log(cwd, log, scenario):
    if len(log)>1:
        print("SAVING PROCESSING TIMES LOG FILE")
        dir = cwd+f"/msc_experiments/logs/03jul/processing_times/{scenario}.csv"
        fields = ["FRAME NR",               # index 0
                  "VISION TOTAL",           # index 1
                  "OBJECTS DETECTION",      # index 2
                  "SCAN ARRENGMENT",        # index 3
                  "BOUNDARY DETECTION",     # index 4
                  "LINE DETECTION",         # index 5
                  "CAMERA TRANSFORMATION"]  # index 6
        
        with open(dir, 'w') as f:
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(log)    

def get_image_from_frame_nr(path_to_images_folder, frame_nr):
    dir = path_to_images_folder+f'/cam/{frame_nr}.png'
    img = cv2.imread(dir)
    return img

def serialize_vision_processing_times_for_log(frame_nr, times):
    times = np.array(times)
    elapsed_times = times[1:] - times[:-1]
    data = [frame_nr,
            times[-1]-times[0],
            elapsed_times[0],
            elapsed_times[1],
            elapsed_times[2],
            elapsed_times[3],
            elapsed_times[4]]
    return data
    
if __name__ == "__main__":
    cwd = os.getcwd()

    WINDOW_NAME = "Vision Processing"
    scenarios = ['rnd_01', 'sqr_02', 'igs_03']
    debug = False

    for scenario in scenarios:
        path = f'/home/vision-blackout/ssl-navigation-dataset-jetson/data/{scenario}'
        path_to_log = path+'/logs/processed.csv'
        data = load_localization_data.Read(path_to_log, is_raw=False, degrees=False)
        frames = data.get_frames()

        vision = JetsonVision(vertical_lines_nr=1, 
                            enable_field_detection=True,
                            enable_randomized_observations=True,
                            score_threshold=0.35,
                            draw=debug,
                            debug=debug)
        vision.jetson_cam.setPoseFrom3DModel(167, 106.7)
    
        frame_nr = frames[0]
        log = []

        while frame_nr<frames[-1]:
            img = get_image_from_frame_nr(path, frame_nr)
            height, width = img.shape[0], img.shape[1]
            _, _, _, _, particle_filter_observations, times = vision.process(img, timestamp=time.time())
            has_goal, boundary_ground_points, line_ground_points = particle_filter_observations
            vision_data = serialize_vision_processing_times_for_log(frame_nr, times)
            if debug:
                cv2.imshow(WINDOW_NAME, img[:300, :])
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('s'):
                    cv2.imwrite(WINDOW_NAME + '.png', img[:300, :])
            frame_nr=frame_nr+1
            
            print(f"Scenario: {scenario}, Frame: {frame_nr}, Objects: {vision_data[2]:.3f}, Boundary: {vision_data[4]:.3f}, Line: {vision_data[5]:.3f}")
            if vision_data[1]<1:
                log.append(vision_data)
            
        save_processing_times_log(cwd, log, scenario)
    
    print("FINISHED VISION LOGGING")