import numpy as np
import matplotlib.pyplot as plt
from load_processing_data import *

if __name__ == "__main__":
    # CHOOSE SCENARIO
    scenarios = ['rnd_01', 'sqr_02', 'igs_03', 'tst_01']
    scenario = scenarios[0]

    # SET PATH AND READ DATA
    cwd = os.getcwd()
    path_to_log = cwd+f"/msc_experiments/23jun/{scenario}.csv"
    data = Read(path_to_log)

    # Processing times in seconds
    vision_total_time = np.mean(1000*data.get_vision()[1:, 0])
    objects_detection_time = np.mean(1000*data.get_vision()[1:, 3])
    boundary_detection_time = np.mean(1000*data.get_vision()[1:, 2])
    camera_transformation_time = np.mean(1000*data.get_vision()[1:, 1])

    # Define the labels and processing times
    labels = ['Objects Detection', 'Boundary Detection', 'Camera Transformation', 'Total']
    processing_times = [objects_detection_time, boundary_detection_time, camera_transformation_time, vision_total_time]

    # Set the colors for the bars
    colors = ['blue', 'orange', 'green', 'red']

    # Calculate the starting position and width for each bar
    start_positions = np.cumsum([0] + processing_times[:-1])
    start_positions[-1] = 0
    bar_widths = processing_times

    # Create the figure and axes objects
    fig, ax = plt.subplots(figsize=(10,3))

    # Create the bars for processing times
    bars = ax.barh(labels, bar_widths, left=start_positions, color=colors, height=0.7)

    # Set the x-axis label
    ax.set_xlabel('Processing Time (ms)')

    # Set the title of the plot
    ax.set_title('Vision Processing Time Comparison')

    # Add the values next to the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.annotate(f'{width:.2f}ms', xy=(start_positions[i] + width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords='offset points', ha='left', va='center')


    # Display the plot
    plt.tight_layout()
    plt.show()
