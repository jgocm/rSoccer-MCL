import os
import numpy as np
import matplotlib.pyplot as plt
import load_vision_processing_data

def load_and_box_plot_vision_processing_times(path, scenario):
    path_to_log = path + f'/processing_times/{scenario}.csv'
    print(f'Scenario: {scenario}')

    data = load_vision_processing_data.Read(path_to_log)

    # Processing times in seconds
    total = np.mean(data.get_total_ms())
    objects_detection = np.mean(data.get_objects_detection_ms())
    scan_arrangment = np.mean(data.get_scan_arrangment_ms())
    boundary_detection = np.mean(data.get_boundary_detection_ms())
    line_detection = np.mean(data.get_line_detection_ms())
    camera_transformation = np.mean(data.get_camera_transformation_ms())

    # Define the labels and processing times
    labels = ['Objects Detection',
              'Scan Line Arrangment', 
              'Boundary Detection',
              'Marking Detection',
              'Camera Transformation',
              'Total']
    processing_times = [objects_detection,
                        scan_arrangment,
                        boundary_detection, 
                        line_detection, 
                        camera_transformation, 
                        total]

    # Set the colors for the bars
    colors = ['orange', 'blue', 'fuchsia', 'green', 'yellow', 'red']

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
    ax.set_title(f'Vision Processing Time Analysis')

    # Add the values next to the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.annotate(f'{width:.3f}ms', xy=(start_positions[i] + width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords='offset points', ha='left', va='center')


    # Display the plot
    plt.tight_layout()
    plt.show()    

if __name__ == "__main__":
    # SET PATH AND READ DATA
    cwd = os.getcwd()

    # CHOOSE SCENARIO
    scenarios = ['rnd_01', 'sqr_02', 'igs_03']

    for scenario in scenarios:
        path = cwd+f'/msc_experiments/logs/03jul'
        load_and_box_plot_vision_processing_times(path, scenario)