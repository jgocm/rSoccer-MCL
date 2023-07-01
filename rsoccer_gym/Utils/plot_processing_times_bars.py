import os
import numpy as np
import matplotlib.pyplot as plt
import load_processing_data

def load_and_bar_plot_localization_processing_times(path, scenario, round, is_seed):
    if is_seed:
        path_to_log = path + f'/fixed/seed/processing_times/{scenario}_{round}.csv'
        print(f'scenario: {scenario}_{round} from seed position')

    else:
        path_to_log = path + f'/fixed/random/processing_times/{scenario}_{round}.csv'
        print(f'scenario: {scenario}_{round} from random position')

    data = load_processing_data.Read(path_to_log)

    # Processing times in seconds
    frame_nr = 100
    set_size = data.get_set_sizes()[frame_nr]
    localization_total_time = data.get_localization_total_ms()[frame_nr]
    resampling_time = data.get_resampling_ms()[frame_nr]
    avg_particle_time = data.get_avg_particle_ms()[frame_nr]
    weights_normalization_time = data.get_weights_normalization_ms()[frame_nr]
    likelihood_estimation_time = data.get_likelihood_update_ms()[frame_nr]
    particles_propagation_time = data.get_particles_propagation_ms()[frame_nr]

    # Define the labels and processing times
    labels = ['Particles Propagation',
              'Likelihood Estimation', 
              'Weights Update',
              'Confidence & Size Update',
              'Resampling',
              'Total']
    processing_times = [particles_propagation_time,
                        likelihood_estimation_time,
                        weights_normalization_time, 
                        avg_particle_time, 
                        resampling_time, 
                        localization_total_time]

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
    ax.set_title(f'Localization Processing Time Comparison (M={set_size})')

    # Add the values next to the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.annotate(f'{width:.2f}ms', xy=(start_positions[i] + width, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords='offset points', ha='left', va='center')


    # Display the plot
    plt.tight_layout()
    plt.show()

def load_and_bar_plot_vision_processing_times(path, scenario, round, is_seed):
    if is_seed:
        path_to_log = path + f'/fixed/seed/processing_times/{scenario}_{round}.csv'
        print(f'scenario: {scenario}_{round} from seed position')

    else:
        path_to_log = path + f'/fixed/random/processing_times/{scenario}_{round}.csv'
        print(f'scenario: {scenario}_{round} from random position')

    data = load_processing_data.Read(path_to_log)

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


if __name__ == "__main__":
    # SET PATH AND READ DATA
    cwd = os.getcwd()

    # CHOOSE SCENARIO
    scenarios = ['rnd_01', 'sqr_02', 'igs_03', 'tst_01']
    scenario = scenarios[0]
    round = 1

    path = cwd+f'/msc_experiments/results'
    is_seed = True
    load_and_bar_plot_localization_processing_times(path, scenario, round, is_seed)
    #load_and_bar_plot_vision_processing_times(path, scenario, round, is_seed)


