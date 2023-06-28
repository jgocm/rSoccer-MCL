import numpy as np
import cv2
import matplotlib.pyplot as plt
from load_trajectories import *

def load_plot_and_save_trajectories(path, scenario, round, end, linewidth):
    data = Read(path + f"/{scenario}_{round}.csv")

    # Load the data from the text files
    ground_truth_data = data.get_ground_truth()
    mcl_data = data.get_mcl()
    odometry_data = data.get_odometry()

    # Extract the X and Y coordinates from the data
    gt_x_coords = ground_truth_data[:end, 0]
    gt_y_coords = ground_truth_data[:end, 1]
    mcl_x_coords = mcl_data[:end, 0]
    mcl_y_coords = mcl_data[:end, 1]
    odo_x_coords = odometry_data[:end, 0]
    odo_y_coords = odometry_data[:end, 1]

    # Create a new figure with a single subplot
    fig, ax = plt.subplots()

    # Create a scatter plot of the X and Y coordinates for each trajectory
    ax.plot(gt_x_coords, gt_y_coords, label='Ground Truth', linewidth=linewidth, color='red')
    ax.plot(mcl_x_coords, mcl_y_coords, label='MCL', linewidth=linewidth, linestyle='--', color='blue')
    ax.plot(odo_x_coords, odo_y_coords, label='Odometry', linewidth=linewidth, linestyle=':', color='green')

    # Set the X and Y limits of the plot
    ax.set_xlim([-0.3, 4.2])
    ax.set_ylim([-3, 3])

    # Increase the font size of the X and Y axis numbers
    ax.tick_params(axis='both', which='major', labelsize=17)

    # Add a legend to the plot
    ax.legend(fontsize=17)

    # Save the figure as a PNG file
    fig.savefig(path + f'/trajectory_comparison_{scenario}_{round}.png', dpi=400)

    # Show the plot
    plt.show()

def load_plot_and_save_distances(path, scenario, round, end, linewidth):
    data = Read(path + f"/{scenario}_{round}.csv")

    # Load the data from the text files
    ground_truth_data = data.get_ground_truth()
    mcl_data = data.get_mcl()
    odometry_data = data.get_odometry()
    timestamps_data = data.get_timestamps()

    # Extract the X and Y coordinates from the data
    end = min(end, len(timestamps_data))
    gt_coords = ground_truth_data[:end, :-1]
    mcl_coords = mcl_data[:end, :-1]
    odo_coords = odometry_data[:end, :-1]
    timestamps = timestamps_data[:end]

    dist_mcl = np.linalg.norm(mcl_coords-gt_coords, axis=1)
    dist_odometry = np.linalg.norm(odo_coords-gt_coords, axis=1)
    
    # Create a new figure with a single subplot
    fig, ax = plt.subplots()

    # Create a scatter plot of the X and Y coordinates for each trajectory
    ax.plot(timestamps, dist_mcl, label='MCL', linewidth=linewidth, linestyle='--', color='blue')
    ax.plot(timestamps, dist_odometry, label='Odometry', linewidth=linewidth, color='green')

    # Set the X and Y limits of the plot
    #ax.set_xlim([-0.3, 4.2])
    #ax.set_ylim([0, 0.9])

    # Increase the font size of the X and Y axis numbers
    ax.tick_params(axis='both', which='major', labelsize=17)

    # Add a legend to the plot
    ax.legend(fontsize=17)

    # Save the figure as a PNG file
    fig.savefig(path + f'/distance_comparison_{scenario}_{round}.png', dpi=400)

    # Show the plot
    # plt.show()

def crop_trajectory_fig(path, scenario, round):
    img = cv2.imread(path + f'/trajectory_comparison_{scenario}_{round}.png')
    crop = img[150:-50, 140:-220]
    cv2.imwrite(path + f'/trajectory_comparison_{scenario}_{round}_crop.png', crop)

def crop_distance_fig(path, scenario, round):
    img = cv2.imread(path + f'/distance_comparison_{scenario}_{round}.png')
    crop = img[200:-50, 100:-220]
    cv2.imwrite(path + f'/distance_comparison_{scenario}_{round}_crop.png', crop)

if __name__ == "__main__":
    import os

    cwd = os.getcwd()

    # Choose scenario
    scenarios = ['rnd_01', 'sqr_02', 'igs_03']
    rounds = [1, 2, 3]
    for scenario in scenarios:
        for round in rounds:
            end = 2000
            if scenario == 'rnd_01': end = 800
            linewidth = 2.5
            path = cwd+f'/msc_experiments/logs/27jun/random/localization'
            load_plot_and_save_distances(path, scenario, round, end, linewidth)
            crop_distance_fig(path, scenario, round)
            load_plot_and_save_trajectories(path, scenario, round, end, linewidth)
            crop_trajectory_fig(path, scenario, round)

