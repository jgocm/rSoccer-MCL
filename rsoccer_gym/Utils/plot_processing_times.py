import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import load_processing_data

def compare_processing_times(path, scenario, round, is_seed, linewidth):
    if is_seed:
        path_to_mcl = path + f'/fixed/seed/processing_times/{scenario}_{round}.csv'
        path_to_amcl = path + f'/adaptive/seed/processing_times/{scenario}_{round}.csv'
        print(f'scenario: {scenario}_{round} from seed position')

    else:
        path_to_mcl = path + f'/fixed/random/processing_times/{scenario}_{round}.csv'
        path_to_amcl = path + f'/adaptive/random/processing_times/{scenario}_{round}.csv'
        print(f'scenario: {scenario}_{round} from random position')

    data_mcl = load_processing_data.Read(path_to_mcl)
    data_amcl = load_processing_data.Read(path_to_amcl)

    # Get processing times from MCL
    mcl_frames = data_mcl.get_frames()
    mcl_set_sizes = data_mcl.get_set_sizes()
    mcl_times = 1000*data_mcl.get_localization()[:, 0]

    # Get processing times from AMCL
    amcl_frames = data_amcl.get_frames()
    amcl_set_sizes = data_amcl.get_set_sizes()
    amcl_times = 1000*data_amcl.get_localization()[:, 0]

    # Create a figure and axes
    fig, ax1 = plt.subplots()

    # Plot amcl_set_sizes on the left y-axis
    ax1.plot(mcl_frames, mcl_set_sizes, linewidth=linewidth, color='blue', label='MCL Set Sizes')    
    ax1.plot(amcl_frames, amcl_set_sizes, linewidth=linewidth, linestyle='--', color='blue', label='AMCL Set Sizes')
    ax1.set_xlabel('Frame Nr')
    ax1.set_ylabel('Number of Particles', color='b')

    # Create a twin axes for amcl_times on the right y-axis
    ax2 = ax1.twinx()
    ax2.plot(mcl_frames, mcl_times,  linewidth=linewidth, color='red', label='MCL Times')
    ax2.plot(amcl_frames, amcl_times, linewidth=linewidth, linestyle='--', color='red', label='AMCL Times')
    ax2.set_ylabel('Processing Time (ms)', color='r')

    # Draw the red horizontal line and add a legend
    line = ax2.axhline(y=44, color='black')
    # Add annotation above the line
    ax2.annotate('44', xy=(1.02*mcl_frames[-1], 44), xytext=(1.06*mcl_frames[-1], 46),
                arrowprops=dict(facecolor='black', arrowstyle='->'))

    # Draw the red horizontal line and add a legend
    line = ax2.axhline(y=11.5, color='black')
    # Add annotation above the line
    ax2.annotate('11.5', xy=(1.02*mcl_frames[-1], 11.5), xytext=(1.06*mcl_frames[-1], 7),
                arrowprops=dict(facecolor='black', arrowstyle='->'))

    # Set the title with increased font size
    ax1.set_title('Processing Times Comparison', fontsize=17)
    
    # Show the legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    # Increase the font size of the X and Y axis numbers
    ax1.tick_params(axis='both', which='major', labelsize=17)
    ax2.tick_params(axis='both', which='major', labelsize=17)

    # Show the plot
    plt.show()


def load_and_box_plot_processing_times(path, scenario, round):
    data = load_processing_data.Read(path + f"/{scenario}_{round}.csv")
    
    total = data.get_totals()[1:]
    vision = data.get_vision()[1:, 0]
    localization = data.get_localization()[1:, 0]

    # Set plots
    fig = plt.figure(figsize = (10, 7))

    # Set field labels
    labels = ['Total', 'Vision', 'Localization']

    # Create a figure and axis
    ax = fig.add_axes([0.07, 0.07, 0.91, 0.91])

    # Plot the boxplots
    boxplot = ax.boxplot([total, vision, localization], showfliers=False)

    # Set the x-axis tick labels
    ax.set_xticklabels(labels)

    # Set y-axis label
    ax.set_ylabel('Processing Time (s)')

    # Show the plot
    plt.show()
    
if __name__ == "__main__":
    # SET PATH AND READ DATA
    cwd = os.getcwd()

    # CHOOSE SCENARIO
    scenarios = ['rnd_01', 'sqr_02', 'igs_03']
    rounds = [1, 2, 3]

    for scenario in scenarios:
        for round in rounds:
            path = cwd+f'/msc_experiments/results'
            is_seed = True
            linewidth = 1.5
            compare_processing_times(path, scenario, round, is_seed, linewidth)
