import os
import csv
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
    