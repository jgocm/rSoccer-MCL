import csv
import matplotlib.pyplot as plt

scenario = 'igs_03'
file_path = f'/home/vision-blackout/rSoccer-MCL/msc_experiments/23jun/{scenario}.csv'

# Initialize empty lists to store the data
columns = []
data = []

# Read the CSV file
with open(file_path, 'r') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)  # Read the header row
    columns = header[1:]  # Exclude the first column

    # Iterate over the remaining rows
    for row in csv_reader:
        data.append([float(value) for value in row[1:]])  # Convert values to float and exclude the first column

# Determine the number of subplots
num_subplots = len(columns)

# Set up the subplots
fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 6), sharex=True)

# Plotting the processing time range for each column
for i, column_data in enumerate(data):
    ax = axes[i]
    ax.boxplot(column_data)

    # Set labels for each subplot
    ax.set_ylabel('Processing Time')

    # Set title for each subplot
    ax.set_title(columns[i])

# Set x-axis label for the last subplot
axes[-1].set_xlabel('Column')

# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()
