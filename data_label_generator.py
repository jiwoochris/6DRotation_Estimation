import csv
import random

# Define the number of data points
num_data_points = 5000

# Open (or create) a csv file in write mode
with open('ypr_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["yaw", "pitch", "roll"])  # Write the header

    # Generate and write the data
    for _ in range(num_data_points):
        x = random.uniform(-50, 50)
        y = random.uniform(-50, 50)
        z = random.uniform(-50, 50)
        writer.writerow([x, y, z])
