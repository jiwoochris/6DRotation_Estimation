import csv
import random

# # Define the number of data points
# num_data_points = 5000

# # Open (or create) a csv file in write mode
# with open('new_xy_data.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["yaw", "roll"])  # Write the header

#     # Generate and write the data
#     for _ in range(num_data_points):
#         x = random.uniform(-50, 50)
#         y = random.uniform(-50, 50)
#         writer.writerow([x, y])




with open('new_xy_data.csv', 'w', newline='') as new_file:

    writer = csv.writer(new_file)
    writer.writerow(['ImageName', "x", "y"])

    with open('ypr_data.csv', 'r') as file:
        csvreader = csv.reader(file)
        next(csvreader)  # Skip the header
        for index, row in enumerate(csvreader):
            print(index+1, row[0], row[2])
            writer.writerow([f"{index+1}.png",row[0], row[2]])
