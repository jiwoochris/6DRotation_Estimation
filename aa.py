import csv
import os

# with open('ypr_data.csv', 'r') as file:
#     csvreader = csv.reader(file)
#     next(csvreader)  # Skip the header
#     for index, row in enumerate(csvreader):
#         with open(f'data\DatasetProduce/{index+1}.txt', 'w') as output_file:
#             output_file.write(','.join(row))


with open('files.txt', 'w') as f:
    for i in range(1, 5001):
        f.write(str(i) + '\n')