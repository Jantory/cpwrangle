import numpy as np
import os

def extract_time_values(file_path):
    time_values = []

    with open(file_path, 'r') as file:
        for line in file:
            if "Train Time:" in line:
                time_section = line.strip().split(':')[-1].strip()
                time_values.append(float(time_section[:-5]))

    time_array = np.array(time_values)

    return time_array

# # Example usage:
# file_path = '../t5-large/Rest.txt'  # Replace with your file path
# time_values = extract_time_values(file_path)
# time_values = time_values.reshape((-1, 100))
# print(np.mean(time_values, axis=1))

# iterate over a folder
def extract_time_values_from_folder(folder_path):
    time_values = []
    for file_name in os.listdir(folder_path):
        print(file_name)
        file_path = os.path.join(folder_path, file_name)
        time_values.append(np.mean(extract_time_values(file_path)))
    time_array = np.array(time_values)
    return time_array

# Example usage:
folder_path = 'ptune'
time_values = extract_time_values_from_folder(folder_path)
print(time_values)

# DBGo-lora8-lr0.001.txt
# AmGo-lora8-lr0.001.txt
# Beer-lora8-lr0.001.txt
# WaAm-lora8-lr0.001.txt
# Buy-lora8-lr0.001.txt
# iTAm-lora8-lr0.001.txt
# Hosp-lora8-lr0.001.txt
# Rest-lora8-lr0.001.txt
# DBAC-lora8-lr0.001.txt
# FoZa-lora8-lr0.001.txt
# [290.49137357  77.64172784   2.61728648  77.32135161   6.05759859
#    5.92817012  14.52422116   7.77364241 142.25592027   9.05253221]


# Hosp-prefix50-lr0.2.txt
# Rest-prefix50-lr0.2.txt
# iTAm-prefix50-lr0.2.txt
# DBAC-prefix50-lr0.2.txt
# AmGo-prefix50-lr0.2.txt
# FoZa-prefix50-lr0.2.txt
# Buy-prefix50-lr0.2.txt
# DBGo-prefix50-lr0.2.txt
# WaAm-prefix50-lr0.2.txt
# Beer-prefix50-lr0.2.txt
# [  8.5884923    6.01445942   3.57752132  87.84038896   50.51268101
#    5.60572532   4.62420552 170.64380411  50.18732839   1.83698005]

# Beer-prompt50-lr0.2.txt
# WaAm-prompt50-lr0.2.txt
# DBGo-prompt50-lr0.2.txt
# Buy-prompt50-lr0.2.txt
# DBAC-prompt50-lr0.2.txt
# FoZa-prompt50-lr0.2.txt
# AmGo-prompt50-lr0.2.txt
# iTAm-prompt50-lr0.2.txt
# Rest-prompt50-lr0.2.txt
# Hosp-prompt50-lr0.2.txt
# [  3.04036859  94.21280858 311.72546287   7.45200588 157.96687261
#    9.88527749  89.46467749    6.70639785   8.70892653  15.39917555]

# DBAC-ptune60-lr0.2.txt
# Rest-ptune60-lr0.2.txt
# Hosp-ptune60-lr0.2.txt
# iTAm-ptune60-lr0.2.txt
# Buy-ptune60-lr0.2.txt
# WaAm-ptune60-lr0.2.txt
# FoZa-ptune60-lr0.2.txt
# DBGo-ptune60-lr0.2.txt
# Beer-ptune60-lr0.2.txt
# AmGo-ptune60-lr0.2.txt
# [179.553902     8.98443866  15.95491045   7.13533565   7.6178374
#   94.83812459  10.71266824 375.01323941   3.23964205  98.15533753]



itunes = [5.92817012, 3.57752132, 6.70639785, 7.13533565]
beer = [2.61728648, 1.83698005, 3.04036859, 3.23964205]
foza = [9.05253221, 5.60572532, 9.88527749, 10.71266824]
woam = [77.32135161, 50.18732839, 94.21280858, 94.83812459]
amgo = [77.64172784, 50.51268101, 89.46467749, 98.15533753]
dbac = [142.25592027, 87.84038896, 157.96687261, 179.553902]
dbgo = [290.49137357, 170.64380411, 311.72546287, 375.01323941]
hosp = [14.52422116, 8.5884923, 15.39917555, 15.95491045]
buy = [6.05759859, 4.62420552, 7.45200588, 7.6178374]
rest = [7.77364241, 6.01445942, 8.70892653, 8.98443866]







