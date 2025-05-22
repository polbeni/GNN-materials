# Pol Benítez Colominas, May 2025
# Universitat Politècnica de Catalunya

# Generates training, validation, and test sets



################################# LIBRARIES ###############################
import csv
import random
###########################################################################



############################# FIXED PARAMETERS ############################
train_set_size = 0.7                                                    # Fraction of the trainin set size
validation_set_size = 0.15                                              # Fraction of the validation set size
test_set_size = 0.15                                                    # Fraction of the test set size

path_to_graphs = '../materials-dataset-new/normalized_graphs/'          # Path to the normalized graphs
path_to_csv = '../materials-dataset-new/graphs-bg.csv'                  # Path to csv file with graphs names and value

min_bg_boolean = True                                                   # Ignore materials with band gaps smaller than the desired value
min_bg = 0.4    
###########################################################################



################################### MAIN ##################################
# Split between train+validation and test
file_list = []

with open(path_to_csv, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    for row in csv_reader:
        if min_bg_boolean == True:
            if float(row[1]) > min_bg: 
                file_list.append(row[0])
        else:           
            file_list.append(row[0])

random.shuffle(file_list)

file_list_train = []
file_list_test = []
for graph_num in range(len(file_list)):
    if graph_num <= ((train_set_size + validation_set_size) * len(file_list)):
        file_list_train.append(file_list[graph_num])
    else:
        file_list_test.append(file_list[graph_num])

with open('train_val_graphs.txt', 'w') as file:
    for graph_num in range(len(file_list_train)):
        file.write(f'{file_list_train[graph_num]}\n')

with open('test_graphs.txt', 'w') as file:
    for graph_num in range(len(file_list_test)):
        file.write(f'{file_list_test[graph_num]}\n')
###########################################################################