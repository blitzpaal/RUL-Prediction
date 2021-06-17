import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import csv

# Import data.
structures_10_training = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model3_6/100_structures_randomSearch_6/Postprocessing/run-7a9af60455c9cd328bf4cc277cfe6bd0_execution0_train-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        structures_10_training[0].append(int(row[1]))
        structures_10_training[1].append(float(row[2]))

structures_10_validation = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model3_6/100_structures_randomSearch_6/Postprocessing/run-7a9af60455c9cd328bf4cc277cfe6bd0_execution0_validation-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        structures_10_validation[0].append(int(row[1]))
        structures_10_validation[1].append(float(row[2]))

structures_15_training = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model3_6/100_structures_randomSearch_5/Postprocessing/run-2c4fe9f99736869bf2cd09100d91bf9c_execution0_train-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        structures_15_training[0].append(int(row[1]))
        structures_15_training[1].append(float(row[2]))

structures_15_validation = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model3_6/100_structures_randomSearch_5/Postprocessing/run-2c4fe9f99736869bf2cd09100d91bf9c_execution0_validation-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        structures_15_validation[0].append(int(row[1]))
        structures_15_validation[1].append(float(row[2]))

structures_20_training = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model3_6/100_structures_randomSearch_4/Postprocessing/run-5517673b884050688ecd77962f1f6126_execution0_train-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        structures_20_training[0].append(int(row[1]))
        structures_20_training[1].append(float(row[2]))

structures_20_validation = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model3_6/100_structures_randomSearch_4/Postprocessing/run-5517673b884050688ecd77962f1f6126_execution0_validation-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        structures_20_validation[0].append(int(row[1]))
        structures_20_validation[1].append(float(row[2]))

structures_30_training = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model3_6/100_structures_randomSearch_3/Postprocessing/run-c8c73b517af1a2749565b41ba3df98ab_execution0_train-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        structures_30_training[0].append(int(row[1]))
        structures_30_training[1].append(float(row[2]))

structures_30_validation = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model3_6/100_structures_randomSearch_3/Postprocessing/run-c8c73b517af1a2749565b41ba3df98ab_execution0_validation-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        structures_30_validation[0].append(int(row[1]))
        structures_30_validation[1].append(float(row[2]))


# Plot the diagram.
fig, ax = plt.subplots(figsize=(4,3))

ax.plot(structures_10_training[0], structures_10_training[1], linestyle='--', linewidth=0.75, marker='', color='k', label='Training')
ax.plot(structures_10_validation[0], structures_10_validation[1], linestyle='-', linewidth=1.5, marker='', color='b', label='Validation')

ax.set(xlabel='Training epochs', ylabel='Accuracy')
ax.legend()
ax.grid()

plt.savefig('accuracy_100_structures_random_search_10_TCN.pdf', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(4,3))

ax.plot(structures_15_training[0], structures_15_training[1], linestyle='--', linewidth=0.75, marker='', color='k', label='Training')
ax.plot(structures_15_validation[0], structures_15_validation[1], linestyle='-', linewidth=1.5, marker='', color='b', label='Validation')

ax.set(xlabel='Training epochs', ylabel='Accuracy')
ax.legend()
ax.grid()

plt.savefig('accuracy_100_structures_random_search_15_TCN.pdf', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(4,3))

ax.plot(structures_20_training[0], structures_20_training[1], linestyle='--', linewidth=0.75, marker='', color='k', label='Training')
ax.plot(structures_20_validation[0], structures_20_validation[1], linestyle='-', linewidth=1.5, marker='', color='b', label='Validation')

ax.set(xlabel='Training epochs', ylabel='Accuracy')
ax.legend()
ax.grid()

plt.savefig('accuracy_100_structures_random_search_20_TCN.pdf', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(4,3))

ax.plot(structures_30_training[0], structures_30_training[1], linestyle='--', linewidth=0.75, marker='', color='k', label='Training')
ax.plot(structures_30_validation[0], structures_30_validation[1], linestyle='-', linewidth=1.5, marker='', color='b', label='Validation')

ax.set(xlabel='Training epochs', ylabel='Accuracy')
ax.legend()
ax.grid()

plt.savefig('accuracy_100_structures_random_search_30_TCN.pdf', bbox_inches='tight')
plt.show()