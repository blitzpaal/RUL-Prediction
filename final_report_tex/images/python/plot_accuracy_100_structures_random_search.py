import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import csv

# Import data.
structures_5_test = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model1_3/100_structures_randomSearch_9/postprocessing/run-a44e7c007603279dfb3e81ade853bbab_execution0_train-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        structures_5_test[0].append(int(row[1]))
        structures_5_test[1].append(float(row[2]))

structures_5_validation = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model1_3/100_structures_randomSearch_9/postprocessing/run-a44e7c007603279dfb3e81ade853bbab_execution0_validation-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        structures_5_validation[0].append(int(row[1]))
        structures_5_validation[1].append(float(row[2]))

structures_10_test = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model1_3/100_structures_randomSearch_8/postprocessing/run-b96db96a9c08a9c0499517083bd1c121_execution0_train-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        structures_10_test[0].append(int(row[1]))
        structures_10_test[1].append(float(row[2]))

structures_10_validation = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model1_3/100_structures_randomSearch_8/postprocessing/run-b96db96a9c08a9c0499517083bd1c121_execution0_validation-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        structures_10_validation[0].append(int(row[1]))
        structures_10_validation[1].append(float(row[2]))

structures_15_test = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model1_3/100_structures_randomSearch_7/postprocessing/run-130015d61b732d67af03f762c35bb0b4_execution0_train-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        structures_15_test[0].append(int(row[1]))
        structures_15_test[1].append(float(row[2]))

structures_15_validation = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model1_3/100_structures_randomSearch_7/postprocessing/run-130015d61b732d67af03f762c35bb0b4_execution0_validation-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        structures_15_validation[0].append(int(row[1]))
        structures_15_validation[1].append(float(row[2]))

structures_20_test = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model1_3/100_structures_randomSearch_6/postprocessing/run-764395dcbdd6ddbec683de292e9a87f3_execution0_train-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        structures_20_test[0].append(int(row[1]))
        structures_20_test[1].append(float(row[2]))

structures_20_validation = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model1_3/100_structures_randomSearch_6/postprocessing/run-764395dcbdd6ddbec683de292e9a87f3_execution0_validation-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        structures_20_validation[0].append(int(row[1]))
        structures_20_validation[1].append(float(row[2]))

structures_30_test = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model1_3/100_structures_randomSearch_4/postprocessing/run-5cf9a4569a169863092154708f71d48d_execution0_train-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        structures_30_test[0].append(int(row[1]))
        structures_30_test[1].append(float(row[2]))

structures_30_validation = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model1_3/100_structures_randomSearch_4/postprocessing/run-5cf9a4569a169863092154708f71d48d_execution0_validation-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        structures_30_validation[0].append(int(row[1]))
        structures_30_validation[1].append(float(row[2]))

structures_40_test = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model1_3/100_structures_randomSearch_5/postprocessing/run-7e8000fe71b5416675827b5b1d399584_execution0_train-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        structures_40_test[0].append(int(row[1]))
        structures_40_test[1].append(float(row[2]))

structures_40_validation = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model1_3/100_structures_randomSearch_5/postprocessing/run-7e8000fe71b5416675827b5b1d399584_execution0_validation-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        structures_40_validation[0].append(int(row[1]))
        structures_40_validation[1].append(float(row[2]))


# Plot the diagram.
fig, ax = plt.subplots(figsize=(4,3))

ax.plot(structures_5_test[0], structures_5_test[1], linestyle='--', linewidth=0.75, marker='', color='k', label='Test')
ax.plot(structures_5_validation[0], structures_5_validation[1], linestyle='-', linewidth=1.5, marker='', color='b', label='Validation')

ax.set(xlabel='Training epochs', ylabel='Accuracy')
ax.legend()
ax.grid()

plt.savefig('accuracy_100_structures_random_search_5.pdf', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(4,3))

ax.plot(structures_10_test[0], structures_10_test[1], linestyle='--', linewidth=0.75, marker='', color='k', label='Test')
ax.plot(structures_10_validation[0], structures_10_validation[1], linestyle='-', linewidth=1.5, marker='', color='b', label='Validation')

ax.set(xlabel='Training epochs', ylabel='Accuracy')
ax.legend()
ax.grid()

plt.savefig('accuracy_100_structures_random_search_10.pdf', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(4,3))

ax.plot(structures_15_test[0], structures_15_test[1], linestyle='--', linewidth=0.75, marker='', color='k', label='Test')
ax.plot(structures_15_validation[0], structures_15_validation[1], linestyle='-', linewidth=1.5, marker='', color='b', label='Validation')

ax.set(xlabel='Training epochs', ylabel='Accuracy')
ax.legend()
ax.grid()

plt.savefig('accuracy_100_structures_random_search_15.pdf', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(4,3))

ax.plot(structures_20_test[0], structures_20_test[1], linestyle='--', linewidth=0.75, marker='', color='k', label='Test')
ax.plot(structures_20_validation[0], structures_20_validation[1], linestyle='-', linewidth=1.5, marker='', color='b', label='Validation')

ax.set(xlabel='Training epochs', ylabel='Accuracy')
ax.legend()
ax.grid()

plt.savefig('accuracy_100_structures_random_search_20.pdf', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(4,3))

ax.plot(structures_30_test[0], structures_30_test[1], linestyle='--', linewidth=0.75, marker='', color='k', label='Test')
ax.plot(structures_30_validation[0], structures_30_validation[1], linestyle='-', linewidth=1.5, marker='', color='b', label='Validation')

ax.set(xlabel='Training epochs', ylabel='Accuracy')
ax.legend()
ax.grid()

plt.savefig('accuracy_100_structures_random_search_30.pdf', bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(4,3))

ax.plot(structures_40_test[0], structures_40_test[1], linestyle='--', linewidth=0.75, marker='', color='k', label='Test')
ax.plot(structures_40_validation[0], structures_40_validation[1], linestyle='-', linewidth=1.5, marker='', color='b', label='Validation')

ax.set(xlabel='Training epochs', ylabel='Accuracy')
ax.legend()
ax.grid()

plt.savefig('accuracy_100_structures_random_search_40.pdf', bbox_inches='tight')
plt.show()