import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import csv

# Import data.
training = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model1_3/500_structures_adaptiveLR_1/postprocessing/run-train-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        training[0].append(int(row[1]))
        training[1].append(float(row[2]))

validation = [[],[]]
with open('C:/Users/pajo8/RUL Prediction/CNN_Model1_3/500_structures_adaptiveLR_1/postprocessing/run-validation-tag-epoch_sparse_categorical_accuracy.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    next(reader)
    for row in reader:
        validation[0].append(int(row[1]))
        validation[1].append(float(row[2]))
        
# Plot the diagram.
fig, ax = plt.subplots(figsize=(4,3))

ax.plot(training[0], training[1], linestyle='--', linewidth=0.75, marker='', color='k', label='Training')
ax.plot(validation[0], validation[1], linestyle='-', linewidth=1.5, marker='', color='b', label='Validation')

ax.set(xlabel='Training epochs', ylabel='Accuracy')
ax.legend()
ax.grid()

plt.savefig('accuracy_500_structures_adaptiveLR_CNN.pdf', bbox_inches='tight')
plt.show()