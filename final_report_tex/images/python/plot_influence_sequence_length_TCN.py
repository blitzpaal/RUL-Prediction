import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# Enter data.
sliding_window_length = np.array([10, 15, 20, 30])
val_accuracy = np.array([0.754811764, 0.747607946, 0.780011714, 0.771783829])
#p_1 = np.poly1d(np.polyfit(wallThickness,displacement[0],3))

# Plot the diagram.
fig, ax = plt.subplots(figsize=(5,3))

ax.plot(sliding_window_length, val_accuracy, linestyle='-', marker='o', color='k')

ax.set(xlabel='Sliding window length', ylabel='Validation accuracy')
#ax.legend()
ax.grid()

plt.savefig('influence_sequence_length_tcn.pdf', bbox_inches='tight')
plt.show()