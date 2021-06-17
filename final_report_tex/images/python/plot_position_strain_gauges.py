import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import csv

position = [[3, 14, 25], [14, 14, 14]]

# Plot the diagram.
fig, ax = plt.subplots(figsize=(4,3))

ax.plot(position[0], position[1], linestyle='', linewidth=0.75, marker='o', color='k')
#ax.annotate('(' + str(position[0][0]) + ',' + str(position[0][1] + ')', (position[0][0], position[0][1]))

ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)

plt.text(position[0][0]-7, position[1][0]+5, '({}, {})'.format(position[0][0], position[1][0]))
plt.text(position[0][1]-7, position[1][1]-10, '({}, {})'.format(position[0][1], position[1][1]))
plt.text(position[0][2]-7, position[1][2]+5, '({}, {})'.format(position[0][2], position[1][2]))

ax.set(xlabel='x-position in $\mathrm{mm}$', ylabel='y-position in $\mathrm{mm}$')
#ax.legend()
ax.grid()

plt.title('Strain gauges position')
plt.savefig('strain_gauges_position.pdf', bbox_inches='tight')
plt.show()