import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

l=0.5
nb_bins = 20 # including one extra bin for RUL>upper_bin_bound
lower_bin_bound = 0
upper_bin_bound = 80000
highest_RUL = 81896

bins = np.linspace(lower_bin_bound**l, upper_bin_bound**l, nb_bins)**(1/l)
bins = np.append(bins, highest_RUL) # Append last class for RUL > 80000

labels=[i for i in range(bins.shape[0]-1)]

# Setup a plot such that only the bottom spine is shown
def setup(ax):
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(which='major', width=1.00)
    ax.tick_params(which='major', length=5)
    ax.tick_params(which='minor', width=0.75)
    ax.tick_params(which='minor', length=2.5)
    ax.set_xlim(-5000, 85000)
    ax.set_ylim(0, 0.5)
    ax.patch.set_alpha(0.0)

plt.figure(figsize=(7, 0.5))
n = 1
ax = plt.subplot(n, 1, 1)
setup(ax)
ax.xaxis.set_major_locator(ticker.MultipleLocator(10000))
ax.set_clip_on(False)
ax.plot(bins[:-1],np.full(bins[:-1].shape[0],0),'o', zorder=10, clip_on=False)

plt.savefig('RUL_classification.pdf', bbox_inches='tight')
plt.show()