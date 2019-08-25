import os

import matplotlib as mpl
import matplotlib.pyplot as plt

RESULTS = '../results'

if not os.path.isdir(RESULTS):
    os.mkdir(RESULTS)


def save_image(data, dir_, name):
    mpl.rcParams['savefig.pad_inches'] = 0

    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)

    # Plot the data.
    plt.plot(data)
    # Save the figure and display the figure.
    plt.savefig(f'{dir_}/{name}')
    plt.show()
