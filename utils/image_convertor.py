import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


def save_image(data, dir_, name):
    mpl.rcParams['savefig.pad_inches'] = 0

    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)

    # Plot the data.
    plt.plot(data)
    plt.savefig(f'{dir_}/{name}')
    plt.show()

    if os.path.isfile(f'{dir_}/{name}'):
        print(f'File {name} succesful saved!')


def main():
    DATA_FOLDER = '../data'
    RESULTS = '../results'
    DATA = os.listdir(DATA_FOLDER)

    if not os.path.isdir(RESULTS):
        os.mkdir(RESULTS)

    for image in DATA:
        data = pd.read_csv(f'{DATA_FOLDER}/{image}', index_col='time', engine='python')
        print(data.shape)
        name = image.split('.')[0]
        name += '.png'
        # save image
        save_image(data, RESULTS, name)


if __name__ == '__main__':
    pass
    # main()



