import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


DATA_FOLDER = '../data'
RESULTS = '../data/results'
DATA = os.listdir(DATA_FOLDER)
print(os.path.isfile(RESULTS))

if not os.path.isdir(RESULTS):
    os.mkdir(RESULTS)


def save_image(data, dir, name):
    mpl.rcParams['savefig.pad_inches'] = 0
    ax = plt.axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)

    # Plot the data.
    plt.plot(data)
    # Save the figure and display the figure.
    plt.savefig(f'{dir}/{name}')
    plt.show()

    if os.path.isfile(f'{dir}/{name}'):
        print(f'File {name} succsesful saved!')


def main():
    for image in DATA:

        data = pd.read_csv(f'{DATA_FOLDER}/{image}', index_col='time', engine='python')
        print(data.shape)
        name = image.split('.')[0]
        name += '.png'

        save_image(data, RESULTS, name)

if __name__ == '__main__':
    main()





