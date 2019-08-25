import os
import csv
import pandas as pd
import numpy as np
from PIL import Image

if __name__ == '__main__':
    folders = {'Y-': 1, 'Y+': 2, 'NO': 0}

    for folder, label in folders.items():
        images = os.listdir(folder)
        len_img = len(images)

        border = int(len_img * 0.8)
        train = images[:border]
        test = images[border:]

        for piece in [train, test]:
            if len(piece) > 100:
                name = 'train.csv'
            else:
                name = 'test.csv'
            for image in piece:

                # check if file is directory - skip it
                if os.path.isdir(f'{folder}/{image}'):
                    continue
                with open(name, 'a') as f:
                    writer = csv.writer(f)
                    a = [f'{folder}/{image}', label]
                    writer.writerow(a)