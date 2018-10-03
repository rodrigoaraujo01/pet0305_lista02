#!/usr/local/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans


class DataGenerator(object):
    def __init__(self):
        self.full_data = np.array(
            [
                [-7.82, -4.58, -3.97],
                [-6.68, 3.16, 2.71],
                [4.36, -2.19, 2.09],
                [6.72, 0.88, 2.80],
                [-8.64, 3.06, 3.50],
                [-6.87, 0.57, -5.45],
                [4.47, -2.62, 5.76],
                [6.73, -2.01, 4.18],
                [-7.71, 2.34, -6.33],
                [-6.91, -0.49, -5.68],
                [6.18, 2.81, 5.82],
                [6.72, -0.93, -4.04],
                [-6.25, -0.26, 0.56],
                [-6.94, -1.22, 1.13],
                [8.09, 0.20, 2.25],
                [6.81, 0.17, -4.15],
                [-5.19, 4.24, 4.04],
                [-6.38, -1.74, 1.43],
                [4.08, 1.30, 5.33],
                [6.27, 0.93, -2.78],
            ]
        )


def main():
    generator = DataGenerator()
    data = generator.full_data
    fig = plt.figure()
    for i in range(6):
        y_pred = KMeans(n_clusters=i+3, random_state=100).fit_predict(data)
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        ax.set_title(f'{i+3} Clusters')
        ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=y_pred, depthshade=False)
    plt.show()


if __name__ == "__main__":
    main()
