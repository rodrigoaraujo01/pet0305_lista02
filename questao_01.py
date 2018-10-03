#!/usr/local/bin/python
# -*- coding: utf-8 -*-


import numpy as np
from scipy.stats import multivariate_normal as mvn
from minisom import MiniSom
import matplotlib.pyplot as plt

class DataGenerator(object):
    def __init__(self):
        self.mu = [
            np.array([0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([4, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 4, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 4]),
        ]
        self.cov = np.identity(8)
        data = []
        for mu in self.mu:
            tmp_data = mvn.rvs(mean=mu, cov=self.cov, size=100)
            data.append(tmp_data)
        self.full_data = np.vstack(data)
        self.targets = []
        for i in range(4):
            target = [i for j in range(100)]
            self.targets += target


def main():
    generator = DataGenerator()
    # data normalization
    data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, generator.full_data)

    # Initialization and training
    som = MiniSom(10, 10, 8, sigma=3, learning_rate=0.5, neighborhood_function='gaussian')
    som.random_weights_init(data)
    print("Training...")
    som.train_random(data, 500)  # random training
    print("\n...ready!")

    plt.figure(figsize=(7, 7))
    # Plotting the response for each pattern in the iris dataset
    plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
    #plt.colorbar()

    # use different colors and markers for each label
    markers = ['o', 's', 'D', 'o']
    colors = ['C0', 'C1', 'C2', 'C3']
    t = generator.targets
    for cnt, xx in enumerate(data):
        w = som.winner(xx)  # getting the winner
        # palce a marker on the winning position for the sample xx
        plt.plot(w[0]+.5, w[1]+.5, markers[t[cnt]], markerfacecolor='None',
                markeredgecolor=colors[t[cnt]], markersize=12, markeredgewidth=2)
    plt.axis([0, 10, 0, 10])
    plt.show()

    
if __name__ == "__main__":
    main()
