#!/usr/local/bin/python
# -*- coding: utf-8 -*-


import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DataGenerator(object):
    def __init__(self):
        self.micro_set_length = 100
        self.mu = [
            np.array([-1, 5]),
            np.array([1, 5]),
        ]
        self.cov = 0.25*np.identity(2)
        data = []
        for mu in self.mu:
            tmp_data = mvn.rvs(mean=mu, cov=self.cov, size=self.micro_set_length)
            data.append(tmp_data)
        self.full_data = np.vstack(data)
        self.targets = []
        for i in range(2):
            target = [[i] for j in range(self.micro_set_length)]
            self.targets += target
        self.targets = np.array(self.targets)
        self.all_data = np.hstack([self.full_data, self.targets])

    def plot(self):
        fig = plt.figure()
        pi = np.pi
        p = 2
        mu = np.array([-1, 5])
        C = np.matrix([[0.25,0],[0, 0.25]])
        factor_1 = 1/(2*pi)**(p/2)
        factor_2 = 1/0.0625**(-1/2)
        y_func = lambda x, y: factor_1*factor_2*np.exp(-1/2 * ((np.array([x,y])-mu).T @ C.I @ (np.array([x,y])-mu)))
        ax = Axes3D(fig)
        x = np.linspace(-10, 10, 100)
        y = np.linspace(-10, 10, 100)
        xx, yy = np.meshgrid(x, y)
        y_func_vec = np.vectorize(y_func)
        z = y_func_vec(xx, yy)
        ax.plot_wireframe(xx, yy, z, colors=['r'], linewidths=[1], label='Função original')
        plt.show()


class GaussianClassifier(object):
    def __init__(self, data):
        np.random.shuffle(data)
        self.data = data
        data_split = 0.3
        limit = int(data_split * len(self.data))
        self.train_data = self.data[:limit]
        self.test_data = self.data[limit:]
        self.predicted_data = []
        self.means = []
        self.covs = []
        self.calculate_means_covariance()
        self.predict()
        self.plot_data()

    def calculate_means_covariance(self):
        for i in range(2):
            # Get all lines whose 2nd column equals i
            filtered_data = self.train_data[i == self.train_data[:,2]]
            self.means.append([filtered_data[:,0].mean(), filtered_data[:,1].mean()])
            self.covs.append(np.cov(filtered_data[:,:2].T))

    @staticmethod
    def calculate_probability(x, mean, cov):
        p = 2
        factor_1 = 1/(2*np.pi)**(p/2)
        factor_2 = 1/0.0625**(-1/2)
        cov_inv = np.linalg.inv(cov)
        factor_3 = np.exp(-1/2 * ((x-mean).T @ cov_inv @ (x-mean)))
        return factor_1 * factor_2 * factor_3

    def predict(self):
        results = []
        for x, y, c in self.test_data:
            data = np.array([x, y])
            probs = [self.calculate_probability(data, self.means[i], self.covs[i]) for i in range(2)]
            pred_c = probs.index(max(probs))
            results.append([pred_c, c])
            self.predicted_data.append([x, y, pred_c])
        self.predicted_data = np.array(self.predicted_data)
        right = wrong = 0
        for pred, c in results:
            if pred == c:
                right += 1
            else:
                wrong += 1
        print(right/(right+wrong))

    def plot_data(self):
        fig = plt.figure()
        ax11 = fig.add_subplot(221)
        ax11.set_title('Dados originais')
        ax12 = fig.add_subplot(222)
        ax12.set_title('Dados de treinamento')
        ax21 = fig.add_subplot(223)
        ax21.set_title('Dados de teste')
        ax22 = fig.add_subplot(224)
        ax22.set_title('Dados de teste vs predições')
        for i in range(2):
            filtered_data = self.data[i == self.data[:,2]]
            ax11.scatter(filtered_data[:,0], filtered_data[:,1])
        for i in range(2):
            filtered_data = self.train_data[i == self.train_data[:,2]]
            ax12.scatter(filtered_data[:,0], filtered_data[:,1])
        for i in range(2):
            filtered_data = self.test_data[i == self.test_data[:,2]]
            ax21.scatter(filtered_data[:,0], filtered_data[:,1])
        for i in range(2):
            filtered_data = self.test_data[i == self.test_data[:,2]]
            ax22.scatter(filtered_data[:,0], filtered_data[:,1])
        for i in range(2):
            filtered_data = self.predicted_data[i == self.predicted_data[:,2]]
            ax22.scatter(filtered_data[:,0], filtered_data[:,1], marker='x')
        plt.show()




def main():
    generator = DataGenerator()
    # generator.calculate_means_covariance()    
    # generator.plot()
    clf = GaussianClassifier(generator.all_data)
    # generator.plot_data()

if __name__ == "__main__":
    main()



# References
# https://www.antoniomallia.it/lets-implement-a-gaussian-naive-bayes-classifier-in-python.html