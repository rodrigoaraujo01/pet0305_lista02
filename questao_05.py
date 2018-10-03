#!/usr/local/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.naive_bayes import GaussianNB


class DataGenerator(object):
    def __init__(self):
        """
        Coluna 1: Não=0, Sim=1
        Coluna 2: Solteiro=0, Casado=1, Divorciado=2
        Coluna 3: Baixo=0, Médio=1, Alto=2
        """
        self.data = [
            [1, 0, 2],
            [0, 1, 1],
            [0, 0, 0],
            [1, 1, 2],
            [0, 2, 1],
            [0, 1, 0],
            [1, 2, 2],
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, 1],
        ]
        self.target = [
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            1,
        ]


def main():
    generator = DataGenerator()
    gnb = GaussianNB()
    y_pred = gnb.fit(generator.data, generator.target).predict([[0, 2, 1]])
    print(['Não', 'Sim'][y_pred[0]])


if __name__ == "__main__":
    main()
