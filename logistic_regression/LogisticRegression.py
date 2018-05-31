#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/31 14:40
# @Author  : xiedan
# @File    : LogisticRegression.py


import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=100000, theta=None):
        self.lr = lr
        self.n_iters = n_iters
        self.theta = theta

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __loss(self, y, h):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def predict(self, X, theta):
        return self.__sigmoid(np.dot(X, theta)) >= 0.5

    def fit(self, X, y):
        for i in range(self.n_iters):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, h - y) / y.size
            self.theta -= gradient * self.lr
            if i % 10000 == 0:  # 迭代10000次时打印一次loss
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(self.__loss(y, h))
