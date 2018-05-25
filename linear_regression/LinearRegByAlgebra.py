#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/24 15:04
# @Author  : xiedan
# @File    : LinearRegByAlgebra.py
# 该方法主要练习简单线性回归，即只有一个feature，直接利用公式求线性回归的系数，详见：https://www.geeksforgeeks.org/linear-regression-python-implementation/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def estimate_coef(x, y):
    n = np.size(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    ss_xy = np.sum(x * y - n * x_mean * y_mean)
    ss_xx = np.sum(x * x - n * x_mean * x_mean)

    b_1 = ss_xy / ss_xx
    b_0 = y_mean - b_1 * x_mean

    return [b_0, b_1]


def plot_regression_line(x, y, b):
    plt.scatter(x, y, color="r", marker="o", s=20)

    y_pred = b[0] + b[1] * x
    plt.plot(x, y_pred, color="y")

    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()


def main():
    path = "F:\code\ml_algorithm_practice\linear_regression\data\\10.Advertising.csv"
    data = pd.read_csv(path)
    x = data['TV']
    y = data['Sales']
    b = estimate_coef(x, y)
    plot_regression_line(x, y, b)
    print("estimate coefficients:\n b[0]={} \n b[1]={}".format(b[0], b[1]))


if __name__ == "__main__":
    main()
