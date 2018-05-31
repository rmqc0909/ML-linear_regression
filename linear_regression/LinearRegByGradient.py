#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/24 15:03
# @Author  : xiedan
# @File    : LinearRegByGradient.py
# 该方法利用梯度下降方法来拟合简单线性回归的参数，
# 详见：https://medium.com/meta-design-ideas/linear-regression-by-using-gradient-descent-algorithm-your-first-step-towards-machine-learning-a9b9c0ec41b1


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def step_gradient(x, y, curr_m, curr_b, learning_rate):
    gradient_m = 0
    gradient_b = 0
    n = float(len(x))

    for i in range(0, len(x)):
        xi = x[i]
        yi = y[i]
        gradient_m += -(2 / n) * xi * (yi - curr_m * xi - curr_b)
        gradient_b += -(2 / n) * (yi - curr_m * xi - curr_b)
    new_m = curr_m - (learning_rate * gradient_m)
    new_b = curr_b - (learning_rate * gradient_b)
    return [new_m, new_b]


def gradient_run(x, y, init_m, init_b, learning_rate, n_times):
    m = init_m
    b = init_b
    for i in range(n_times):
        m, b = step_gradient(x, y, m, b, learning_rate)
        # print("第{0}次的 m = {1}, b = {2}".format(i + 1, m, b))
    return [m, b]


def main():
    path = "F:\code\ml_algorithm_practice\linear_regression\data\\10.Advertising.csv"
    data = pd.read_csv(path)  # 利用梯度下降算法求线性模型参数时，若数据基本不是线性走向，则该方法很难得到收敛结果。
    # x = data["TV"]
    # y = data["Sales"]
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # 该数据基本上为线性走向。
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
    init_m = 0
    init_b = 0
    learning_rate = 0.0001
    n_times = 1000
    [m, b] = gradient_run(x, y, init_m, init_b, learning_rate, n_times)
    print("final m = {0}, b = {1}".format(m, b))

    plt.scatter(x, y, color="r", marker="o", s=20)

    y_pred = b + m * x
    plt.plot(x, y_pred, color="y")

    plt.xlabel("x")
    plt.ylabel("y")

    plt.show()


if __name__ == "__main__":
    main()
