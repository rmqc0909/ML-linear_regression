#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    path = '/Users/xiedan/PycharmProjects/ml_algorithm_practice/regression/10.Advertising.csv'

    # numpy读入
    # p = np.loadtxt(path, delimiter=',', skiprows=1)
    # print(p)
    # print('\n\n===============\n\n')

    # # pandas读入
    data = pd.read_csv(path)    # TV、Radio、Newspaper、Sales
    x = data[['TV', 'Radio', 'Newspaper']]
    # x = data[['TV', 'Radio']]
    y = data['Sales']

    #
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    #
    # # 绘制1
    # plt.plot(data['TV'], y, 'ro', label='TV')
    # plt.plot(data['Radio'], y, 'g^', label='Radio')
    # plt.plot(data['Newspaper'], y, 'ms', label='Newspaer')
    # plt.legend(loc='lower right')
    # plt.grid()
    # plt.show()
    # #
    # # 绘制2
    plt.figure(figsize=(9,12))
    plt.subplot(311)    #“311”表示 3行1列 第2个
    plt.plot(data['TV'], y, 'ro')
    plt.title('TV')
    plt.grid()
    plt.subplot(312)    #“311”表示 3行1列 第2个
    plt.plot(data['Radio'], y, 'g^')
    plt.title('Radio')
    plt.grid()
    plt.subplot(313)    #“311”表示 3行1列 第3个
    plt.plot(data['Newspaper'], y, 'b*')
    plt.title('Newspaper')
    plt.grid()
    plt.tight_layout()
    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)
    print(x_train, y_train)

    linreg = LinearRegression()
    model = linreg.fit(x_train, y_train)
    print(model)
    print(linreg.coef_)     # 斜率
    print(linreg.intercept_)    # 截距

    print("x_test is \n", np.array(x_test))
    y_hat = linreg.predict(np.array(x_test))
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print(mse, rmse)

    # 绘制预测数据和真实数据销量图（60个样本）
    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.legend(loc='upper right')
    plt.title(u'线性回归预测销量', fontsize=16)
    plt.grid()
    plt.show()
