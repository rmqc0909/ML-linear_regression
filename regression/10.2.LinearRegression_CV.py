#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso,Ridge
from sklearn.model_selection import GridSearchCV


if __name__ == "__main__":
    path = 'F:\code\ml_algorithm_practice\\regression\\10.Advertising.csv'
    # pandas读入
    data = pd.read_csv(path)    # TV、Radio、Newspaper、Sales
    x = data[['TV', 'Radio', 'Newspaper']]
    # x = data[['TV', 'Radio']]
    y = data['Sales']


    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

    # 对最小普通二乘增加罚项，控制theta系数大小，防止其过大
    model = Lasso()     # L1-norm
    # model = Ridge()   # L2-norm
    alpha_can = np.logspace(-3, 2, 10)
    print(alpha_can)
    lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)    # cv表示是几折（将原始数据分为几份）交叉验证，该例中为5折交叉验证
    lasso_model.fit(x_train, y_train)
    print('超参数：\n', lasso_model.best_params_)

    y_hat = lasso_model.predict(np.array(x_test))
    print(lasso_model.score(x_test, y_test))    # 确定性系数,表示模型对现实数据的拟合程度，一定是介于0~1间的数。
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print(mse, rmse)

    t = np.arange(len(x_test))
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    plt.title(u'线性回归预测销量', fontsize=18)
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
