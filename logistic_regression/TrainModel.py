#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/31 15:10
# @Author  : xiedan
# @File    : TrainModel.py
# 该方法利用梯度下降法求logistic回归模型的参数，并预测出模型准确率

from LogisticRegression import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = (iris.target != 0) * 1
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    theta = np.zeros(X.shape[1])  # theta初始化为0
    model = LogisticRegression(lr=0.1, n_iters=300000, theta=theta)
    model.fit(x_train, y_train)
    preds = model.predict(x_test, model.theta)
    accu = (preds == y_test).mean()
    print(accu)


