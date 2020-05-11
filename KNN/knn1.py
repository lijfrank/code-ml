#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   knn 
@Author    @Version
-------    --------
Frank      1.0
@time   : 2020/4/24 19:14:32
@Desciption : None
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
# print(iris)
df = pd.DataFrame(iris.data, columns=iris.feature_names)
# df 添加columns列- iris.target
df['class'] = iris.target
# print(df)
df['class'] = df['class'].map({0: iris.target_names[0], 1: iris.target_names[1], 2: iris.target_names[2]})
# print(df)

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=35,
                                                    stratify=iris.target)


class KNN(object):
    def __init__(self, x, y, k, fun_distance):
        self.y = y
        self.x = x
        self.k = k
        self.fun_distance = fun_distance

    def predict(self):
        y_pred = np.zeros((self.x.shape[0], 1), dtype=self.y.dtype)
        for i, x_ in enumerate(self.x):
            distances = self.fun_distance(x_, x_test)
            sort_index = np.argsort(distances)
            k_y = self.y[sort_index[:self.k]].ravel()
            y_pred[i] = np.argmax(np.bincount(k_y))
        return y_pred


def fun_dist2(X, x):
    dists = np.sqrt(np.sum((X - x) ** 2, axis=1))
    return dists


def fun_dist1(X, x):
    dists = np.sum(np.abs(X - x), axis=1)
    return dists


knn = KNN(x_test, y_test, 5, fun_dist2)
y_pred = knn.predict()
accuracy = accuracy_score(y_test, y_pred)
print("accuracy:", accuracy)

# 保存结果list
result_list = []
# 针对不同的参数
for p in [1, 2]:
    fun_distance = fun_dist1 if p == 1 else fun_dist2

    # 不同的n_neighbors
    for k in range(1, 10, 2):
        knn = KNN(x_test, y_test, k, fun_distance)
        y_pred = knn.predict()
        accuracy = accuracy_score(y_test, y_pred)
        result_list.append([k, 'l1_distance' if p == 1 else 'l2_distance', accuracy])

df = pd.DataFrame(result_list, columns=['n_neighbors', 'distance', "accracy"])
print(df)
