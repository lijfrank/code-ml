#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   knn2 
@Author    @Version
-------    --------
Frank      1.0
@time   : 2020/4/24 21:18:03
@Desciption : None
"""
# !/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: JYRoooy

from sklearn import neighbors
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
# print(iris)

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=123, stratify=iris.target)
knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data, iris.target)                 # 用KNN的分类器进行建模，这里利用的默认的参数，大家可以自行查阅文档
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("accuracy_score:", accuracy)