# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: PyCharm (code-ml)
#     language: python
#     name: pycharm-88a21ce0
# ---

from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# + pycharm={"name": "#%%\n"}
wine = load_wine()
wine # 字典的格式
wine.data

# + pycharm={"name": "#%%\n"}
wine.target

# + pycharm={"name": "#%%\n"}
wine.data.shape

# + pycharm={"name": "#%%\n"}
pd.concat([pd.DataFrame(wine.data),pd.DataFrame(wine.target)],axis=1)

# + pycharm={"name": "#%%\n"}
wine.feature_names

# + pycharm={"name": "#%%\n"}
wine.target_names

# + pycharm={"name": "#%%\n"}
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)

# + pycharm={"name": "#%%\n"}
X_train.shape

# + pycharm={"name": "#%%\n"}
clf = DecisionTreeClassifier(criterion="entropy")


# + pycharm={"name": "#%%\n"}
clf = clf.fit(X_train,y_train)
score = clf.score(X_test, y_test)
score


# + pycharm={"name": "#%%\n"}
import graphviz
from sklearn import tree
dot_data = tree.export_graphviz(clf,
                                feature_names=wine.feature_names,
                                class_names=wine.target_names,
                                filled=True,
                                rounded=True)
graph = graphviz.Source(dot_data)
graph
