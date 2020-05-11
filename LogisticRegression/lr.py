#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   lr 
@Author    @Version
-------    --------
Frank      1.0
@time   : 2020/5/1 15:50:09
@Desciption : None
"""
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # 在线性模型里面

iris = load_iris()
