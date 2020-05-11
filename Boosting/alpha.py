#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   alpha 
@Author    @Version
-------    --------
Frank      1.0
@time   : 2020/5/4 17:28:42
@Desciption : None
"""
import matplotlib.pyplot as plt
import numpy as np
import math

# print(e)
x = np.arange(0.0, 1.0, 0.01)
# print(x)
y = []
for i in x:
    # print(i)
    y_i = 0.5 * math.log((1.0 - i) / i, math.e)
    # print(y_i)
    y.append(y_i)
# print(y)
plt.title(r'$\alpha$')
plt.plot(x, y)
plt.show()