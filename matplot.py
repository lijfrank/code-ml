#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   matplot 
@Author    @Version
-------    --------
Frank      1.0
@time   : 2020/4/29 19:44:19
@Desciption : None
"""
import matplotlib.pyplot as plt
import numpy as np

# A:为信号幅值
A = 1
B = 2
# fi:为信号频率
fi = 0.4
# time_s:为时间长度（s）
time_s = 7
# sample:为信号采样频率
sample = 0.5

x_data = ['t1', 't2', 't3', 't4', 't5', 't6', 't7']
y_data = [1,0,1,2,1,0,1]
y_data2 = [1+2,2+2,1+2,0+2,1+2,2+2,1+2]


# plt.xlabel(r"Self-connection Coefficent $\alpha$")
# plt.ylabel("Macro-F1")

ln1, = plt.plot(x_data, y_data, color='red', linewidth=1.0, marker="o")
ln2, = plt.plot(x_data, y_data2, color='blue', linewidth=1.0, marker="o")

# y_ticks =[i/10 for i in range(0, 11, 1)]
# plt.yticks(y_ticks)

# # 设置数字标签
# for x, y in zip(x_data, y_data):
#     if y == 0.844:
#         plt.text(x, y, y, ha='center', va='bottom', fontsize=10)
#         break
# for x, y2 in zip(x_data, y_data2):
#     if y2 == 0.887:
#         plt.text(x, y2, y2, ha='center', va='bottom', fontsize=10)
# plt.legend(handles=[ln1, ln2], labels=['Unweighted', 'Weighted'])

# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
plt.show()

