import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np 

# #read data
# dataframe = pd.read_fwf('brain_body.txt')
# x_values = dataframe[['Brain']]
# y_values = dataframe[['Body']]

# #train model on data
# body_reg = linear_model.LinearRegression()
# body_reg.fit(x_values, y_values)

# #visualize results
# plt.scatter(x_values, y_values)
# plt.plot(x_values, body_reg.predict(x_values))
# plt.show()

points = np.genfromtxt("challenge_dataset.txt", delimiter=",")
x = points[:, 0].reshape(-1, 1)

# numpy中reshape函数的三种常见相关用法,,,,-1表示不知道分成多少行，但是我知道分成多少列
# reshape(1,-1)转化成1行：
# reshape(2,-1)转换成两行：
# reshape(-1,1)转换成1列：
# reshape(-1,2)转化成两列

print(x)
print(points[:,0])
y = points[:,1]
print(y)
# print(points)
# print(x)
challenge_reg = linear_model.LinearRegression()
challenge_reg.fit(x, y)
plt.scatter(x, y)
plt.plot(x, challenge_reg.predict(x))
plt.show()
