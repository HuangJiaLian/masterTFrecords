'''
@Description: 生成训练数据
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-08-04 15:17:13
@LastEditors: Jack Huang
@LastEditTime: 2019-08-04 18:43:35
'''
import numpy as np

x_data = np.linspace(-1,1,300)[:, np.newaxis] 
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

print(x_data)
print(y_data)

training_data = np.concatenate((x_data,y_data),axis=1)
print(training_data)

np.save('./Training_data.npy',training_data)
