# -*-coding:utf-8-*-
'''
@Description: 比较两种方法得到的每个batch是否都相同
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-08-04 18:20:52
@LastEditors: Jack Huang
@LastEditTime: 2019-08-04 18:38:13
'''

import numpy as np 
###############################################################
# 加载调试使用的mini-batch
# 数据分别由A_train_with_npy.py 和 B1_load_tfrec_to_train.py生成
###############################################################
npy_batch_list = np.load('npy_batch_list.npy')
tfr_batch_list = np.load('tfr_batch_list.npy')
# print(npy_batch_list[0])
# print(tfr_batch_list[0])

###################################
# 测试所有的数据是不是一样
###################################
# print(np.allclose(npy_batch_list[0],tfr_batch_list[0]))
# print(npy_batch_list==tfr_batch_list) # 因为是浮点数, 一般不会绝对相等
print(np.allclose(npy_batch_list,tfr_batch_list)) # 比较两个array是不是每一元素都相等，默认在1e-05的误差范围内

# 输出结果为True表示:
# 1. 使用两种方法得到的每一个mini-batch都是一样的
# 2. 生成和读取tfrecord的程序是对的
# 3. 数据处理是对的，问题在其他地方

# 输出结果为False表示:
# 1. TFrecord数据处理有问题，需要修改，直到得到True