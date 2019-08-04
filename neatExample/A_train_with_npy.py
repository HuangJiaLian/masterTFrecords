# -*-coding:utf-8-*-
'''
@Description: 方法A: 直接使用npy文件，训练一个回归模型
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-08-04 17:12:30
@LastEditors: Jack Huang
@LastEditTime: 2019-08-04 19:09:21
'''
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import time 

def add_layer(inputs, in_size, out_size, actication_function = None):
	Weights = tf.Variable(tf.random_normal([in_size,out_size]))
	biases = tf.Variable(tf.zeros([1,out_size]) + 0.1) # Because the recommend initial
												# value of biases != 0; so add 0.1
	Wx_plus_b = tf.matmul(inputs, Weights) + biases
	
	if actication_function is None:
		outputs = Wx_plus_b
	else:
		outputs = actication_function(Wx_plus_b)
	return outputs


trainning_data = np.load('Training_data.npy')
Train_images = trainning_data[:,0].reshape([-1,1])
Train_labels = trainning_data[:,1].reshape([-1,1])
print(trainning_data)
print(Train_images)
print(Train_labels)
batch_size = 10

xs = tf.placeholder(tf.float32, [None, 1]) # * rows, 1 col
ys = tf.placeholder(tf.float32, [None, 1]) # * rows, 1 col


# define hidden layer and output layer
l1 = add_layer(xs, 1, 20, actication_function = tf.nn.relu)
prediction = add_layer(l1, 20, 1, actication_function = None)


loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), 
				reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(Train_images,Train_labels)
plt.ion()
plt.show()

# Debug use:
npy_batch_list = []

for i in range (200):
	print('Epoch {} trainning ...'.format(i+1))
	batch_x = [Train_images[k:k+batch_size] for k in range(0,len(Train_images), batch_size)]
	batch_y = [Train_labels[k:k+batch_size] for k in range(0,len(Train_labels), batch_size)]
	for j in range(len(batch_x)):
		sess.run(train_step, feed_dict={xs:batch_x[j], ys:batch_y[j]})
		# Debug use:
		training_batch = np.concatenate((batch_x[j], batch_y[j]), axis=1)
		npy_batch_list.append(training_batch)
		print(training_batch)

	if i % 10 == 0:
		print(sess.run(loss,feed_dict={xs:Train_images, ys:Train_labels}))
		try:
			ax.lines.remove(lines[0])
		except Exception :
			pass
		prediction_value = sess.run(prediction, feed_dict={xs:Train_images})
		lines = ax.plot(Train_images, prediction_value, 'r-', lw = 4)
		plt.pause(0.1)

# Debuge use:
print('Saving batches to debug ...')
npy_batch_list = np.array(npy_batch_list)
np.save('npy_batch_list.npy', npy_batch_list)
print('Done.')