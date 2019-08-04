# -*-coding:utf-8-*-
'''
@Description: 方法B:步骤1：加载上一步生成的tfrecord, 训练同一个回归模型
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-08-04 17:31:04
@LastEditors: Jack Huang
@LastEditTime: 2019-08-04 18:42:57
'''

import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt 

batch_size = 10
epochs = 2


###########################
# TF_records
###########################
# Return a TF dataset for specified filename(s)
def mnist_dataset(filenames):

  def decode_example(example_proto):

    features = tf.parse_single_example(
      example_proto,
      features = {
        'data_x': tf.FixedLenFeature([], tf.float32),
        'data_y': tf.FixedLenFeature([], tf.float32)
      }
    )

    image = features['data_x']
    label = features['data_y']

    return [image, label]

  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(decode_example)
  # dataset = dataset.repeat()
  # dataset = dataset.shuffle(10000)
  dataset = dataset.batch(batch_size)
  return dataset


# Keras generator that yields batches from the speicfied tfrecord filename(s)
def mnist_generator(filenames):
  dataset = mnist_dataset(filenames)
  iter = dataset.make_one_shot_iterator()
  batch = iter.get_next()

  while True:
    yield K.batch_get_value(batch)


number_of_tfrecord = 1


###########################
# NN 
###########################
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

xs = tf.placeholder(tf.float32, [None, 1]) # * rows, 1 col
ys = tf.placeholder(tf.float32, [None, 1]) # * rows, 1 col

# define hidden layer and output layer
l1 = add_layer(xs, 1, 10, actication_function = tf.nn.relu)
prediction = add_layer(l1, 10, 1, actication_function = None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), 
				reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()




#################################
# Load tf_records data to train
#################################
sess = tf.Session()
sess.run(init)

demo = np.load('Training_data.npy')
Train_images = demo[:,0].reshape([-1,1])
Train_labels = demo[:,1].reshape([-1,1])
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(Train_images,Train_labels)
plt.ion()
plt.show()

# Debug use:
tfr_batch_list = []

# 使用迭代的方式得到数据，这样就不会出现内存不够的情况
for i in range(200):
    print('Epoch {} trainning ...'.format(i+1))
    # 每个Epoch重新生成一个iterator, 这样就不需要前面的 dataset.repeat()了
    # 理论上不需要shuffle也是可以的,在测试阶段数据不shuffle,检测每一个batch里面的
    # 所有元素是不是都一样，必须要保证都一样。
    iterator =  mnist_generator(['./tfrecords/train_'+str(k)+'.tfrecords' 
                                for k in range(number_of_tfrecord)])
    for j in range(30):
        batch_xs_, batch_ys_ = next(iterator)
        batch_xs = batch_xs_.reshape(-1,1)
        batch_ys = batch_ys_.reshape(-1,1)
        sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})

        # Debug Use:
        training_batch = np.concatenate((batch_xs,batch_ys),axis=1)
        tfr_batch_list.append(training_batch)
        print(training_batch)
        
    if i % 10 == 0:
		print('Loss = {}'.format(sess.run(loss,feed_dict={xs:Train_images, ys:Train_labels})))
		try:
			ax.lines.remove(lines[0])
		except Exception :
			pass
		prediction_value = sess.run(prediction, feed_dict={xs:Train_images})
		lines = ax.plot(Train_images, prediction_value, 'r-', lw = 4)
		plt.pause(0.1)

# Debuge use:
print('Saving batches to debug ...')
tfr_batch_list = np.array(tfr_batch_list)
np.save('tfr_batch_list.npy', tfr_batch_list)
print('Done.')