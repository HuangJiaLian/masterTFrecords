'''
@Description: Train MNIST with a list of  tfrecords
@Author: Jack Huang
@Date: 2019-05-21 17:10:53
@LastEditTime: 2019-05-21 20:02:12
@LastEditors: Please set LastEditors
'''
# -*-coding:utf-8-*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K


batch_size = 128
num_classes = 10
epochs = 2
steps_per_epoch = 500*2

# Return a TF dataset for specified filename(s)
def mnist_dataset(filenames):

  def decode_example(example_proto):

    features = tf.parse_single_example(
      example_proto,
      features = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
      }
    )

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    # You can do some data processing here
    image = tf.cast(image, tf.float32) / 255.
    label = tf.one_hot(tf.cast(features['label'], tf.int32), num_classes)


    return [image, label]

  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(decode_example)
  dataset = dataset.repeat()
  dataset = dataset.shuffle(10000)
  dataset = dataset.batch(batch_size)
  return dataset


# Keras generator that yields batches from the speicfied tfrecord filename(s)
def mnist_generator(filenames):

  dataset = mnist_dataset(filenames)
  iter = dataset.make_one_shot_iterator()
  batch = iter.get_next()

  while True:
    yield K.batch_get_value(batch)


# -------------------------------------
# Neural Network Stuff
# -------------------------------------
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W):
	# stride [1, x_movement,y_movement,1]
	# Must have strides[0] = strides[3]=1
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	# Must have strides[0] = strides[3]=1
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
# 255 ???
xs = tf.placeholder(tf.float32, [None, 784])/255  # Not define how many samples, but every sample
									        # have 784(28x28) nodes
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs, [-1,28,28,1])
# print(x_image.shape) #[n_samples,28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5,5,1,32]) # patch/kernel 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1) 					     # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5,5,32,64]) # patch/kernel 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2) 					     # output size 7x7x64


## func1 layer ##
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7, 7, 64]
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## func2 layer ##
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

number_of_tfrecord = 116
iterator =  mnist_generator(['../makeMultiTFrecords/tfrecords/train_'+str(i)+'.tfrecords' for i in range(number_of_tfrecord)])

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
	batch_xs, batch_ys = next(iterator)
	sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
	print('The ' + str(i) + 'th batch training ...')