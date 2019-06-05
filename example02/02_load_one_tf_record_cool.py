'''
@Description: Example of loading a group of tfrecords.
@Author: Jack Huang
@Date: 2019-06-05 09:34:38
@LastEditTime: 2019-06-05 13:33:48
@LastEditors: Please set LastEditors
'''

# Load data 
# -*-coding:utf-8-*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
import tensorflow.keras.backend as K

# batch_size = 1 just an example intend to check load data 
batch_size = 1

# Return a TF dataset for specified filename(s)
def mnist_dataset(filenames):

  def decode_example(example_proto):

    features = tf.parse_single_example(
      example_proto,
      features = {
        'data_x': tf.FixedLenFeature([16000], tf.float32),
        'data_y': tf.FixedLenFeature([2], tf.int64)
      }
    )

    image = tf.cast(features['data_x'], tf.float32)
    label = tf.cast(features['data_y'], tf.uint8)


    return [image, label]

  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(decode_example)
  dataset = dataset.repeat()
  # Comment the following line to check load data if you want
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



# iterator =  mnist_generator(['../makeMultiTFrecords/tfrecords/train_'+str(i)+'.tfrecords' 
#                             for i in range(number_of_tfrecord)])
iterator =  mnist_generator(['./tfrecords/training_data01.tfrecords'])


# 使用迭代的方式得到数据，这样就不会出现内存不够的情况
for i in range(100):
	batch_xs, batch_ys = next(iterator)
	# print(batch_xs.shape,batch_ys.shape)
  # Check restored data
	print(batch_xs[0],batch_ys[0])

