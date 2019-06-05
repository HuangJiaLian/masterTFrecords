'''
@Description: Example of making one tfrecords.
@Author: Jack Huang
@Date: 2019-06-05 09:30:24
@LastEditTime: 2019-06-05 13:24:12
@LastEditors: Please set LastEditors
'''
import tensorflow as tf
import numpy as np
import os
import time
# Load a small group of trainning data
# Load data and label 
data_x = np.load('./data_source/11train_config.npy')
data_y = np.load('./data_source/11cut.npy')

# Learn more about the data 
print(data_x.shape)
print(data_y.shape)
print(data_x[0])
print(data_y[0])
print(data_x[0].shape)
print(data_y[0].shape)
print(type(data_x[0][0]))
print(type(data_y[0][0])) # The type of lable element is no need to be float64. int8 is enough.

# Save as tfrecords
def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# For list feature
def _int64_feature_list(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature_list(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def convert_to(images,labels, name):
  """Converts a dataset to tfrecords."""
  if images.shape[0] != labels.shape[0]:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], labels.shape[0]))
  
  num_examples = images.shape[0]
  print('Debug:',images[0].shape, type(images[0]))
  print('Debug:',labels[0].shape, type(labels[0]))
  
  filename = os.path.join('./', name + '.tfrecords')
  print('Writing', filename)
  print('Start converting ...')
  with tf.python_io.TFRecordWriter(filename) as writer:
    for index in range(num_examples):
      if index < 100:
        print(images[index],labels[index])
      example = tf.train.Example(features=tf.train.Features(feature={
          'data_x': _float_feature_list(images[index]),
          'data_y': _int64_feature_list(labels[index].astype(int))
          }))
      writer.write(example.SerializeToString())
  print('Completed.')

convert_to(data_x, data_y, 'tfrecords/training_data01')
