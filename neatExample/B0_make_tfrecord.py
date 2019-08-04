# -*- coding: utf-8 -*-  
'''
@Description: 方法B:步骤0:将npy格式的训练数据变成tfrecord格式
@Author: Jack Huang
@Github: https://github.com/HuangJiaLian
@Date: 2019-08-04 16:06:23
@LastEditors: Jack Huang
@LastEditTime: 2019-08-04 19:04:06
'''

import tensorflow as tf
import numpy as np
import os
import time

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def convert_to(images,labels, name):
  """Converts a dataset to tfrecords."""

  if images.shape[0] != labels.shape[0]:
    raise ValueError('Images size %d does not match label size %d.' %
                     (images.shape[0], labels.shape[0]))
  
  num_examples = images.shape[0]
  rows = images.shape[0]
  cols = images.shape[1]
  

  filename = os.path.join('./', name + '.tfrecords')
  print('Writing', filename)
  print('Debug:',images[0].shape, type(images[0]))
  print('Debug:',labels[0],labels[0].shape, type(labels[0]))
  
  with tf.python_io.TFRecordWriter(filename) as writer:
    for index in range(num_examples):
      example = tf.train.Example(features=tf.train.Features(feature={
          'data_y': _float_feature(labels[index]),
          'data_x': _float_feature(images[index])}))
      writer.write(example.SerializeToString())


def main():
  # Get the data.
  training_data = np.load('./Training_data.npy')
  x_train = training_data[:,0].reshape([-1,1]) # 数据
  y_train = training_data[:,1].reshape([-1,1]) # 数据对应的label  

  # 这里的2只是做一个演示
  # 当数据量过大的时候，应该将数据转换成很多个小的tfrecord
  number_of_tfrecord = 1
  print('Start converting ...')
  for i in range(number_of_tfrecord):
    # Convert to Examples and write the result to TFRecords.
    convert_to(x_train, y_train, 'tfrecords/train_' + str(i))
  print('Done :-)')

main()
