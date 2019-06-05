'''
@Description: Make many small Tfrecord
@Author: Jack Huang
@Date: 2019-05-21 16:55:29
@LastEditTime: 2019-06-05 10:41:46
@LastEditors: Please set LastEditors
'''
# -*-coding:utf-8-*-
import os
import tensorflow as tf
import numpy as np


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
  rows = images.shape[1]
  cols = images.shape[2]
  
 
  filename = os.path.join('./', name + '.tfrecords')
  print('Writing', filename)
  print('Debug:',images[0].shape, type(images[0]))
  print('Debug:',labels[0],labels[0].shape, type(labels[0]))
  with tf.python_io.TFRecordWriter(filename) as writer:
    for index in range(num_examples):
      image_raw = images[index].tostring()
      example = tf.train.Example(features=tf.train.Features(feature={
          'height': _int64_feature(rows),
          'width': _int64_feature(cols),
          'label': _int64_feature(int(labels[index])),
          'image_raw': _bytes_feature(image_raw)}))
      writer.write(example.SerializeToString())


def main():
  # Get the data.
  x_train = np.load('./originalData/x_train.npy') # 数据
  y_train = np.load('./originalData/y_train.npy') # 数据对应的label  
  x_test = np.load('./originalData/x_test.npy')
  y_test = np.load('./originalData/y_test.npy')

  # 这里的2只是做一个演示
  # 当数据量过大的时候，应该将数据转换成很多个小的tfrecord
  number_of_tfrecord = 2
  print('Start converting ...')
  for i in range(number_of_tfrecord):
    # Convert to Examples and write the result to TFRecords.
    convert_to(x_train, y_train, 'tfrecords/train_' + str(i))
    convert_to(x_test, y_test, 'tfrecords/test_' + str(i))
  print('Done :-)')
main()


