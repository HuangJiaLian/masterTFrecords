'''
Load MNIST with a list of  tfrecords
'''
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
steps_per_epoch = 500*116

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



number_of_tfrecord = 2
iterator =  mnist_generator(['../makeMultiTFrecords/tfrecords/train_'+str(i)+'.tfrecords' 
                            for i in range(number_of_tfrecord)])

# 使用迭代的方式得到数据，这样就不会出现内存不够的情况
for i in range(100):
	batch_xs, batch_ys = next(iterator)
	print(batch_xs.shape,batch_ys.shape)