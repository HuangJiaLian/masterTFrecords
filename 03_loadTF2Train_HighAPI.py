'''
@Description: Train MNIST with tfrecords yielded from a Numpy Dataset
@Author: Jack Huang
@Date: 2019-05-21 17:07:37
@LastEditTime: 2019-05-21 20:02:55
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
number_of_tfrecord = 2
steps_per_epoch = 500*number_of_tfrecord

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


model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(784,)))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


history = model.fit_generator(
  mnist_generator(['./tfrecords/train_'+str(i)+'.tfrecords' for i in range(number_of_tfrecord)]),
  steps_per_epoch=steps_per_epoch,
  epochs=epochs,
  verbose=1,
  #validation_data=mnist_generator('./test.tfrecords'),
  #validation_steps=steps_per_epoch,
  workers = 0  # runs generator on the main thread
)

score = model.evaluate_generator(
  mnist_generator(['./tfrecords/test_'+str(i)+'.tfrecords' for i in range(number_of_tfrecord)]),
  steps=steps_per_epoch,
  workers = 0  # runs generator on the main thread
)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
