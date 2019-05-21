# Master TFrecords

目的: 解决不能够加载超大数据进行训练的问题。

主要思路: 将大数据转换成多个`tfrecord`数据，然后使用迭代器加载数据进行训练。

实例程序说明:

`01_makeMultiTfrecords.py`:  使用`mnist`作为演示，每一个原始的`mnist`数据集就是一个小的`tfrecord`数据，通过重复很多次，用来模拟大量的数据。最后的效果等效于将超大的数据分解成很多晓得`tfrecord`, 保存在`tfrercords`里面。

`02_loadTFrecords.py`: 加载多个`tfrecords`, 使用迭代批量提取数据。

`03_loadTF2Train_HighAPI.py`: 还原`tfrecords`数据后, 使用Tensorflow高级API训练神经。Tensorflow高级API相比之前版本要简洁许多。

`04_loadTF2Train_OldVersion.py`: 还原`tfrecords`数据后，使用旧版本的Tensorflow写法训练神经网络。

