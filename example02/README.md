# 保存加载tfrecords的另一个例子
## 文件结构
├── 01_make_one_tf_record_cool.py

├── 02_load_one_tf_record_cool.py

├── data_source

│   ├── 11cut.npy

│   └── 11train_config.npy

├── README.md

└── tfrecords

​    └── training_data01.tfrecords

## 说明
依次执行:
01_make_one_tf_record_cool.py (生成tfrecord)
02_load_one_tf_record_cool.py (加载tfrecord)

运行两个程序都会打印出前100项训练数据，对比两组数据一样就说明这一组程序是正确的。

## 注意:
在实际训练网络的时候要记得:
- 修改batch_size
- uncomment `dataset = dataset.shuffle(10000)`
