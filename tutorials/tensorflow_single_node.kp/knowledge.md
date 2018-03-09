---
title: Tensorflow tutorial - MNIST For ML Beginners
authors:
- Danny Deep
- Lenny Learner
tags:
- knowledge
- example
- deeplearning
- tensorflow
created_at: 2016-06-29 00:00:00
updated_at: 2018-03-09 14:19:34.692974
tldr: 'This notebook demonstrates how to use TensorFlow on the Spark driver node to
  fit a neural network on MNIST handwritten digit recognition data.


  Prerequisites:

  * A GPU-enabled cluster on Databricks.

  * TensorFlow installed with GPU support.


  The content of this notebook is [copied from TensorFlow project](https://www.tensorflow.org/versions/r0.11/tutorials/index.html)
  under [Apache 2.0 license](https://github.com/tensorflow/tensorflow/blob/master/LICENSE)
  with slight modification to run on Databricks. Thanks to the developers of TensorFlow
  for this example!

  '
---
```python
import tensorflow as tf
```

```python
# Some of this code is licensed by Google under the Apache 2.0 License

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
```

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
```
Load the data (this step may take a while)


```python
mnist = input_data.read_data_sets('/tmp/data', one_hot=True)
```
Define the model


```python
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b
```
Define loss and optimizer


```python
y_ = tf.placeholder(tf.float32, [None, 10])
```
The raw formulation of cross-entropy,

```tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)), reduction_indices=[1]))```

can be numerically unstable.

So here we use `tf.nn.softmax_cross_entropy_with_logits` on the raw
outputs of 'y', and then average across the batch.


```python
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
```

```python
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
summary = tf.summary.scalar("accuracy", accuracy)
```
Start TensorBoard so we can monitor training progress.


```python
log_dir = "/tmp/tensorflow_log_dir"
dbutils.tensorboard.start(log_dir)
```
Train our model using small batches of data.


```python
sess = tf.InteractiveSession()

# Make sure to use the same log directory for both start TensorBoard in your training.
summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)

tf.global_variables_initializer().run()
for batch in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  _, batch_summary = sess.run([train_step, summary], feed_dict={x: batch_xs, y_: batch_ys})
  summary_writer.add_summary(batch_summary, batch)
```
Test the trained model. The final accuracy is reported at the bottom. You can compare it with the accuracy reported by the other frameworks!


```python
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))
```
Tensorboard will stay active after your training is finished so you can view a summary of the process, even if you detach your notebook. Use `dbutils.tensorboard.stop()` from this or any notebook to stop TensorBoard.


```python
dbutils.tensorboard.stop()
```
If you place your log directory in `/tmp/` it will be deleted when your cluster shutsdown. If you'd like to save your training logs you can copy them to a permenant location, for example somewhere on the Databricks file system.


```python
import shutil
shutil.move(log_dir, "/dbfs/tensorflow/logs")
```
