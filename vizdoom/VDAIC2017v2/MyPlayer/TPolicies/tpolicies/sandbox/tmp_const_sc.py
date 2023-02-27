import tensorflow as tf


with tf.variable_scope('xxx'):
  a = tf.constant([[1,2], [3,4]])


for v in tf.global_variables():
  print(v)
