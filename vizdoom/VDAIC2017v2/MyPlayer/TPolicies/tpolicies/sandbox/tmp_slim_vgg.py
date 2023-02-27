import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import vgg


inputs = tf.random_normal(shape=(4, 224, 224, 3), dtype=tf.float32)

with slim.arg_scope(vgg.vgg_arg_scope()):
  outputs, end_points = vgg.vgg_a(inputs)

print(outputs)
print(end_points)