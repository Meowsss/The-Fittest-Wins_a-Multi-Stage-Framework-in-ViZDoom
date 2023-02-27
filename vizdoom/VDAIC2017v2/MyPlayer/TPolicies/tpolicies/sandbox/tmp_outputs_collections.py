import tensorflow.contrib.layers as layers
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils as lutils

inputs = tf.random_normal(shape=(4, 32, 32, 3), dtype=tf.float32)
oc = 'xxx_out'
with tf.variable_scope('xxx/yyy', reuse=tf.AUTO_REUSE):
  y1 = layers.conv2d(inputs, 32, [3, 3], outputs_collections=oc)

with tf.variable_scope('xxx/yyy', reuse=tf.AUTO_REUSE):
  y2 = layers.conv2d(inputs, 32, [3, 3], outputs_collections=oc)

print(tf.global_variables())
print(tf.get_collection(oc, scope='xxx'))
print(lutils.convert_collection_to_dict(oc))
pass