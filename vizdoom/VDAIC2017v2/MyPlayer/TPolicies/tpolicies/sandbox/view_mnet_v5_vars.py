from os import path
from collections import OrderedDict

import tensorflow as tf
from timitate.lib4.pb2all_converter import PB2AllConverter
import joblib

from tpolicies.net_zoo.mnet_v5.data import MNetV5Config
from tpolicies.net_zoo.mnet_v5.utils import _make_inputs_palceholders
from tpolicies.net_zoo.mnet_v5.utils import (
  _make_mnet_v5_compatible_params_from_v4)
from tpolicies.net_zoo.mnet_v5.mnet_v5 import mnet_v5
import tpolicies.tp_utils as tp_utils
import tpolicies.ops as tp_ops


var_names = [
  'gridnet_enc/small/conv1/weights',
  'gridnet_enc/small/conv1/weights',
  'gridnet_enc/small/conv1/biases',
  'gridnet_enc/small/conv2/weights',
  'gridnet_enc/small/conv2/biases',
  'gridnet_enc/small/conv3/weights',
  'gridnet_enc/small/conv3/biases',
  'gridnet_enc/small/conv4/weights',
  'gridnet_enc/small/conv4/biases',
  'gridnet_enc/small/conv5/weights',
  'gridnet_enc/small/conv5/biases',
  'fc_xs/weights',
  'fc_xs/biases'
]
sandbox_dir = '/Users/pengsun/code/sc2_rl/TPolicies/tpolicies/sandbox'
#model_name = 'vars/im1303e/IL-model_20200103121545.model'
#model_name = 'vars/im1303e/IL-model_20200103194940.model'
#model_name = 'vars/im1303e/IL-model_20200104031712.model'
#model_name = 'vars/im1303e/IL-model_20200104031712.model'
#model_name = 'vars/im1210/IL-model_20191210035840.model'
#model_name = 'vars/im1210/IL-model_20191210101428.model'
model_name = 'vars/im1210/IL-model_20191210162134.model'
use_v4_model = True
param_names_v4_order_path = path.join(sandbox_dir, 'vars_mnet_v5_in_v4_order')
param_names_v5_order_path = path.join(sandbox_dir, 'vars_mnet_v5')


# build the mnet_v5
mycfg = {
  'test': False,
  'rl': False,
  'use_lstm': True,
  'lstm_cell_type': 'lstm',
  'lstm_layer_norm': True,
  'lstm_dropout_rate': 0.0,
  'batch_size': 1,
  'rollout_len': 1,
  'lstm_duration': 1,
  'nlstm': 256,
  'hs_len': 256 * 2,
  'weight_decay': None,
  'use_base_mask': True,
}
pb2all = PB2AllConverter(input_map_size=(128, 128),
                         output_map_size=(128, 128),
                         dict_space=True)
ob_space = pb2all.space.spaces[0]
ac_space = pb2all.space.spaces[1]
nc = MNetV5Config(ob_space, ac_space, **mycfg)
inputs = _make_inputs_palceholders(nc)
out = mnet_v5(inputs, nc, scope='mnet_v5_consistency_test')

# the (uninitialized) v5 net params and the loading ops
params = tf.trainable_variables(scope='mnet_v5_consistency_test')
params_ph = [tf.placeholder(p.dtype, shape=p.get_shape()) for p in params]
param_assign_ops = [p.assign(new_p) for p, new_p in zip(params, params_ph)]

# load the params
model_path = path.join(sandbox_dir, model_name)
np_params = joblib.load(model_path).model
if use_v4_model:
  with open(param_names_v4_order_path, 'r') as f:
    param_names_v4_order = [line.strip() for line in f.readlines()]
  with open(param_names_v5_order_path, 'r') as f:
    param_names_v5_order = [line.strip() for line in f.readlines()]

  np_params = _make_mnet_v5_compatible_params_from_v4(np_params,
                                                      param_names_v4_order,
                                                      param_names_v5_order)
sess = tf.Session()
sess.run(param_assign_ops,
         feed_dict={ph: va for ph, va in zip(params_ph, np_params)})

# calculate the norm
fetches = OrderedDict()
for name in var_names:
  t = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '.*{}.*'.format(name))
  assert len(t) == 1
  t = t[0]
  t_size = tp_utils.get_size(t)
  tn = tf.norm(tp_ops.to_float32(t), ord=1) / t_size
  fetches[name] = tn
d = sess.run(fetches)

# print
for k, v in d.items():
  print('{}: {}'.format(k, v))
