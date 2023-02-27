""""view norm for mnet_v6d6 Neural Net parameters (vars)"""
import os
from os import path
from collections import OrderedDict

import tensorflow as tf
from timitate.lib6.pb2all_converter import PB2AllConverter
import joblib

from tpolicies.net_zoo.mnet_v6.mnet_v6d6 import net_config_cls
from tpolicies.net_zoo.mnet_v6.mnet_v6d6 import net_inputs_placeholders_fun
from tpolicies.net_zoo.mnet_v6.mnet_v6d6 import net_build_fun


policy_config = {
  'use_xla': False,
  'test': True,
  'rl': True,
  'use_loss_type': 'none',
  'use_value_head': False,
  'use_self_fed_heads': True,
  'use_lstm': True,
  'nlstm': 384,
  'hs_len': 384 * 2,
  'lstm_duration': 1,
  'lstm_dropout_rate': 0.0,
  'lstm_cell_type': 'lstm',
  'lstm_layer_norm': True,
  'weight_decay': 0.00000002,
  'arg_scope_type': 'mnet_v5_type_a',
  'endpoints_verbosity': 10,
  'n_v': 6,
  'distillation': True,
  'fix_all_embed': False,
  'use_base_mask': True,
  'zstat_embed_version': 'v3',
  'trans_version': 'v4',
  'vec_embed_version': 'v3d1',
  'embed_for_action_heads': 'lstm',
  'use_astar_glu': True,
  'use_astar_func_embed': True,
  'pos_logits_mode': '1x1',
  'pos_n_blk': 2,
  'pos_n_skip': 2,
  'sync_statistics': 'none',
  'temperature': 0.8,
  'merge_pi': False,
  'adv_normalize': False,
  'batch_size': 1,
}
# named_vars = OrderedDict([
#   ('embed', '*/embed/*'),
#   ('head_ab', '*/heads/ability/*'),
#   ('head_noop_num', '*/heads/noop_num/*'),
#   ('head_shift', '*/heads/shift/*'),
# ])
named_vars = None
model_dir = '/Users/pengsun/code/tmp/tr1927a_chkpoints'
# model_names = [
#   'None:init_model_20200807073723.model',
#   'init_model:0001_20200807223525.model',
#   'init_model:0001_20200808073425.model',
# ]
model_names = None
this_scope = 'model'


# # decide what vars to view
# if named_vars is None:
#   # defaults to using all vars
#   with open('vars_mnet_v6d6', 'r') as f:
#     all_vars = [line.strip() for line in f.readlines()]
#     named_vars = OrderedDict([(v, v) for v in all_vars])

# get the obs, act space
converter = PB2AllConverter(dict_space=True, zmaker_version='v5')
ob_space, ac_space = converter.space.spaces

# build the net
nc = net_config_cls(ob_space, ac_space, **policy_config)
inputs = net_inputs_placeholders_fun(nc)
out = net_build_fun(inputs, nc, scope=this_scope)

# the (uninitialized) net params and the loading ops
params = tf.trainable_variables(scope=this_scope)
params_ph = [tf.placeholder(p.dtype, shape=p.get_shape()) for p in params]
param_assign_ops = [p.assign(new_p) for p, new_p in zip(params, params_ph)]

# decide what vars to view
if named_vars is None:
  # defaults to using all vars
  kvs = []
  for v in out.vars.all_vars:
    vn = v.name.lstrip(this_scope)
    kvs.append((vn, vn))
  named_vars = OrderedDict(kvs)

# decide what models to view
if model_names is None:
  # defaults to using all models in the dir
  model_names = [f for f in os.listdir(model_dir)
                 if path.isfile(path.join(model_dir, f))
                 and f.endswith('.model') ]
  model_names.sort()  # fine to sort model names in ascending order

# compute the interested param for each model
sess = tf.Session()
for model_name in model_names:
  # load the params
  model_path = path.join(model_dir, model_name)
  np_params = joblib.load(model_path).model
  sess.run(param_assign_ops,
           feed_dict={ph: va for ph, va in zip(params_ph, np_params)})

  # compute the global norm for the tensor list
  fetches = {}
  for name, var_name in named_vars.items():
    tensors = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                scope=this_scope + var_name)
    fetches[name] = tf.global_norm(tensors)
  var_norms = sess.run(fetches)

  for nn, vv in var_norms.items():
    print('{},{},{}'.format(model_name, nn, vv))