import importlib

import joblib
from absl import app
from absl import flags
from tleague.actors.agent import PGAgent2
from tleague.envs.create_envs import create_env
import numpy as np
from tleague.utils import import_module_or_data
from matplotlib import pyplot as plt
from timitate.lib6.z_actions import ZERG_ABILITIES


FLAGS = flags.FLAGS
flags.DEFINE_string("env", "sc2full_formal8_dict", "task env")
flags.DEFINE_string("policy1", "tpolicies.net_zoo.mnet_v6.mnet_v6d2", "policy used for agent 1")
flags.DEFINE_string("policy2", "", "policy used for agent 1")
flags.DEFINE_string("model1", "./model", "model file used for agent 1")
flags.DEFINE_string("model2", "", "model file used for agent 1")
flags.DEFINE_integer("n_v", 1, "number of value heads")
flags.DEFINE_string("policy_config", "", "config used for policy")
flags.DEFINE_string("inter_config", "", "config used for interface")

FLAGS.policy_config = {
  'use_xla': False,
  'test': False,
  'rl': False,
  'use_lstm': True,
  'nlstm': 384,
  'hs_len': 384*2,
  'lstm_duration': 1,
  'lstm_dropout_rate': 0.1,
  'use_base_mask': True,
  'lstm_cell_type': 'lstm',
  'lstm_layer_norm': True,
  'weight_decay': 0.0000000001,
  'arg_scope_type': 'type_a',
  'endpoints_verbosity': 10,
  'use_self_fed_heads': True,
  'use_loss_type': 'none',
  'zstat_embed_version': 'v3',
}

FLAGS.inter_config = {
  'zstat_data_src': '/Users/hanlei/Desktop/rp1706-mv9-zstat-mmr6500-KairosJunction-Victor',
  'dict_space': True,
  'max_bo_count': 50,
  'max_bobt_count': 20,
  'zmaker_version': 'v5',
  'correct_pos_radius': 3.5,
}


ability_names = [a[0] for a in ZERG_ABILITIES]
plt.figure(1)
plt.figure(2)


def _default_action():
  return {'A_AB': 0,
          'A_NOOP_NUM': 0,
          'A_SHIFT': 0,
          'A_SELECT': np.array([600]*64, dtype=np.int32),
          'A_CMD_UNIT': 0,
          'A_CMD_POS': 0}


def softmax(logits):
  p = np.exp(logits-np.max(logits))
  return p / np.sum(p)


def plot_logits(logits, a_ab):
  logit_A_AB = softmax(logits['A_AB'][0])
  logit_A_CMD_POS = softmax(logits['A_CMD_POS'][0])
  plot_bar(logit_A_AB, ability_names, a_ab)
  plot_map(np.reshape(logit_A_CMD_POS, (128, 128)))


def plot_bar(v, x_labels, a_ab):
  v_one_hot = np.zeros_like(v)
  v_one_hot[a_ab] = 1

  v1 = v[:61]
  x_labels1 = x_labels[:61]
  v_one_hot1 = v_one_hot[:61]
  v2 = v[61:]
  x_labels2 = x_labels[61:]
  v_one_hot2 = v_one_hot[61:]

  plt.figure(1)

  plt.subplot(211)
  x1 = np.arange(len(v1))
  plt.bar(x1, v1, alpha=0.9, width=0.35, facecolor='blue', edgecolor='blue', label='one', lw=1)
  plt.bar(x1, v_one_hot1*v1, alpha=0.9, width=0.35, facecolor='red', edgecolor='red', label='one1', lw=1)
  plt.xticks(x1, x_labels1, rotation=270, fontsize=10)
  plt.ylim(0, 1)

  plt.subplot(212)
  x2 = np.arange(len(v2))
  plt.bar(x2, v2, alpha=0.9, width=0.35, facecolor='blue', edgecolor='blue', label='two', lw=1)
  plt.bar(x2, v_one_hot2*v2, alpha=0.9, width=0.35, facecolor='red', edgecolor='red', label='two1', lw=1)
  plt.xticks(x2, x_labels2, rotation=270, fontsize=10)
  plt.ylim(0, 1)


def plot_map(v):
  if np.min(v) == np.max(v):
    return
  v /= np.max(v)
  plt.figure(2)
  plt.imshow(v*255, cmap="gray")


def main(_):
  env = create_env(FLAGS.env, inter_config=FLAGS.inter_config)
  obs = env.reset()
  print(env.observation_space.spaces)
  policy1 = import_module_or_data(FLAGS.policy1)
  if not FLAGS.policy2:
    policy2 = policy1
  else:
    policy_module, policy_name = FLAGS.policy2.rsplit(".", 1)
    policy2 = getattr(importlib.import_module(policy_module), policy_name)
  policies = [policy1, policy2]
  policy_config = FLAGS.policy_config
  agent = PGAgent2(policy1, env.observation_space.spaces[0], env.action_space.spaces[0],
                   n_v=FLAGS.n_v, scope_name='p1', policy_config=policy_config)
  model_file1 = FLAGS.model1
  model_0 = joblib.load(model_file1)
  agent.load_model(model_0.model)
  agent.reset(obs[0])
  while True:
    if obs[0] is not None:
      act, logits = agent.step_logits(obs[0])
      print(ZERG_ABILITIES[act['A_AB']][0])
      plt.figure(1)
      plt.clf()
      plt.figure(2)
      plt.clf()
      plot_logits(logits, act['A_AB'])
      plt.figure(1)
      plt.draw()
      plt.pause(0.0001)
      plt.figure(2)
      plt.draw()
      plt.pause(0.0001)
      print('')
    else:
      act = None
    obs, rwd, done, info = env.step([act, _default_action()])
    if done:
      print(rwd)
      break


if __name__ == '__main__':
  app.run(main)
