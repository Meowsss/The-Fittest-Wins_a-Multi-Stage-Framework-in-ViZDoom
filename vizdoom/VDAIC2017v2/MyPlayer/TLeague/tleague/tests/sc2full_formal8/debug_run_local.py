import importlib

import joblib
from absl import app
from absl import flags
from tleague.actors.agent import PGAgent2
from tleague.envs.create_envs import create_env
from tleague.utils import read_config_dict
from tleague.utils import import_module_or_data


FLAGS = flags.FLAGS
flags.DEFINE_string("env", "sc2full_formal8_dict", "task env")
flags.DEFINE_string("policy1", "tpolicies.net_zoo.mnet_v6.mnet_v6d1", "policy used for agent 1")
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
}


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
  agents = [PGAgent2(policy, ob_sp, ac_sp, n_v=FLAGS.n_v, scope_name=name, policy_config=policy_config)
            for policy, ob_sp, ac_sp, name in
            zip(policies,
                env.observation_space.spaces,
                env.action_space.spaces,
                ['p1', 'p2'])]
  model_file1 = FLAGS.model1
  model_file2 = FLAGS.model2 or model_file1
  model_0 = joblib.load(model_file1)
  model_1 = joblib.load(model_file2)
  agents[0].load_model(model_0.model)
  agents[1].load_model(model_1.model)
  agents[0].reset(obs[0])
  agents[1].reset(obs[1])
  while True:
    act = []
    for agent, ob in zip(agents, obs):
      if ob is not None:
        act.append(agent.step(ob))
      else:
        act.append(None)
    obs, rwd, done, info = env.step(act)
    if done:
      print(rwd)
      break


if __name__ == '__main__':
  app.run(main)
