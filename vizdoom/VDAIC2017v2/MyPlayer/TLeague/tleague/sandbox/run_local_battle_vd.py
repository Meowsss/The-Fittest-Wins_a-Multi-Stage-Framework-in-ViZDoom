import joblib
from absl import app
from absl import flags
from tleague.actors.agent import PGAgent
from tleague.envs.create_envs import create_env
from tleague.utils import read_config_dict
from tleague.utils import import_module_or_data

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "vizdoom_cig2017_track1", "task env")
flags.DEFINE_string("env_config", "{'num_players': 1, 'num_bots': 0, 'is_window_visible':True,'train_mode': 1, 'evaluation': 1 }",
                    "python dict config used for env. ")
flags.DEFINE_string("policy1", "tpolicies.net_zoo.conv_lstm.conv_lstm", "policy used for agent 1")
# 0041_20200921113601.model for frag 0009_20200820081343.model for navi
flags.DEFINE_string("model1", "../../../model/0041_20200921113601.model", "model file used for agent 1")
flags.DEFINE_integer("n_v", 1, "number of value heads")
flags.DEFINE_string("policy_config", "{'test': False, 'use_loss_type': 'none', 'use_value_head': True, 'rollout_len': 1, 'use_lstm': True, 'nlstm': 128, 'hs_len': 256, 'lstm_dropout_rate': 0.2, 'lstm_layer_norm': True, 'weight_decay': 0.00002, 'sync_statistics': None}", "config used for policy")
flags.DEFINE_integer("episodes", 1, "number of episodes")

def main(_):
  env_config = read_config_dict(FLAGS.env_config)
  env = create_env(FLAGS.env, env_config=env_config)
  obs = env.reset()
  policy1 = import_module_or_data(FLAGS.policy1)
  policies = [policy1]

  policy_config = read_config_dict(FLAGS.policy_config)
  agents = [PGAgent(policy, ob_sp, ac_sp, n_v=FLAGS.n_v, scope_name=name, policy_config=policy_config)
          for policy, ob_sp, ac_sp, name in
            zip(policies,
                env.observation_space.spaces,
                env.action_space.spaces,
                ['p1'])]
  model_file1 = FLAGS.model1
  model_0 = joblib.load(model_file1)
  agents[0].load_model(model_0.model)
  agents[0].reset(obs[0])

  episodes = FLAGS.episodes
  iter = 0
  while iter < episodes:
    while True:
      act = [agent.step(ob) for agent, ob in zip(agents, obs)]
      obs, rwd, done, info = env.step(act)
      if done:
        print(rwd)
        obs = env.reset()
        break
    iter += 1


if __name__ == '__main__':
  app.run(main)
