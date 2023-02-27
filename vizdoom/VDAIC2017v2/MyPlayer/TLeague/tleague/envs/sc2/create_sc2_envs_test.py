from absl import app
from absl import flags

from tleague.envs.sc2.create_sc2_envs import create_sc2_env
from tleague.utils.gym_space import assert_matched


def sc2full_formal4_test():
  env_config = {
    'map_name': 'KairosJunction',
    'step_mul': 1,
  }
  inter_config = {
    'zstat_data_src': '../../sandbox/tmp_replay_ds/rp1113-zstat-selected',
    'mmr': 6200,
    'dict_space': True
  }

  env = create_sc2_env('sc2full_formal4', env_config, inter_config)
  obs = env.reset()
  print(obs)
  assert_matched(env.observation_space, obs)
  for i in range(0, 5):
    act = env.action_space.sample()
    act[0]['A_NOOP_NUM'] = 0
    act[1]['A_NOOP_NUM'] = 1
    assert_matched(env.action_space, act)
    obs, rwd, done, info = env.step(act)
    [ob is None or assert_matched(sp, ob)
     for sp, ob in zip(env.observation_space.spaces, obs)]
    print('step {}, obs {}'.format(i, obs))
    if done:
      obs = env.reset()
      assert_matched(env.observation_space, obs)


def sc2full_formal6_test():
  env_config = {
    'map_name': 'KairosJunction',
    'step_mul': 1,
  }
  inter_config = {
    'zstat_data_src': '../../sandbox/tmp_rp1429-mv-zstat-tmp-selected2',
    'zstat_presort_order_name': 'type_a',
    'mmr': 6200,
    'raw_selection': True,
    'dict_space': True,
  }

  env = create_sc2_env('sc2full_formal6_dict', env_config, inter_config)
  obs = env.reset()
  print(obs)
  assert_matched(env.observation_space, obs)
  for i in range(0, 500):
    act = env.action_space.sample()
    act[0]['A_NOOP_NUM'] = 0
    act[1]['A_NOOP_NUM'] = 1
    assert_matched(env.action_space, act)
    obs, rwd, done, info = env.step(act)
    [ob is None or assert_matched(sp, ob)
     for sp, ob in zip(env.observation_space.spaces, obs)]
    print('step {}, obs {}'.format(i, obs))
    if done:
      obs = env.reset()
      assert_matched(env.observation_space, obs)


def sc2full_formal7_test():
  env_config = {
    'map_name': 'KairosJunction',
    'step_mul': 1,
    'use_trt': True
  }
  inter_config = {
    'zstat_data_src': '/Users/pengsun/code/tmp/rp1522-mmr-ge6000-winner-selected-8',
    'zstat_presort_order_name': '',
    'zmaker_version': 'v5',
    'mmr': 6200,
    'raw_selection': True,
    'dict_space': True,
  }

  env = create_sc2_env('sc2full_formal7_dict', env_config, inter_config)
  obs = env.reset()
  print(obs)
  assert_matched(env.observation_space, obs)
  for i in range(0, 500):
    act = env.action_space.sample()
    act[0]['A_NOOP_NUM'] = 0
    act[1]['A_NOOP_NUM'] = 1
    assert_matched(env.action_space, act)
    obs, rwd, done, info = env.step(act)
    [ob is None or assert_matched(sp, ob)
     for sp, ob in zip(env.observation_space.spaces, obs)]
    print('step {}, obs {}'.format(i, obs))
    if done:
      obs = env.reset()
      assert_matched(env.observation_space, obs)

def main(_):
  #sc2full_formal4_test()
  #sc2full_formal6_test()
  sc2full_formal7_test()


if __name__ == '__main__':
  app.run(main)
