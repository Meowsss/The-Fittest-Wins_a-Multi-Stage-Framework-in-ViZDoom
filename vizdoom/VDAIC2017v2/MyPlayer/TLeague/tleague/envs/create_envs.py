from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app


RECOGNIZED_SC2_ID = {
  'sc2',
  'sc2_with_micro',
  'sc2_without_micro',
  'sc2_unit_rwd_micro',
  'sc2_unit_rwd_no_micro',
  'sc2vsbot_unit_rwd_no_micro',
  'sc2_battle',
  'sc2_battle_fixed_oppo',
  'sc2full_formal',
  'sc2full_formal2',
  'sc2full_formal3',
  'sc2full_formal_nolastaction3',
  'sc2full_formal4',
  'sc2full_formal4_dict',
  'sc2full_formal5_dict',
  'sc2full_formal6_dict',
  'sc2full_formal7_dict',
  'sc2full_formal8_dict',
}

RECOGNIZED_POMMERMAN_ID = {
  'pommerman_v1',
  'pommerman_v1_fog',
  'pommerman_v2',
  'pommerman_v2_fog',
}

RECOGNIZED_VIZDOOM_ID = {
  'vizdoom_dogfight',
  'vizdoom_cig2017_track1',
  'vizdoom_cig2017_track2',
}

RECOGNIZED_SOCCER_ID = {
  'soccer',
}

RECOGNIZED_PONG_ID = {
  'pong_2p'
}

def create_env(arena_id, env_config=None, inter_config=None):
  """ create env from arena/env id using LAZY IMPORT, i.e., the corresponding
  game core (StarCraftII, Pommerman, ViZDoom,... ) is loaded only when used, and
   you don't have to install the game core when not used. """
  if arena_id in RECOGNIZED_SC2_ID:
    from tleague.envs.sc2 import create_sc2_env
    return create_sc2_env(arena_id, env_config=env_config,
                          inter_config=inter_config)
  elif arena_id in RECOGNIZED_POMMERMAN_ID:
    from tleague.envs.pommerman import create_pommerman_env
    return create_pommerman_env(arena_id, env_config=env_config,
                                inter_config=inter_config)
  elif arena_id in RECOGNIZED_VIZDOOM_ID:
    from tleague.envs.vizdoom import create_vizdoom_env
    return create_vizdoom_env(arena_id, env_config, inter_config)
  elif arena_id in RECOGNIZED_SOCCER_ID:
    from tleague.envs.soccer import create_soccer_env
    return create_soccer_env(arena_id)
  elif arena_id in RECOGNIZED_PONG_ID:
    from tleague.envs.pong import create_pong_env
    return create_pong_env(arena_id)
  else:
    raise Exception('Unknown arena_id {}'.format(arena_id))


def env_space(arena_id, env_config=None, inter_config=None):
  """ get observation_space and action_space from arena/env id.

  This is theoretically equivalent to querying the env.observation_space &
  env.action_space. However, for some env these two fields are only correctly
  set AFTER env.reset() is called (e.g., some SC2 env relies on the loaded map),
  which means the game core will be loaded and it can be time consuming.

  In this case, this env_space function can be helpful. It allows the caller to
  quickly get the spaces by "hacking" the arena/env id (when available) without
  having to install the game core. """
  if arena_id in RECOGNIZED_SC2_ID:
    from tleague.envs.sc2 import sc2_env_space
    return sc2_env_space(arena_id, env_config, inter_config)
  elif arena_id in RECOGNIZED_POMMERMAN_ID:
    from tleague.envs.pommerman import pommerman_env_space
    return pommerman_env_space(arena_id, env_config, inter_config)
  elif arena_id in RECOGNIZED_VIZDOOM_ID:
    from tleague.envs.vizdoom import vizdoom_env_space
    return vizdoom_env_space(arena_id)
  elif arena_id in RECOGNIZED_SOCCER_ID:
    from tleague.envs.soccer import soccer_env_space
    return soccer_env_space(arena_id)
  elif arena_id in RECOGNIZED_PONG_ID:
    from tleague.envs.pong import pong_env_space
    return pong_env_space(arena_id)
  else:
    raise Exception('Unknown arena_id {}'.format(arena_id))


def main(_):
  # for temporary testing
  env = create_env('pommerman_v2_fog')
  obs = env.reset()
  # print(env.observation_space.spaces)
  # print(obs[0].shape)
  # print(env.action_space.spaces[0].nvec)
  import time
  for i in range(0, 1000):
    env.render()
    time.sleep(0.1)
    act = env.action_space.sample()
    obs, rwd, done, info = env.step(act)
    if done:
      obs = env.reset()


if __name__ == '__main__':
  app.run(main)
