#!/usr/bin/python
"""Test script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import random

from absl import app
from pysc2.env import sc2_env

from arena import *
from arena.wrappers.basic_env_wrapper import VecRwd
from arena.wrappers.basic_env_wrapper import StepMul
from arena.interfaces.sc2full.zerg_sc2learner_obs_int import ZergSC2LearnerObsInt
from arena.interfaces.sc2full.zerg_data_int import ZergDataInt
from arena.interfaces.sc2full.zerg_prod_str_build_order_act_int import ZergProdStrBuildOrderActInt
from arena.interfaces.sc2full.rules.zerg_prod_str_build_order_act import BuildOrderStrategyLing
from arena.interfaces.sc2full.rules.zerg_prod_str_build_order_act import BuildOrderStrategyHydralisk
from arena.interfaces.sc2full.rules.zerg_prod_str_build_order_act import BuildOrderStrategyRoach
from arena.interfaces.sc2full.rules.zerg_prod_str_build_order_act import BuildOrderStrategyMutualisk
from arena.interfaces.sc2full.zerg_auto_int import ZergBuildingAutoInt
from arena.interfaces.sc2full.zerg_auto_int import ZergResourceAutoInt
from arena.interfaces.sc2full.zerg_auto_int import ZergCombatStrategyAutoInt
from arena.interfaces.sc2full.zerg_auto_int import ZergCombatMicroAutoInt
from arena.interfaces.sc2full.zerg_auto_int import ZergScoutAutoInt

timer_t = 0
timer_n = 0


def _print_space(env):
  print('obs space: {}'.format(env.observation_space))
  print('act space: {}'.format(env.action_space))


def full_zvz(mode):
  config_path = 'arena.interfaces.sc2full.tstarbot_rules.sandbox.tmp1'
  bos1 = [
    BuildOrderStrategyLing(),
    BuildOrderStrategyRoach(),
    BuildOrderStrategyHydralisk(),
    BuildOrderStrategyMutualisk(),
  ]
  bos2 = [
    BuildOrderStrategyLing(),
    BuildOrderStrategyRoach(),
    BuildOrderStrategyHydralisk(),
    BuildOrderStrategyMutualisk(),
  ]
  inter1 = RawInt()
  inter1 = ZergSC2LearnerObsInt(inter1, use_spatial_features=False, use_regions=False)
  inter1 = ZergDataInt(inter1, config_path=config_path)
  inter1 = ZergProdStrBuildOrderActInt(inter1, build_order_strategies=bos1, act_freq_game_loop=16*60*4)
  inter1 = ZergBuildingAutoInt(inter1)
  inter1 = ZergResourceAutoInt(inter1)
  inter1 = ZergCombatStrategyAutoInt(inter1)
  inter1 = ZergCombatMicroAutoInt(inter1)
  inter1 = ZergScoutAutoInt(inter1)

  inter2 = RawInt()
  inter2 = ZergSC2LearnerObsInt(inter2, use_spatial_features=False, use_regions=False)
  inter2 = ZergDataInt(inter2, config_path=config_path)
  inter2 = ZergProdStrBuildOrderActInt(inter2, build_order_strategies=bos2, act_freq_game_loop=16*60*4)
  inter2 = ZergBuildingAutoInt(inter2)
  inter2 = ZergResourceAutoInt(inter2)
  inter2 = ZergCombatStrategyAutoInt(inter2)
  inter2 = ZergCombatMicroAutoInt(inter2)
  inter2 = ZergScoutAutoInt(inter2)

  if mode == '2p':
    players = [sc2_env.Agent(sc2_env.Race.zerg), sc2_env.Agent(sc2_env.Race.zerg)]
  else:
    players = [sc2_env.Agent(sc2_env.Race.zerg),
               sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.cheat_insane)]

  env = SC2BaseEnv(players=players,
                   agent_interface='feature',
                   map_name="AbyssalReef",
                   visualize=False,
                   screen_resolution=64,
                   score_index=-1,
                   score_multiplier=1.0,
                   step_mul=32,
                   max_steps_per_episode=30*4*12)
  #env = VecRwd(env, append=True)
  if mode == '2p':
    env = EnvIntWrapper(env, [inter1, inter2])
  else:
    env = EnvIntWrapper(env, [inter1])
  env = StepMul(env, step_mul=30*4)

  obs = env.reset()
  print("Reset env done!")
  _print_space(env)

  global timer_t
  global timer_n
  for i in range(12000):
    start_time = time.time()
    actions = [ac_sp.sample() for ac_sp in env.action_space.spaces]
    if mode == '2p':
      print('act 0: {}, act 1: {}'.format(actions[0], actions[1]))
      obs, rwd, done, info = env.step(actions)
    else:
      print('act: {}'.format(actions[0]))
      obs, rwd, done, info = env.step(actions)
    _print_space(env)
    print(rwd)
    timer_t += (time.time() - start_time)
    timer_n += 1
    if done:
      print('ep done. rew: {}'.format(rwd))
      obs = env.reset()
  # print(obs[0])
  env.close()


def main(unused_argv):
  try:
    full_zvz(mode='1p')
  except KeyboardInterrupt:
    pass
  print('\nMean time for step: %f' % (timer_t / timer_n))


if __name__ == "__main__":
  app.run(main)
