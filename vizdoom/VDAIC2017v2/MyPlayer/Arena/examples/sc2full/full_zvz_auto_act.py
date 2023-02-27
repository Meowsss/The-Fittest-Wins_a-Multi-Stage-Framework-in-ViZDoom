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
from arena.interfaces.sc2full.zerg_data_int import ZergDataInt
from arena.interfaces.sc2full.zerg_auto_int import ZergProductStrategyAutoInt
from arena.interfaces.sc2full.zerg_auto_int import ZergBuildingAutoInt
from arena.interfaces.sc2full.zerg_auto_int import ZergResourceAutoInt
from arena.interfaces.sc2full.zerg_auto_int import ZergCombatStrategyAutoInt
from arena.interfaces.sc2full.zerg_auto_int import ZergCombatMicroAutoInt
from arena.interfaces.sc2full.zerg_auto_int import ZergScoutAutoInt

timer_t = 0
timer_n = 0


def full_zvz(mode):
  inter1 = RawInt()
  inter1 = ZergDataInt(inter1, config_path='arena.interfaces.sc2full.tstarbot_rules.agents.dft_config')
  inter1 = ZergProductStrategyAutoInt(inter1)
  inter1 = ZergBuildingAutoInt(inter1)
  inter1 = ZergResourceAutoInt(inter1)
  inter1 = ZergCombatStrategyAutoInt(inter1)
  inter1 = ZergCombatMicroAutoInt(inter1)
  inter1 = ZergScoutAutoInt(inter1)

  inter2 = RawInt()
  inter2 = ZergDataInt(inter2)
  inter2 = ZergProductStrategyAutoInt(inter2)
  inter2 = ZergBuildingAutoInt(inter2)
  inter2 = ZergResourceAutoInt(inter2)
  inter2 = ZergCombatStrategyAutoInt(inter2)
  inter2 = ZergCombatMicroAutoInt(inter2)
  inter2 = ZergScoutAutoInt(inter2)

  if mode == '2p':
    players = [sc2_env.Agent(sc2_env.Race.zerg), sc2_env.Agent(sc2_env.Race.zerg)]
  else:
    players = [sc2_env.Agent(sc2_env.Race.zerg),
               sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.very_hard)]

  env = SC2BaseEnv(players=players,
                   agent_interface='feature',
                   map_name="AbyssalReef",
                   visualize=False,
                   screen_resolution=64,
                   score_index=0,
                   score_multiplier=1.0 / 1000)
  env = VecRwd(env, append=True)
  if mode == '2p':
    env = EnvIntWrapper(env, [inter1, inter2])
  else:
    env = EnvIntWrapper(env, [inter1])

  print(inter1.observation_space)
  print(inter1.action_space)

  obs = env.reset()
  print("Reset env done!")
  print(inter1.observation_space)
  print(inter1.action_space)

  global timer_t
  global timer_n
  for i in range(12000):
    start_time = time.time()
    if mode == '2p':
      obs, rwd, done, info = env.step([[], []])
    else:
      obs, rwd, done, info = env.step([[]])
    # print(rwd)
    timer_t += (time.time() - start_time)
    timer_n += 1
    if done:
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
