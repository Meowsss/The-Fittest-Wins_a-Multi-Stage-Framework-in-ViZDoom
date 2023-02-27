#!/usr/bin/python

"""Test script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from absl import app
from pysc2.env import sc2_env
from arena import *
# This requires sc2learner
from arena.interfaces.sc2full.zerg_obs_int import ZergNonspatialObsInt
from arena.interfaces.sc2full.zerg_prod_str_act_int import ZergProdActInt
from arena.interfaces.sc2full.zerg_resource_act_int import ZergResourceActInt
from arena.interfaces.sc2full.zerg_com_str_act_int import ZergCombatActInt
from arena.interfaces.sc2full.zerg_data_int import ZergDataInt

timer_t = 0
timer_n = 0

def ZergNonspatialObsExp():
    inter1 = RawInt()
    inter1 = ZergDataInt(inter1)
    inter1 = ZergProdActInt(inter1, append=False)
    inter1 = ZergResourceActInt(inter1, auto_resource=False)
    inter1 = ZergCombatActInt(inter1)
    inter1 = ZergNonspatialObsInt(inter1, override=True, n_action=0)

    inter2 = RawInt()
    inter2 = ZergDataInt(inter2)
    inter2 = ZergProdActInt(inter2, append=False)
    inter2 = ZergResourceActInt(inter2)
    inter2 = ZergCombatActInt(inter2)
    inter2 = ZergNonspatialObsInt(inter2, override=True, n_action=0)

    players = [sc2_env.Agent(sc2_env.Race.zerg), sc2_env.Agent(sc2_env.Race.zerg)]
    env = SC2BaseEnv(players=players, agent_interface='feature', map_name="AbyssalReef",
                  screen_resolution=64)
    env = EnvIntWrapper(env, [inter1, inter2])
    agent1 = RandomAgent()
    agent2 = RandomAgent()

    obs = env.reset()
    print("Reset env done!")
    print(inter1.observation_space)
    print(inter1.action_space)
    agent1.setup(env.observation_space.spaces[0], env.action_space.spaces[0])
    agent1.reset(obs[0])
    agent2.setup(env.observation_space.spaces[1], env.action_space.spaces[1])
    agent2.reset(obs[1])
    global timer_t
    global timer_n
    for i in range(10000):
        action1 = agent1.step(obs[0])
        action2 = agent2.step(obs[1])
        start_time = time.time()
        obs, rwd, done, info = env.step([action1, action2])
        # print('LastAction num : {}'.format([len(raw_obs.actions) for raw_obs in env.env.env._obs]))
        # print('Error Message: {}'.format([raw_obs.action_errors for raw_obs in env.env.env._obs]))
        timer_t += (time.time() - start_time)
        timer_n += 1
        if done:
            obs = env.reset()
    print(obs)
    env.close()


def main(unused_argv):
    try:
        ZergNonspatialObsExp()
    except KeyboardInterrupt:
        pass
    print('\nMean time for step: %f' % (timer_t / timer_n))

if __name__ == "__main__":
    app.run(main)

