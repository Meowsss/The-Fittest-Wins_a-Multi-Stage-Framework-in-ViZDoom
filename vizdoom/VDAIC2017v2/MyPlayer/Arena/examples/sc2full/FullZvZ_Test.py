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
from arena.interfaces.sc2full.zerg_prod_str_act_int import ZergProdActInt
from arena.interfaces.sc2full.zerg_resource_act_int import ZergResourceActInt
from arena.interfaces.sc2full.zerg_com_str_act_int import ZergCombatActInt
from arena.interfaces.sc2full.zerg_data_int import ZergDataInt
from arena.interfaces.sc2full.zerg_obs_int import ZergAreaObsInt
from arena.interfaces.sc2full.zerg_obs_int import ZergNonspatialObsInt
from arena.interfaces.sc2full.zerg_obs_int import AppendMaskInt

timer_t = 0
timer_n = 0


def FullZvZ():
    step_mul = 16
    action_mask = True

    inter1 = RawInt()
    inter1 = ZergDataInt(inter1)
    inter1 = ZergProdActInt(inter1, append=False, action_mask=action_mask, step_mul=2)
    inter1 = ZergResourceActInt(inter1, step_mul=2)
    inter1 = ZergCombatActInt(inter1, append=True, sub_actions=list(range(4)) + [16,17], step_mul=2)
    inter1 = ZergNonspatialObsInt(inter1, override=True, use_features=(False, False, True, True), n_action=0)
    inter1 = ZergAreaObsInt(inter1, override=False)
    inter1 = AppendMaskInt(inter1)

    inter2 = RawInt()
    inter2 = ZergDataInt(inter2)
    inter2 = ZergCombatActInt(inter2, append=False, sub_actions=list(range(4)) + [16,17], step_mul=4)
    inter2 = ZergProdActInt(inter2, append=True, action_mask=action_mask, step_mul=4)
    inter2 = ZergResourceActInt(inter2, step_mul=4)
    inter2 = ZergAreaObsInt(inter2, override=True)
    inter2 = ZergNonspatialObsInt(inter2, override=False, use_features=(False, False, True, True), n_action=0)
    inter2 = AppendMaskInt(inter2)

    players = [sc2_env.Agent(sc2_env.Race.zerg), sc2_env.Agent(sc2_env.Race.zerg)]
               #sc2_env.Bot(sc2_env.Race.zerg,sc2_env.Difficulty.very_easy)]
    env = SC2BaseEnv(players=players,
                     agent_interface='feature',
                     map_name="AbyssalReef",
                     #map_name="Simple64",
                     visualize=False,
                     screen_resolution=64,
                     step_mul=step_mul,
                     disable_fog=True)
    env = EnvIntWrapper(env, [inter1, None])
    agent1 = RandomAgent()
    agent2 = RandomAgent()
    agent2 = AgtIntWrapper(agent2, inter2)

    obs = env.reset()
    print("Reset env done!")
    print(inter1.observation_space)
    print(inter1.action_space)
    print(inter2.observation_space)
    print(inter2.action_space)
    agent1.setup(env.observation_space.spaces[0], env.action_space.spaces[0])
    agent1.reset(obs[0])
    agent2.setup(env.observation_space.spaces[1], env.action_space.spaces[1])
    agent2.reset(obs[1])
    print(agent1.observation_space)
    print(agent1.action_space)
    print(agent2.observation_space)
    print(agent2.action_space)
    print("Reset agents done!")
    global timer_t
    global timer_n
    actions = [0, 4, 9, 13, 16, 17, 20]
    # print(obs[0])
    for i in range(100000):
        #action1 = list(inter1.action_space.sample())
        #action1[0] = actions[random.randint(0, 6)]
        #action2 = list(inter2.action_space.sample())
        #action2[1] = actions[random.randint(0, 6)]
        # print(action1)
        action1 = agent1.step(obs[0])
        action2 = agent2.step(obs[1])
        start_time = time.time()
        obs, rwd, done, info = env.step([action1, action2])
        time_dur = time.time() - start_time
        #print('step time: {}'.format(time_dur))
        timer_t += time_dur
        timer_n += 1
        if done:
            obs = env.reset()
    # print(obs[0])
    env.close()


def main(unused_argv):
    try:
        FullZvZ()
    except KeyboardInterrupt:
        pass
    print('\nMean time for step: %f' % (timer_t / timer_n))


if __name__ == "__main__":
    app.run(main)

