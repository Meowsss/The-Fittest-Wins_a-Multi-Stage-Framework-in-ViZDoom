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
from arena.interfaces.sc2full.zerg_prod_str_act_int import ZergProdActInt
from arena.interfaces.sc2full.zerg_resource_act_int import ZergResourceActInt
from arena.interfaces.sc2full.zerg_com_str_act_int import ZergCombatActInt
from arena.interfaces.sc2full.zerg_scout_str_act_int import ZergScoutActInt
from arena.interfaces.sc2full.zerg_data_int import ZergDataInt

timer_t = 0
timer_n = 0


def FullZvZScout():
    inter1 = RawInt()
    inter1 = ZergDataInt(inter1)
    inter1 = ZergCombatActInt(inter1, append=False)
    inter1 = ZergProdActInt(inter1)
    inter1 = ZergResourceActInt(inter1)
    inter1 = ZergScoutActInt(inter1)

    inter2 = RawInt()
    inter2 = ZergDataInt(inter2)
    inter2 = ZergCombatActInt(inter2, append=False)
    inter2 = ZergProdActInt(inter2)
    inter2 = ZergResourceActInt(inter2)
    inter2 = ZergScoutActInt(inter2)

    players = [sc2_env.Agent(sc2_env.Race.zerg), sc2_env.Agent(sc2_env.Race.zerg)]
    # players = [sc2_env.Agent(sc2_env.Race.zerg),
    #           sc2_env.Bot(sc2_env.Race.zerg,sc2_env.Difficulty.very_hard)]
    env = SC2BaseEnv(players=players,
                     agent_interface='feature',
                     map_name="AbyssalReef",
                     visualize=True,
                     screen_resolution=64,
                     score_index=0,
                     score_multiplier=1.0/1000)
    env = VecRwd(env, append=True)
    env = EnvIntWrapper(env, [inter1, inter2])
    #env = EnvIntWrapper(env, [inter1])
    agent1 = RandomAgent()
    agent2 = RandomAgent()

    print(inter1.observation_space)
    print(inter1.action_space)

    obs = env.reset()
    print("Reset env done!")
    print(inter1.observation_space)
    print(inter1.action_space)
    agent1.setup(env.observation_space.spaces[0], env.action_space.spaces[0])
    agent1.reset(obs[0])
    agent2.setup(env.observation_space.spaces[1], env.action_space.spaces[1])
    agent2.reset(obs[1])
    print("Reset agents done!")
    global timer_t
    global timer_n
    actions = [0, 4, 9, 13, 16, 17, 20]
    #print(obs[0])
    for i in range(1000):
        action1 = list(agent1.step(obs[0]))
        #action1[-1] = actions[random.randint(0,6)]
        print(action1)
        action2 = list(agent2.step(obs[1]))
        #action2[-1] = actions[random.randint(0, 6)]
        start_time = time.time()
        obs, rwd, done, info = env.step([action1, action2])
        #obs, rwd, done, info = env.step([action1])
        print(rwd)
        timer_t += (time.time() - start_time)
        timer_n += 1
        if done:
            obs = env.reset()
    #print(obs[0])
    env.close()


def main(unused_argv):
    try:
        FullZvZScout()
    except KeyboardInterrupt:
        pass
    print('\nMean time for step: %f' % (timer_t / timer_n))


if __name__ == "__main__":
    app.run(main)

