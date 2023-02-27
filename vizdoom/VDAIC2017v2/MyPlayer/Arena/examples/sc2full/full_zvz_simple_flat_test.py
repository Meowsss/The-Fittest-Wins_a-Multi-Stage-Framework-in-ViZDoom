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
from arena.interfaces.sc2full.zerg_simple_flat_act_int import ZergSimpleFlatActInt
from arena.interfaces.sc2full.zerg_data_int import ZergDataInt
from arena.interfaces.sc2full.zerg_simple_obs_int import ZergSimpleObsInt

timer_t = 0
timer_n = 0


def FullZvZ():
    step_mul = 16
    action_mask = False

    inter1 = RawInt()
    inter1 = ZergDataInt(inter1)
    inter1 = ZergSimpleFlatActInt(inter1, action_mask=action_mask)
    inter1 = ZergSimpleObsInt(inter1)

    inter2 = RawInt()
    inter2 = ZergDataInt(inter2)
    inter2 = ZergSimpleFlatActInt(inter2, action_mask=action_mask)
    inter2 = ZergSimpleObsInt(inter2)

    players = [sc2_env.Agent(sc2_env.Race.zerg), sc2_env.Agent(sc2_env.Race.zerg)]
               #sc2_env.Bot(sc2_env.Race.zerg,sc2_env.Difficulty.very_easy)]
    env = SC2BaseEnv(players=players,
                     agent_interface='feature',
                     map_name="Simple64",
                     visualize=False,
                     screen_resolution=64,
                     step_mul=step_mul,
                     disable_fog=True)
    env = EnvIntWrapper(env, [inter1, inter2])
    agent1 = RandomAgent()
    agent2 = RandomAgent()
    #agent2 = AgtIntWrapper(agent2, inter2)

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
    # print(obs[0])
    for i in range(100000):
        action1 = inter1.action_space.sample()
        action2 = inter2.action_space.sample()
        # print(action1)
        start_time = time.time()
        obs, rwd, done, info = env.step([action1, action2])
        time_dur = time.time() - start_time
        # print('step time: {}'.format(time_dur))
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

