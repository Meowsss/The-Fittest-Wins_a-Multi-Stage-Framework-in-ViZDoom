#!/usr/bin/python

"""IntExp script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from absl import app
from pysc2.env import sc2_env
from arena import *

timer_t = 0
timer_n = 0

def ImgIntExp0():
    players = [sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_hard)]
    env = SC2BaseEnv(players=players, agent_interface='feature',
                  map_name="ImmortalZealot", screen_resolution=64,
                  visualize=True)

    inter = RawInt()
    print(inter.observation_space)
    print(inter.action_space)
    inter = ImgObsInt(inter, env._gameinfo)
    print(inter.observation_space)
    print(inter.action_space)
    inter = ImgActInt(inter) # ImgActInt should be placed adjacently after ImgObsInt
    print(inter.observation_space)
    print(inter.action_space)

    env = EnvIntWrapper(env, [inter])

    obs = env.reset()
    global timer_t
    global timer_n
    for i in range(1000):
        action = inter.action_space.sample()
        start_time = time.time()
        obs, rwd, done, info = env.step([action])
        timer_t += (time.time() - start_time)
        timer_n += 1
        if done:
            obs = env.reset()
    print(obs)
    env.close()

def ImgIntExp1():
    players = [sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_hard)]
    env = SC2BaseEnv(players=players, agent_interface='feature',
                  map_name="ImmortalZealot", screen_resolution=64,
                  visualize=True)

    inter = RawInt()
    print(inter.observation_space)
    print(inter.action_space)
    inter = ImgObsInt(inter, env._gameinfo)
    print(inter.observation_space)
    print(inter.action_space)
    inter = ImgActInt(inter) # ImgActInt should be placed adjacently after ImgObsInt
    print(inter.observation_space)
    print(inter.action_space)

    agent = RandomAgent(inter.action_space)
    agent = AgtIntWrapper(agent, inter)

    obs = env.reset()
    global timer_t
    global timer_n
    for i in range(1000):
        action = agent.step(obs[0])
        start_time = time.time()
        obs, rwd, done, info = env.step([action])
        timer_t += (time.time() - start_time)
        timer_n += 1
        if done:
            obs = env.reset()
    print(obs)
    env.close()

def main(unused_argv):
    try:
        ImgIntExp0()
        ImgIntExp1()
    except KeyboardInterrupt:
        pass
    print('\nMean time for step: %f' % (timer_t / timer_n))

if __name__ == "__main__":
    app.run(main)
