#!/usr/bin/python

"""Test script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from absl import app
from pysc2.env import sc2_env
from arena import *

timer_t = 0
timer_n = 0

def EnvIntWrapperExp0():
    players = [sc2_env.Agent(sc2_env.Race.terran), sc2_env.Agent(sc2_env.Race.terran)]
    env = SC2BaseEnv(players=players,
            agent_interface=None,
            map_name="ImmortalZealot",
            screen_resolution=64)
    env = EnvIntWrapper(env, [None, None])
    print(env.observation_space)
    print(env.action_space)
    obs = env.reset()
    global timer_t
    global timer_n
    for i in range(1000):
        #actions = [env.action_space[0].sample(), env.action_space[1].sample()]
        actions = env.action_space.sample()
        start_time = time.time()
        obs, rwd, done, info = env.step(actions)
        timer_t += (time.time() - start_time)
        timer_n += 1
        if done:
            obs = env.reset()
    env.close()

def EnvIntWrapperExp1():
    players = [sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_hard),
            sc2_env.Agent(sc2_env.Race.terran)]
    env = SC2BaseEnv(players=players,
            agent_interface=None,
            map_name="ImmortalZealot",
            screen_resolution=64)
    env = EnvIntWrapper(env, [None])
    print(env.observation_space)
    print(env.action_space)
    obs = env.reset()
    global timer_t
    global timer_n
    for i in range(1000):
        actions = env.action_space.sample()
        start_time = time.time()
        obs, rwd, done, info = env.step(actions)
        timer_t += (time.time() - start_time)
        timer_n += 1
        if done:
            obs = env.reset()
    env.close()

def main(unused_argv):
    try:
        EnvIntWrapperExp0()
        EnvIntWrapperExp1()
    except KeyboardInterrupt:
        pass
    print('\nMean time for step: %f' % (timer_t / timer_n))

if __name__ == "__main__":
    app.run(main)
