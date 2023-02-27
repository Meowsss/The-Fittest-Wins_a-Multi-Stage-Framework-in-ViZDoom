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

def AgtIntWrapperExp0():
    players = [sc2_env.Agent(sc2_env.Race.terran), sc2_env.Agent(sc2_env.Race.terran)]
    env = SC2BaseEnv(players=players,
            agent_interface=None,
            map_name="ImmortalZealot",
            screen_resolution=64)
    agent1 = RandomAgent(SC2RawActSpace())
    agent1 = AgtIntWrapper(agent1, None)
    agent2 = RandomAgent(SC2RawActSpace())
    agent2 = AgtIntWrapper(agent2, None)
    print(env.observation_space)
    print(env.action_space)
    obs = env.reset()
    global timer_t
    global timer_n
    for i in range(1000):
        actions = [agent1.step(obs[0]), agent2.step(obs[1])]
        start_time = time.time()
        obs, rwd, done, info = env.step(actions)
        timer_t += (time.time() - start_time)
        timer_n += 1
        if done:
            obs = env.reset()
    env.close()

def AgtIntWrapperExp1():
    players = [sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.very_hard),
            sc2_env.Agent(sc2_env.Race.terran)]
    env = SC2BaseEnv(players=players,
            agent_interface=None,
            map_name="ImmortalZealot",
            screen_resolution=64)
    agent = RandomAgent(SC2RawActSpace())
    agent = AgtIntWrapper(agent, None)
    print(env.observation_space)
    print(env.action_space)
    obs = env.reset()
    global timer_t
    global timer_n
    for i in range(1000):
        actions = [agent.step(obs[0])]
        start_time = time.time()
        obs, rwd, done, info = env.step(actions)
        timer_t += (time.time() - start_time)
        timer_n += 1
        if done:
            obs = env.reset()
    env.close()

def main(unused_argv):
    try:
        AgtIntWrapperExp0()
        AgtIntWrapperExp1()
    except KeyboardInterrupt:
        pass
    print('\nMean time for step: %f' % (timer_t / timer_n))

if __name__ == "__main__":
    app.run(main)
