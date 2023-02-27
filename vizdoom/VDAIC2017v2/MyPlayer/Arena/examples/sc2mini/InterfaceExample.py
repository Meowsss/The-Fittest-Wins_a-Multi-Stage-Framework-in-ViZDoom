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

def Example0():
    inter_1 = UnitOrderInt()
    inter_1 = Discre8MnAInt(inter_1)
    inter_1 = UnitAttrInt(inter_1, override=True)
    inter_2 = UnitOrderInt()
    inter_2 = Discre4M2AInt(inter_2)
    inter_2 = CombineActInt(inter_2)
    inter_2 = UnitAttrInt(inter_2, override=True)

    players = [sc2_env.Agent(sc2_env.Race.terran), sc2_env.Agent(sc2_env.Race.terran)]
    env = SC2BaseEnv(players=players, agent_interface=None, map_name="ImmortalZealot", screen_resolution=64)
    env = MergeUnits(env)
    env = EnvIntWrapper(env, [inter_1, inter_2])
    agent_1 = RandomAgent()
    agent_2 = RandomAgent()

    obs = env.reset()
    agent_1.setup(env.observation_space, env.action_space)
    agent_1.reset(obs[0])
    agent_2.setup(env.observation_space, env.action_space)
    agent_2.reset(obs[1])

    print(obs)
    print(inter_1.observation_space)
    print(inter_1.action_space)
    print(inter_2.observation_space)
    print(inter_2.action_space)
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
            agent_1.setup(env.observation_space.spaces[0], env.action_space.spaces[0])
            agent_1.reset(obs[0])
            agent_2.setup(env.observation_space.spaces[1], env.action_space.spaces[1])
            agent_2.reset(obs[1])

    env.close()

def Example1():
    inter_1 = UnitOrderInt()
    inter_1 = Discre8MnAInt(inter_1)
    inter_1 = UnitAttrInt(inter_1, override=True)
    inter_2 = UnitOrderInt()
    inter_2 = Discre4M2AInt(inter_2)
    inter_2 = CombineActInt(inter_2)
    inter_2 = UnitAttrInt(inter_2, override=True)

    players = [sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Agent(sc2_env.Race.terran)]
    env = SC2BaseEnv(players=players, agent_interface=None, map_name="ImmortalZealotNoReset",
            screen_resolution=64)
    env = MergeUnits(env)
    env = EpisodicLife(env, max_lives=50, max_step=200)

    agent_1 = RandomAgent()
    agent_1 = AgtIntWrapper(agent_1, inter_1)
    agent_2 = RandomAgent()
    agent_2 = AgtIntWrapper(agent_2, inter_2)

    # observation_space and action_space are updated here
    obs = env.reset()
    print(env.observation_space)
    agent_1.setup(env.observation_space.spaces[0], env.action_space.spaces[0])
    agent_1.reset(obs[0])
    agent_2.setup(env.observation_space.spaces[1], env.action_space.spaces[1])
    agent_2.reset(obs[1])

    print(inter_1.observation_space)
    print(inter_1.action_space)
    print(inter_2.observation_space)
    print(inter_2.action_space)
    global timer_t
    global timer_n
    for i in range(1000):
        actions = [agent_1.step(obs[0]), agent_2.step(obs[1])]
        start_time = time.time()
        obs, rwd, done, info = env.step(actions)
        timer_t += (time.time() - start_time)
        timer_n += 1
        if done:
            obs = env.reset()
            agent_1.setup(env.observation_space, env.action_space)
            agent_1.reset(obs[0])
            agent_2.setup(env.observation_space, env.action_space)
            agent_2.reset(obs[1])
    env.close()

def main(unused_argv):
    try:
        Example0()
    except KeyboardInterrupt:
        pass
    print('\nMean time for step: %f' % (timer_t / timer_n))


if __name__ == "__main__":
    app.run(main)
