""" This config file is modified from the default TStarBot config.

A default template config file for an agent.

Treat it as a plain py file.
Define the required configurations in "flat structure", e.g.,
var1 = value1
var2 = value2
...

Do NOT ABUSE it, do NOT define nested or complex data structure.
"""
sleep_per_step = 0.0
building_verbose = 0
building_placer = 'hybrid_v3_1'  # 'naive_predef' | 'hybrid_v2' | 'hybrid_v3'
building_placer_verbose = 0
resource_verbose = 0
production_verbose = 0
combat_verbose = 0
scout_explore_version = 2
explore_rl_support = False
max_forced_scout_count = 0  # num of drones used to scout
default_micro_version = 1
game_version = '4.7.0'  # 4.7.0 REQUIRED!