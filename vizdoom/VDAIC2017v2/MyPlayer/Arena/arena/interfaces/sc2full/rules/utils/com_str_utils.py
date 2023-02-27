from ...tstarbot_rules.data.pool.macro_def import COMBAT_UNITS
import numpy as np
from enum import Enum

COM_CMD_TYPE = Enum('CMD_TYPE', ('ATK', 'RAL_BASE', 'RAL_BEFORE_ATK', 'DEF', 'HAR', 'ROC'))


def get_zergling_6pos(closest_enemy, base_pos, r=6):
    x = closest_enemy.float_attr.pos_x - base_pos['x']
    y = closest_enemy.float_attr.pos_y - base_pos['y']
    target_x = base_pos['x'] + r * y / (x ** 2 + y ** 2) ** 0.5
    target_y = base_pos['y'] - r * x / (x ** 2 + y ** 2) ** 0.5
    return {'x': target_x, 'y': target_y}


def get_mutalisk_safe_pos(dc):
    base_pool = dc.dd.base_pool
    if len(list(base_pool.bases.values())) > 0:
        base_rand = list(base_pool.bases.values())[0].unit
    else:
        return []
    mutalisk_safe_pos = []
    if base_rand.float_attr.pos_x > base_rand.float_attr.pos_y:
        mutalisk_safe_pos.append({'x': 20, 'y': 5})
        mutalisk_safe_pos.append({'x': 20, 'y': 140})
        mutalisk_safe_pos.append({'x': 20, 'y': 5})
        mutalisk_safe_pos.append({'x': 20, 'y': 75})
        mutalisk_safe_pos.append({'x': 180, 'y': 140})
        mutalisk_safe_pos.append({'x': 100, 'y': 140})
    else:
        mutalisk_safe_pos.append({'x': 180, 'y': 140})
        mutalisk_safe_pos.append({'x': 180, 'y': 5})
        mutalisk_safe_pos.append({'x': 180, 'y': 140})
        mutalisk_safe_pos.append({'x': 180, 'y': 75})
        mutalisk_safe_pos.append({'x': 20, 'y': 5})
        mutalisk_safe_pos.append({'x': 100, 'y': 5})
    return mutalisk_safe_pos


def get_rally_pos(dc):
    base_pool = dc.dd.base_pool
    if list(base_pool.bases.values())[0].unit.float_attr.pos_x < 44:
        return {'x': 50.5, 'y': 60.5}
    else:
        return {'x': 37.5, 'y': 27.5}


def distance(pos_a, pos_b):
    return ((pos_a['x'] - pos_b['x']) ** 2 + (pos_a['y'] - pos_b['y']) ** 2) ** 0.5


def find_base_pos_in_danger(dc, enemy_pos):
    bases = dc.dd.base_pool.bases
    d_min = 10000
    pos = None
    for tag in bases:
        base_pos = {'x': bases[tag].unit.float_attr.pos_x,
                    'y': bases[tag].unit.float_attr.pos_y}
        d = distance(base_pos, enemy_pos)
        if d < d_min:
            d_min = d
            pos = base_pos
    # if len(bases) > 8:
    #     pos = {'x': self._dc.dd.base_pool.home_pos[0],
    #            'y': self._dc.dd.base_pool.home_pos[1]}
    return pos


def find_closest_base_to_enemy(enemy_pool, dc):
    bases = dc.dd.base_pool.bases
    d_min = 10000
    target_base = None
    for tag in bases:
        base_pos = {'x': bases[tag].unit.float_attr.pos_x,
                    'y': bases[tag].unit.float_attr.pos_y}
        d = distance(base_pos, enemy_pool.closest_cluster.centroid)
        if d < d_min:
            d_min = d
            target_base = bases[tag].unit
    return target_base


def find_closest_enemy(u, enemies):
    dist = []
    for e in enemies:
        dist.append(cal_square_dist(u, e))
    idx = np.argmin(dist)
    return enemies[idx]


def find_enemy_combat_units(emeny_units):
    enemy_combat_units = []
    for u in emeny_units:
        if u.int_attr.unit_type in COMBAT_UNITS:
            enemy_combat_units.append(u)
    return enemy_combat_units


def cal_square_dist(u1, u2):
    return pow(pow(u1.float_attr.pos_x - u2.float_attr.pos_x, 2) +
               pow(u1.float_attr.pos_y - u2.float_attr.pos_y, 2), 0.5)


def find_closest_enemy_to_pos(pos, enemies):
    dist = []
    for e in enemies:
        e_pos = {'x': e.float_attr.pos_x,
                 'y': e.float_attr.pos_y}
        dist.append(distance(pos, e_pos))
    idx = np.argmin(dist)
    return enemies[idx]


def get_center_of_units(units):
    center_x = 0
    center_y = 0
    for u in units:
        center_x += u.float_attr.pos_x
        center_y += u.float_attr.pos_y
    if len(units) > 0:
        center_x /= len(units)
        center_y /= len(units)
    pos = {'x': center_x,
           'y': center_y}
    return pos


def get_main_base_pos(dc):
    return {'x': dc.dd.base_pool.home_pos[0],
            'y': dc.dd.base_pool.home_pos[1]}


def get_second_base_pos(dc):
    bases = dc.dd.base_pool.bases
    if len(bases) < 2:
        return None
    min_dist = 100000
    second_base_pos = None
    for tag in bases:
        area = bases[tag].resource_area
        d = dc.dd.base_pool.home_dist[area]
        if 0.1 < d < min_dist:
            min_dist = d
            second_base_pos = area.ideal_base_pos
    return {'x': second_base_pos[0],
            'y': second_base_pos[1]}


def get_slope_to_xy(slopes, xy):
    if slopes is None:
        return None
    min_dist = 100000
    target_slope = None
    for s in slopes:
        d = distance({'x': s.x, 'y': s.y}, xy)
        if d < min_dist:
            min_dist = d
            target_slope = s
    return target_slope


def get_slope_up_pos(slopes, base_pos, offset_fac):
    slope = get_slope_to_xy(slopes, base_pos)
    max_height = max(slope.height)
    highest_pos = [pos for pos, h in zip(slope.pos, slope.height) if h == max_height]
    target_pos = np.mean(highest_pos, axis=0)
    offset_x = base_pos['x'] - target_pos[0]
    offset_y = base_pos['y'] - target_pos[1]
    x = target_pos[0] + offset_fac * offset_x
    y = target_pos[1] + offset_fac * offset_y
    return {'x': x, 'y': y}
