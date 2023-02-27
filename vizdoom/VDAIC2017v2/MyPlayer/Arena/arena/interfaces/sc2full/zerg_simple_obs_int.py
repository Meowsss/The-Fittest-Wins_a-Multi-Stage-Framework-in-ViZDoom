from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from gym import spaces
from pysc2.lib.typeenums import UNIT_TYPEID

from arena.interfaces.common import AppendObsInt
from arena.utils.constant import AllianceType


class ZergSimpleObsInt(AppendObsInt):
    class Wrapper(object):
        def __init__(self):
            n_dims = 9 * 2 + 2  # self and enemy + money and gas
            self.observation_space = spaces.Box(0.0, float('inf'), [n_dims], dtype=np.float32)

        def observation_transform(self, obs_pre, obs):
            units = obs.observation['units']
            self_units = [u for u in units if u.int_attr.alliance == AllianceType.SELF.value]
            enemy_units = [u for u in units if u.int_attr.alliance == AllianceType.ENEMY.value]

            self_feat = self._get_feature(self_units)
            enemy_feat = self._get_feature(enemy_units)

            self_mineral = obs.observation['player'][1]
            self_gas = obs.observation['player'][2]

            new_obs = self_feat + enemy_feat + [self_mineral / 10000.0, self_gas / 10000.0]

            assert len(new_obs) == self.observation_space.shape[0], \
                'feat is not consistent to obs space'
            new_obs = np.array(new_obs, dtype=np.float32)

            return new_obs

        def _get_feature(self, units):
            feat = []

            hatchery_hp = 0
            for unit in units:
                if unit.unit_type == UNIT_TYPEID.ZERG_HATCHERY.value:
                    hatchery_hp = unit.float_attr.health / unit.float_attr.health_max
            feat += [hatchery_hp]

            extractor_hp = 0
            for unit in units:
                if unit.unit_type == UNIT_TYPEID.ZERG_EXTRACTOR.value:
                    extractor_hp = unit.float_attr.health / unit.float_attr.health_max
            feat += [extractor_hp / 2.0]

            spawningpool_hp = 0
            for unit in units:
                if unit.unit_type == UNIT_TYPEID.ZERG_SPAWNINGPOOL.value:
                    spawningpool_hp = unit.float_attr.health / unit.float_attr.health_max
            feat += [spawningpool_hp]

            roachwarren_hp = 0
            for unit in units:
                if unit.unit_type == UNIT_TYPEID.ZERG_ROACHWARREN.value:
                    roachwarren_hp = unit.float_attr.health / unit.float_attr.health_max
            feat += [roachwarren_hp]

            worker_num = 0
            for unit in units:
                if unit.unit_type == UNIT_TYPEID.ZERG_DRONE.value:
                    worker_num += 1.0
            feat += [worker_num / 100.0]

            overlord_num = 0
            for unit in units:
                if unit.unit_type == UNIT_TYPEID.ZERG_OVERLORD.value:
                    overlord_num += 1.0
            feat += [overlord_num / 25.0]

            queen_num = 0
            for unit in units:
                if unit.unit_type == UNIT_TYPEID.ZERG_QUEEN.value:
                    queen_num += 1.0
            feat += [queen_num / 100.0]

            zergling_cnt = 0
            for unit in units:
                if unit.unit_type == UNIT_TYPEID.ZERG_ZERGLING.value:
                    zergling_cnt += 1.0
            feat += [zergling_cnt / 400.0]

            roach_cnt = 0
            for unit in units:
                if unit.unit_type == UNIT_TYPEID.ZERG_ROACH.value:
                    roach_cnt += 1.0
            feat += [roach_cnt / 100.0]

            return feat

    def reset(self, obs, **kwargs):
        super(ZergSimpleObsInt, self).reset(obs, **kwargs)
        self.wrapper = self.Wrapper()
