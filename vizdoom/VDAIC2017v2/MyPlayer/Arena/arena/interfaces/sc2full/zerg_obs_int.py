from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from gym import spaces
from pysc2.lib.typeenums import UNIT_TYPEID as UNIT_TYPE
from pysc2.lib.typeenums import UNIT_TYPEID, UPGRADE_ID
from arena.interfaces.sc2full.sc2learner_obs.observations.nonspatial_features import PlayerFeature
from arena.interfaces.sc2full.sc2learner_obs.observations.nonspatial_features import UnitTypeCountFeature
from arena.interfaces.sc2full.sc2learner_obs.observations.nonspatial_features import UnitStatCountFeature
from arena.interfaces.sc2full.sc2learner_obs.observations.nonspatial_features import GameProgressFeature
from arena.interfaces.sc2full.sc2learner_obs.observations.nonspatial_features import ScoreFeature
from arena.interfaces.interface import Interface
from arena.interfaces.common import AppendObsInt, ActionSeqFeature
from arena.utils.spaces import NoneSpace

from .tstarbot_rules.production_strategy.util import unique_unit_count
from arena.utils.constant import AllianceType, ZERG_BUILDING_UNITS, \
    ZERG_COMBAT_UNITS, ZERG_COMBAT_UNITS_FEAT_NOR, MAIN_BASE_BUILDS, \
    MINERAL_UNITS


class ZergNonspatialObsFunc(object):
    def __init__(self, action_space, use_features=(True, True, True, True, True),
                 n_action=10, override=True, space_old=None):
        self._unit_count_feature = UnitTypeCountFeature(
            type_list=[UNIT_TYPE.ZERG_LARVA.value,
                       UNIT_TYPE.ZERG_DRONE.value,
                       UNIT_TYPE.ZERG_ZERGLING.value,
                       UNIT_TYPE.ZERG_BANELING.value,
                       UNIT_TYPE.ZERG_ROACH.value,
                       UNIT_TYPE.ZERG_ROACHBURROWED.value,
                       UNIT_TYPE.ZERG_RAVAGER.value,
                       UNIT_TYPE.ZERG_HYDRALISK.value,
                       UNIT_TYPE.ZERG_LURKERMP.value,
                       UNIT_TYPE.ZERG_LURKERMPBURROWED.value,
                       UNIT_TYPE.ZERG_VIPER.value,
                       UNIT_TYPE.ZERG_MUTALISK.value,
                       UNIT_TYPE.ZERG_CORRUPTOR.value,
                       UNIT_TYPE.ZERG_BROODLORD.value,
                       UNIT_TYPE.ZERG_SWARMHOSTMP.value,
                       UNIT_TYPE.ZERG_LOCUSTMP.value,
                       UNIT_TYPE.ZERG_INFESTOR.value,
                       UNIT_TYPE.ZERG_ULTRALISK.value,
                       UNIT_TYPE.ZERG_BROODLING.value,
                       UNIT_TYPE.ZERG_OVERLORD.value,
                       UNIT_TYPE.ZERG_OVERSEER.value,
                       UNIT_TYPE.ZERG_QUEEN.value,
                       UNIT_TYPE.ZERG_CHANGELING.value,
                       UNIT_TYPE.ZERG_SPINECRAWLER.value,
                       UNIT_TYPE.ZERG_SPORECRAWLER.value,
                       UNIT_TYPE.ZERG_NYDUSCANAL.value,
                       UNIT_TYPE.ZERG_EXTRACTOR.value,
                       UNIT_TYPE.ZERG_SPAWNINGPOOL.value,
                       UNIT_TYPE.ZERG_ROACHWARREN.value,
                       UNIT_TYPE.ZERG_HYDRALISKDEN.value,
                       UNIT_TYPE.ZERG_HATCHERY.value,
                       UNIT_TYPE.ZERG_EVOLUTIONCHAMBER.value,
                       UNIT_TYPE.ZERG_BANELINGNEST.value,
                       UNIT_TYPE.ZERG_INFESTATIONPIT.value,
                       UNIT_TYPE.ZERG_SPIRE.value,
                       UNIT_TYPE.ZERG_ULTRALISKCAVERN.value,
                       UNIT_TYPE.ZERG_NYDUSNETWORK.value,
                       UNIT_TYPE.ZERG_LURKERDENMP.value,
                       UNIT_TYPE.ZERG_LAIR.value,
                       UNIT_TYPE.ZERG_HIVE.value,
                       UNIT_TYPE.ZERG_GREATERSPIRE.value])
        self._unit_stat_count_feature = UnitStatCountFeature()
        self._player_feature = PlayerFeature()
        self._game_progress_feature = GameProgressFeature()
        self._score_feature = ScoreFeature()
        self.features = [self._unit_stat_count_feature, self._unit_count_feature,
                         self._player_feature, self._game_progress_feature,
                         self._score_feature]
        self.use_features = use_features
        self.use_action_feature = n_action > 0
        self._action_seq_feature = ActionSeqFeature(action_space, n_action)

        n_dims = sum([feature.num_dims if use_feature else 0
                      for feature, use_feature in zip(self.features, self.use_features)]) \
                 + self._action_seq_feature.num_dims
        observation_space = spaces.Box(0.0, float('inf'), [n_dims], dtype=np.float32)
        self.override = override
        if self.override or space_old is None:
            self.observation_space = spaces.Tuple((observation_space,))
        else:
            self.observation_space = \
                spaces.Tuple(space_old.spaces + (observation_space,))

    def observation_transform(self, obs_old, obs, action):
        if action is not None and self.use_action_feature:
            self._action_seq_feature.push_action(action)
        nonspatial_feat = [feature.features(obs.observation) if use_feature else []
                           for feature, use_feature in zip(self.features, self.use_features)]
        if self.use_action_feature:
            action_seq_feat = self._action_seq_feature.features()
        else:
            action_seq_feat = []

        nonspatial_feat = np.concatenate(nonspatial_feat + [action_seq_feat])
        return [nonspatial_feat] if self.override else list(obs_old) + [nonspatial_feat]


class ZergNonspatialObsInt(AppendObsInt):
    def __init__(self, inter, override=True, use_features=(True, False, True, True, True),
                 n_action=10, step_mul=1):
        super(ZergNonspatialObsInt, self).__init__(inter, override)
        self.n_action = n_action
        self.use_features = use_features
        self._action = None
        self.step_mul = step_mul

    def reset(self, obs, **kwargs):
        super(ZergNonspatialObsInt, self).reset(obs, **kwargs)
        self.wrapper = ZergNonspatialObsFunc(self.inter.action_space,
                                             use_features=self.use_features,
                                             n_action=self.n_action,
                                             override=self.override,
                                             space_old=self.inter.observation_space)

    def obs_trans(self, obs):
        self._obs = obs
        obs_old = obs
        if self.inter:
            obs_old = self.inter.obs_trans(obs)
        if (self.unwrapped().steps - 1) % self.step_mul != 0:
            self._action = None
        return self.wrapper.observation_transform(obs_old, obs, self._action)

    def act_trans(self, action):
        self._action = action
        if self.inter:
            action = self.inter.act_trans(action)
        return action


class ZergAreaObsInt(AppendObsInt):
    class Wrapper(object):
        def __init__(self, dc, override, space_old):
            self.feat_types = list(ZERG_BUILDING_UNITS.union(ZERG_COMBAT_UNITS))
            self.feat_types.sort()
            # In earlier version: [129, 9, 109, 289, 688, 86, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 99, 100, 101, 102, 105, 106, 107, 108, 493, 110, 111, 112, 114, 499, 502, 504, 126]
            # sort areas
            areas = dc.dd.base_pool.resource_cluster
            home_dist_dict = dc.dd.base_pool.home_dist
            dists = [home_dist_dict[areas[i]] for i in range(len(areas))]
            area_sort_id = np.argsort(dists)
            self.areas = [areas[i] for i in area_sort_id]

            n_dims = len(self.feat_types) * len(self.areas) * 2  # self and enemy
            n_dims += len(self.areas) * 6  # base_hp, m_worker, g_worker, idle_worker, m_cont, g_cont
            observation_space = spaces.Box(0.0, float('inf'), [n_dims], dtype=np.float32)
            self.TT = dc.sd.TT

            self.override = override
            if self.override or isinstance(space_old, NoneSpace):
                self.observation_space = spaces.Tuple((observation_space,))
            else:
                self.observation_space = \
                    spaces.Tuple(space_old.spaces + (observation_space,))

        def observation_transform(self, obs_pre, obs):
            units = obs.observation['units']
            self_units = [u for u in units if u.int_attr.alliance == AllianceType.SELF.value]
            enemy_units = [u for u in units if u.int_attr.alliance == AllianceType.ENEMY.value]
            neutral_units = [u for u in units if u.int_attr.alliance == AllianceType.NEUTRAL.value]
            self_area_units = self._split_units_by_areas(self_units)
            enemy_area_units = self._split_units_by_areas(enemy_units)
            neutral_area_units = self._split_units_by_areas(neutral_units)

            new_obs = []
            for a_units in self_area_units + enemy_area_units:
                all_type_counts = unique_unit_count(a_units, self.TT)
                feat_type_counts = []
                for feat in self.feat_types:
                    if feat in ZERG_COMBAT_UNITS_FEAT_NOR.keys():
                        feat_type_counts.append(all_type_counts[feat] * ZERG_COMBAT_UNITS_FEAT_NOR[feat])
                    else:
                        feat_type_counts.append(all_type_counts[feat])
                new_obs += feat_type_counts
            for a_units in self_area_units:
                base_hp_ratio = 0
                worker_num = [0] * 3  # spared miniral/gas worker position, idle worker num
                for unit in a_units:
                    if unit.unit_type in MAIN_BASE_BUILDS:
                        base_hp_ratio = unit.float_attr.health / unit.float_attr.health_max
                        worker_num[0] = (unit.int_attr.ideal_harvesters - unit.int_attr.assigned_harvesters) / 16.0
                    elif unit.unit_type == UNIT_TYPEID.ZERG_EXTRACTOR.value:
                        worker_num[1] += (unit.int_attr.ideal_harvesters - unit.int_attr.assigned_harvesters) / 6.0
                    elif unit.unit_type == UNIT_TYPEID.ZERG_DRONE.value and len(unit.orders) == 0:
                        worker_num[2] += 1.0/16.0
                new_obs += [base_hp_ratio] + worker_num
            for a_units in neutral_area_units:
                gas_ratio = 0
                mineral_ratio = 0
                for unit in a_units:
                    if unit.unit_type in MINERAL_UNITS:
                        mineral_ratio += unit.int_attr.mineral_contents / 10800.0  # 900*4 + 1800*4
                    if unit.unit_type == UNIT_TYPEID.NEUTRAL_VESPENEGEYSER:
                        gas_ratio += unit.int_attr.vespene_contents / 4500.0  # 2250*2
                new_obs += [mineral_ratio, gas_ratio]
            assert len(new_obs) == self.observation_space.spaces[-1].shape[0], \
                'feat is not consistent to obs space'
            new_obs = np.array(new_obs, dtype=np.float32)
            return [new_obs] if self.override else list(obs_pre) + [new_obs]

        def _split_units_by_areas(self, units):
            area_poses = [self.areas[i].ideal_base_pos for i in range(len(self.areas))]
            area_units = [[] for i in range(len(area_poses))]
            for u in units:
                a_id = self._belonged_area_id(u, area_poses)
                area_units[int(a_id)].append(u)
            return area_units

        def _belonged_area_id(self, u, area_poses):
            dists = [self._quick_dist(u, a_pos) for a_pos in area_poses]
            a_id = np.argmin(dists)
            return a_id

        def _quick_dist(self, u, pos):
            return (u.float_attr.pos_x - pos[0]) ** 2 + (u.float_attr.pos_y - pos[1]) ** 2

    def reset(self, obs, **kwargs):
        super(ZergAreaObsInt, self).reset(obs, **kwargs)
        self.wrapper = self.Wrapper(self.unwrapped().dc,
                                    override=self.override,
                                    space_old=self.inter.observation_space)


class AppendMaskInt(Interface):
    def __init__(self, inter):
        super(AppendMaskInt, self).__init__(inter)
        self.n_mask = 0
        assert inter

    def reset(self, obs, **kwargs):
        super(AppendMaskInt, self).reset(obs, **kwargs)
        self.n_mask = self.unwrapped().mask_size
        if self.n_mask > 0:
            assert isinstance(self.inter.observation_space, spaces.Tuple)

    @property
    def observation_space(self):
        if self.n_mask == 0:
            return self.inter.observation_space
        else:
            return spaces.Tuple(self.inter.observation_space.spaces + \
                                (spaces.Box(0, 1, (self.n_mask,), dtype=np.float32),))

    def obs_trans(self, obs):
        obs = self.inter.obs_trans(obs)
        if self.n_mask > 0:
            obs = obs + [np.array(self.unwrapped().mask)]
        return obs


class ZergTechObsInt(AppendObsInt):
    class Wrapper(object):
        def __init__(self, override, space_old):
            '''upgrade of self (enemy's upgrade is unavailable)'''
            self.tech_list = [UPGRADE_ID.BURROW.value,
                              UPGRADE_ID.CENTRIFICALHOOKS.value,
                              UPGRADE_ID.CHITINOUSPLATING.value,
                              UPGRADE_ID.EVOLVEMUSCULARAUGMENTS.value,
                              UPGRADE_ID.GLIALRECONSTITUTION.value,
                              UPGRADE_ID.INFESTORENERGYUPGRADE.value,
                              UPGRADE_ID.ZERGLINGATTACKSPEED.value,
                              UPGRADE_ID.ZERGLINGMOVEMENTSPEED.value,
                              UPGRADE_ID.ZERGFLYERARMORSLEVEL1.value,
                              UPGRADE_ID.ZERGFLYERARMORSLEVEL2.value,
                              UPGRADE_ID.ZERGFLYERARMORSLEVEL3.value,
                              UPGRADE_ID.ZERGFLYERWEAPONSLEVEL1.value,
                              UPGRADE_ID.ZERGFLYERWEAPONSLEVEL2.value,
                              UPGRADE_ID.ZERGFLYERWEAPONSLEVEL3.value,
                              UPGRADE_ID.ZERGGROUNDARMORSLEVEL1.value,
                              UPGRADE_ID.ZERGGROUNDARMORSLEVEL2.value,
                              UPGRADE_ID.ZERGGROUNDARMORSLEVEL3.value,
                              UPGRADE_ID.ZERGMELEEWEAPONSLEVEL1.value,
                              UPGRADE_ID.ZERGMELEEWEAPONSLEVEL2.value,
                              UPGRADE_ID.ZERGMELEEWEAPONSLEVEL3.value,
                              UPGRADE_ID.ZERGMISSILEWEAPONSLEVEL1.value,
                              UPGRADE_ID.ZERGMISSILEWEAPONSLEVEL2.value,
                              UPGRADE_ID.ZERGMISSILEWEAPONSLEVEL3.value]
            observation_space = spaces.Box(0.0, 1.0, [len(self.tech_list)], dtype=np.float32)
            self.override = override
            if self.override or isinstance(space_old, NoneSpace):
                self.observation_space = spaces.Tuple((observation_space,))
            else:
                self.observation_space = \
                    spaces.Tuple(space_old.spaces + (observation_space,))

        def observation_transform(self, obs_pre, obs):
            new_obs = [upgrade in obs.observation['raw_data'].player.upgrade_ids for upgrade in self.tech_list]
            new_obs = np.array(new_obs, dtype=np.float32)
            return [new_obs] if self.override else list(obs_pre) + [new_obs]

    def reset(self, obs, **kwargs):
        super(ZergTechObsInt, self).reset(obs, **kwargs)
        self.wrapper = self.Wrapper(override=self.override,
                                    space_old=self.inter.observation_space)


class ZergUnitProg(object):
    def __init__(self, tech_tree, override, space_old,
                 building_list=None, tech_list=None, dtype=np.float32):
        '''Return (in_progress, progess) for each building and tech
        in_progress includes the period the ordered drone moving to target pos
        Only self, enemy's information not available'''
        self.TT = tech_tree
        self.dtype = dtype
        self.building_list = building_list or \
                             [UNIT_TYPE.ZERG_SPAWNINGPOOL.value,
                              UNIT_TYPE.ZERG_ROACHWARREN.value,
                              UNIT_TYPE.ZERG_HYDRALISKDEN.value,
                              UNIT_TYPE.ZERG_HATCHERY.value,
                              UNIT_TYPE.ZERG_EVOLUTIONCHAMBER.value,
                              UNIT_TYPE.ZERG_BANELINGNEST.value,
                              UNIT_TYPE.ZERG_INFESTATIONPIT.value,
                              UNIT_TYPE.ZERG_SPIRE.value,
                              UNIT_TYPE.ZERG_ULTRALISKCAVERN.value,
                              UNIT_TYPE.ZERG_LURKERDENMP.value,
                              UNIT_TYPE.ZERG_LAIR.value,
                              UNIT_TYPE.ZERG_HIVE.value,
                              UNIT_TYPE.ZERG_GREATERSPIRE.value]
        self.tech_list = tech_list or \
                         [UPGRADE_ID.BURROW.value,
                          UPGRADE_ID.CENTRIFICALHOOKS.value,
                          UPGRADE_ID.CHITINOUSPLATING.value,
                          UPGRADE_ID.EVOLVEMUSCULARAUGMENTS.value,
                          UPGRADE_ID.GLIALRECONSTITUTION.value,
                          UPGRADE_ID.INFESTORENERGYUPGRADE.value,
                          UPGRADE_ID.ZERGLINGATTACKSPEED.value,
                          UPGRADE_ID.ZERGLINGMOVEMENTSPEED.value,
                          UPGRADE_ID.ZERGFLYERARMORSLEVEL1.value,
                          UPGRADE_ID.ZERGFLYERARMORSLEVEL2.value,
                          UPGRADE_ID.ZERGFLYERARMORSLEVEL3.value,
                          UPGRADE_ID.ZERGFLYERWEAPONSLEVEL1.value,
                          UPGRADE_ID.ZERGFLYERWEAPONSLEVEL2.value,
                          UPGRADE_ID.ZERGFLYERWEAPONSLEVEL3.value,
                          UPGRADE_ID.ZERGGROUNDARMORSLEVEL1.value,
                          UPGRADE_ID.ZERGGROUNDARMORSLEVEL2.value,
                          UPGRADE_ID.ZERGGROUNDARMORSLEVEL3.value,
                          UPGRADE_ID.ZERGMELEEWEAPONSLEVEL1.value,
                          UPGRADE_ID.ZERGMELEEWEAPONSLEVEL2.value,
                          UPGRADE_ID.ZERGMELEEWEAPONSLEVEL3.value,
                          UPGRADE_ID.ZERGMISSILEWEAPONSLEVEL1.value,
                          UPGRADE_ID.ZERGMISSILEWEAPONSLEVEL2.value,
                          UPGRADE_ID.ZERGMISSILEWEAPONSLEVEL3.value]
        n_dims = len(self.building_list) * 2 + len(self.tech_list) * 2
        observation_space = spaces.Box(0.0, 1.0, [n_dims], dtype=dtype)
        self.override = override
        if self.override or isinstance(space_old, NoneSpace):
            self.observation_space = spaces.Tuple((observation_space,))
        else:
            self.observation_space = \
                spaces.Tuple(space_old.spaces + (observation_space,))
        self.morph_history = {}  # tag: [ability_id, game_loop_start, game_loop_now]

    def building_progress(self, unit_type, obs, alliance=1):
        in_progress = 0
        progress = 0
        unit_data = self.TT.getUnitData(unit_type)
        if not unit_data.isBuilding:
            print('building_in_progress can only be used for buildings!')
        game_loop = obs.observation.game_loop
        if isinstance(game_loop, np.ndarray):
            game_loop = game_loop[0]
        if unit_type in [UNIT_TYPE.ZERG_LAIR.value,
                         UNIT_TYPE.ZERG_HIVE.value,
                         UNIT_TYPE.ZERG_GREATERSPIRE.value]:
            builders = [unit for unit in obs.observation.raw_data.units
                        if unit.unit_type in unit_data.whatBuilds
                        and unit.alliance == alliance]
            for builder in builders:
                if len(builder.orders) > 0 and builder.orders[0].ability_id == unit_data.buildAbility:
                    # pb do not return the progress of unit morphing
                    if (builder.unit_type not in self.morph_history or
                            self.morph_history[builder.unit_type][0] != unit_data.buildAbility):
                        self.morph_history[builder.unit_type] = [unit_data.buildAbility, game_loop, game_loop]
                    else:
                        self.morph_history[builder.unit_type][2] = game_loop
                    in_progress = 1
                    progress = self.morph_history[builder.unit_type][2] - self.morph_history[builder.unit_type][1]
                    progress /= float(unit_data.buildTime)
        else:
            for unit in obs.observation.raw_data.units:
                if (unit.unit_type == unit_type
                        and unit.alliance == alliance
                        and unit.build_progress < 1):
                    in_progress = 1
                    progress = max(progress, unit.build_progress)
                if (unit.unit_type == UNIT_TYPEID.ZERG_DRONE.value
                    and unit.alliance == alliance
                    and len(unit.orders) > 0
                    and unit.orders[0].ability_id == unit_data.buildAbility):
                    in_progress = 1
        return in_progress, progress

    def update_morph_history(self, obs):
        # pb do not return the progress of unit morphing
        game_loop = obs.observation.game_loop
        if isinstance(game_loop, np.ndarray):
            game_loop = game_loop[0]
        for tag in self.morph_history:
            if self.morph_history[tag][2] != game_loop:
                self.morph_history[tag][0] = None

    def upgrade_progress(self, upgrade_type, obs, alliance=1):
        in_progress = 0
        progress = 0
        data = self.TT.getUpgradeData(upgrade_type)
        builders = [unit for unit in obs.observation.raw_data.units
                    if unit.unit_type in data.whatBuilds
                    and unit.alliance == alliance]
        for builder in builders:
            if len(builder.orders) > 0 and builder.orders[0].ability_id == data.buildAbility:
                in_progress = 1
                progress = builder.orders[0].progress
        return in_progress, progress

    def observation_transform(self, obs_pre, obs):
        new_obs = []
        for building in self.building_list:
            new_obs.extend(self.building_progress(building, obs))
        for upgrade in self.tech_list:
            new_obs.extend(self.upgrade_progress(upgrade, obs))
        self.update_morph_history(obs)
        new_obs = np.array(new_obs, dtype=self.dtype)
        return [new_obs] if self.override else list(obs_pre) + [new_obs]

class ZergUnitProgObsInt(AppendObsInt):
    def reset(self, obs, **kwargs):
        super(ZergUnitProgObsInt, self).reset(obs, **kwargs)
        self.wrapper = ZergUnitProg(self.unwrapped().dc.sd.TT,
                                    override=self.override,
                                    space_old=self.inter.observation_space)


class ZergArmDistObsInt(AppendObsInt):
    ''' mean and variance of combat units for both sides'''
    class Wrapper(object):
        def __init__(self, dc, override, space_old):
            self.TT = dc.sd.TT
            self.home_pos = np.array(dc.dd.base_pool.home_pos)
            self.enemy_home_pos = np.array(dc.dd.base_pool.enemy_home_pos)
            self.reverse = -1.0 if self.home_pos[0] < self.enemy_home_pos[0] else 1.0
            self.middle_point = (self.home_pos + self.enemy_home_pos) / 2.0
            self.unit_types = [UNIT_TYPEID.ZERG_BANELING.value,
                               UNIT_TYPEID.ZERG_BANELINGBURROWED.value,
                               UNIT_TYPEID.ZERG_BROODLING.value,
                               UNIT_TYPEID.ZERG_BROODLORD.value,
                               UNIT_TYPEID.ZERG_CORRUPTOR.value,
                               UNIT_TYPEID.ZERG_HYDRALISK.value,
                               UNIT_TYPEID.ZERG_HYDRALISKBURROWED.value,
                               UNIT_TYPEID.ZERG_INFESTOR.value,
                               UNIT_TYPEID.ZERG_INFESTORBURROWED.value,
                               UNIT_TYPEID.ZERG_MUTALISK.value,
                               UNIT_TYPEID.ZERG_RAVAGER.value,
                               UNIT_TYPEID.ZERG_ROACH.value,
                               UNIT_TYPEID.ZERG_ROACHBURROWED.value,
                               UNIT_TYPEID.ZERG_ULTRALISK.value,
                               UNIT_TYPEID.ZERG_ZERGLING.value,
                               UNIT_TYPEID.ZERG_ZERGLINGBURROWED.value,
                               UNIT_TYPEID.ZERG_LURKERMP.value,
                               UNIT_TYPEID.ZERG_LURKERMPBURROWED.value,
                               UNIT_TYPEID.ZERG_VIPER.value]
            n_dims = 8
            observation_space = spaces.Box(-float('inf'), float('inf'),
                                           [n_dims], dtype=np.float32)
            self.override = override
            if self.override or isinstance(space_old, NoneSpace):
                self.observation_space = spaces.Tuple((observation_space,))
            else:
                self.observation_space = \
                    spaces.Tuple(space_old.spaces + (observation_space,))

        def observation_transform(self, obs_pre, obs):
            def mean_std_units_pos(units, TT):
                pos = np.array([(u.float_attr.pos_x, u.float_attr.pos_y)
                                for u in units], dtype=np.float32)
                weight = np.array([[self.TT.getUnitData(unit.unit_type).supplyCost]
                                   for unit in units], dtype=np.float32)
                weight /= weight.sum()
                mean = (pos*weight).sum(axis=0)
                mean_sq = (pos*pos*weight).sum(axis=0)
                std = np.sqrt(mean_sq - mean*mean)
                return mean, std
            mean_factor = 100.0 * self.reverse
            std_factor = 100.0
            new_obs = []
            units = obs.observation['units']
            self_combat_units = [u for u in units
                                 if u.int_attr.alliance == AllianceType.SELF.value
                                 and u.unit_type in self.unit_types]
            enemy_combat_units = [u for u in units
                                  if u.int_attr.alliance == AllianceType.ENEMY.value
                                  and u.unit_type in self.unit_types]
            if len(self_combat_units) == 0:
                new_obs.append((self.home_pos - self.middle_point) / mean_factor)
                new_obs.append(np.array([0.0, 0.0], dtype=np.float32))
            else:
                mean, std = mean_std_units_pos(self_combat_units, self.TT)
                new_obs.append((mean - self.middle_point) / mean_factor)
                new_obs.append(std / std_factor)
            if len(enemy_combat_units) == 0:
                new_obs.append((self.enemy_home_pos - self.middle_point) / mean_factor)
                new_obs.append(np.array([0.0, 0.0], dtype=np.float32))
            else:
                mean, std = mean_std_units_pos(enemy_combat_units, self.TT)
                new_obs.append((mean - self.middle_point) / mean_factor)
                new_obs.append(std / std_factor)
            new_obs = np.concatenate(new_obs)
            return [new_obs] if self.override else list(obs_pre) + [new_obs]

    def reset(self, obs, **kwargs):
        super(ZergArmDistObsInt, self).reset(obs, **kwargs)
        self.wrapper = self.Wrapper(dc=self.unwrapped().dc,
                                    override=self.override,
                                    space_old=self.inter.observation_space)
