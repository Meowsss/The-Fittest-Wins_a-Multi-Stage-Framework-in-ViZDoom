""" observation interfaces """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from gym import spaces
from pysc2.lib.typeenums import UNIT_TYPEID as UNIT_TYPE

from arena.interfaces.interface import Interface
from arena.interfaces.sc2full.sc2learner_obs.common.data_context \
    import DataContext
from arena.interfaces.sc2full.sc2learner_obs.observations.nonspatial_features \
    import PlayerFeature, ScoreFeature, WorkerFeature, GameProgressFeature
from arena.interfaces.sc2full.sc2learner_obs.observations.nonspatial_features \
    import UnitTypeCountFeature, UnitStatCountFeature
from arena.interfaces.sc2full.sc2learner_obs.observations.spatial_features \
    import UnitTypeCountMapFeature, AllianceCountMapFeature


class ZergSC2LearnerObsInt(Interface):
  """ SC2Learner's observation """

  def __init__(self, inter, use_spatial_features=False, resolution=64,
               use_game_progress=True, use_regions=False, override=True):
    super(ZergSC2LearnerObsInt, self).__init__(inter)
    self._use_spatial_features = use_spatial_features
    self._resolution = resolution
    self._use_game_progress = use_game_progress
    self._use_regions = use_regions
    self._dc = DataContext()

    self._init_nonspatial_featurizer()
    if self._use_spatial_features:
      self._init_spatial_featurizer()

  def reset(self, obs, **kwargs):
    super(ZergSC2LearnerObsInt, self).reset(obs, **kwargs)
    self._dc.reset(obs.observation)
    return self._transform_obs(obs.observation)

  def obs_trans(self, obs):
    _obs = self.inter.obs_trans(obs)
    self._dc.update(obs.observation)
    return self._transform_obs(obs.observation)

  @property
  def observation_space(self):
    """ the gym compatible observation_space exposed to caller """
    n_dims = sum([
        self._unit_stat_count_feature.num_dims,
        self._unit_count_feature.num_dims,
        self._building_count_feature.num_dims,
        self._player_feature.num_dims,
        self._score_feature.num_dims,
        self._worker_feature.num_dims,
        self._game_progress_feature.num_dims if self._use_game_progress else 0
    ])
    nonsp_space = spaces.Box(
        low=0.0, high=float('inf'), shape=(n_dims,), dtype=np.float32)

    if self._use_spatial_features:
      n_channels = sum([self._unit_type_count_map_feature.num_channels,
                        self._alliance_count_map_feature.num_channels])
      sp_space = spaces.Box(
          low=0.0,
          high=float('inf'),
          shape=(n_channels, self._resolution, self._resolution),
          dtype=np.float32),
      return spaces.Tuple([sp_space, nonsp_space])
    else:
      return nonsp_space

  def _init_nonspatial_featurizer(self):
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
                   UNIT_TYPE.ZERG_CHANGELING.value,
                   UNIT_TYPE.ZERG_QUEEN.value],
        use_regions=self._use_regions
    )
    self._building_count_feature = UnitTypeCountFeature(
        type_list=[UNIT_TYPE.ZERG_SPINECRAWLER.value,
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
                   UNIT_TYPE.ZERG_GREATERSPIRE.value],
        use_regions=False
    )
    self._unit_stat_count_feature = UnitStatCountFeature(
        use_regions=self._use_regions)
    self._player_feature = PlayerFeature()
    self._score_feature = ScoreFeature()
    self._worker_feature = WorkerFeature()
    if self._use_game_progress:
      self._game_progress_feature = GameProgressFeature()

  def _init_spatial_featurizer(self):
      self._unit_type_count_map_feature = UnitTypeCountMapFeature(
          type_map={UNIT_TYPE.ZERG_DRONE.value: 0,
                    UNIT_TYPE.ZERG_ZERGLING.value: 1,
                    UNIT_TYPE.ZERG_ROACH.value: 2,
                    UNIT_TYPE.ZERG_ROACHBURROWED.value: 2,
                    UNIT_TYPE.ZERG_HYDRALISK.value: 3,
                    UNIT_TYPE.ZERG_OVERLORD.value: 4,
                    UNIT_TYPE.ZERG_OVERSEER.value: 4,
                    UNIT_TYPE.ZERG_HATCHERY.value: 5,
                    UNIT_TYPE.ZERG_LAIR.value: 5,
                    UNIT_TYPE.ZERG_HIVE.value: 5,
                    UNIT_TYPE.ZERG_EXTRACTOR.value: 6,
                    UNIT_TYPE.ZERG_QUEEN.value: 7,
                    UNIT_TYPE.ZERG_RAVAGER.value: 8,
                    UNIT_TYPE.ZERG_BANELING.value: 9,
                    UNIT_TYPE.ZERG_LURKERMP.value: 10,
                    UNIT_TYPE.ZERG_LURKERMPBURROWED.value: 10,
                    UNIT_TYPE.ZERG_VIPER.value: 11,
                    UNIT_TYPE.ZERG_MUTALISK.value: 12,
                    UNIT_TYPE.ZERG_CORRUPTOR.value: 13,
                    UNIT_TYPE.ZERG_BROODLORD.value: 14,
                    UNIT_TYPE.ZERG_SWARMHOSTMP.value: 15,
                    UNIT_TYPE.ZERG_INFESTOR.value: 16,
                    UNIT_TYPE.ZERG_ULTRALISK.value: 17,
                    UNIT_TYPE.ZERG_CHANGELING.value: 18,
                    UNIT_TYPE.ZERG_SPINECRAWLER.value: 19,
                    UNIT_TYPE.ZERG_SPORECRAWLER.value: 20},
          resolution=self._resolution,
      )
      self._alliance_count_map_feature = AllianceCountMapFeature(
          self._resolution)

  def _transform_obs(self, obs):
    need_flip = True if self._dc.init_base_pos[0] < 100 else False
    nonspatial_features = self._extract_nonspatial_features(obs, need_flip)
    if self._use_spatial_features:
      spatial_features = self._extract_spatial_features(obs, need_flip)
      return (spatial_features, nonspatial_features)
    else:
      return nonspatial_features

  def _extract_nonspatial_features(self, obs, need_flip=False):
    unit_type_feat = self._unit_count_feature.features(obs, need_flip)
    building_type_feat = self._building_count_feature.features(obs, need_flip)
    unit_stat_feat = self._unit_stat_count_feature.features(obs, need_flip)
    player_feat = self._player_feature.features(obs)
    score_feat = self._score_feature.features(obs)
    worker_feat = self._worker_feature.features(self._dc)
    if self._use_game_progress:
      game_progress_feat = self._game_progress_feature.features(obs)
    else:
      game_progress_feat = np.array([], dtype=np.float32)
    return np.concatenate([unit_type_feat,
                           building_type_feat,
                           unit_stat_feat,
                           player_feat,
                           score_feat,
                           worker_feat,
                           game_progress_feat])

  def _extract_spatial_features(self, obs, need_flip=False):
    ally_map_feat = self._alliance_count_map_feature.features(obs, need_flip)
    type_map_feat = self._unit_type_count_map_feature.features(obs, need_flip)
    return np.concatenate([ally_map_feat, type_map_feat])
