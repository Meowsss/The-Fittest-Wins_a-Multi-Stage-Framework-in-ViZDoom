import os
from collections import OrderedDict

import numpy as np
from gym import RewardWrapper, Wrapper
from gym import spaces
from pysc2.env import sc2_env
from pysc2.lib.typeenums import ABILITY_ID
from arena.env.sc2_base_env import SC2BaseEnv
from arena.interfaces.raw_int import RawInt
from arena.interfaces.interface import Interface
from arena.interfaces.sc2full.zerg_data_int import ZergDataInt
from arena.interfaces.sc2full.zerg_obs_int import ZergAreaObsInt
from arena.interfaces.sc2full.zerg_obs_int import ZergNonspatialObsInt
from arena.interfaces.sc2full.zerg_obs_int import AppendMaskInt
from arena.interfaces.sc2full.zerg_obs_int import ZergTechObsInt
from arena.interfaces.sc2full.zerg_obs_int import ZergUnitProgObsInt
from arena.interfaces.sc2full.zerg_scout_int import ZergScoutInt
from arena.interfaces.sc2full.zerg_prod_str_act_int import ZergProdActInt
from arena.interfaces.sc2full.zerg_resource_act_int import ZergResourceActInt
from arena.interfaces.sc2full.zerg_com_str_act_int import ZergCombatActInt
from arena.env.env_int_wrapper import EnvIntWrapper, SC2EnvIntWrapper
from arena.wrappers.basic_env_wrapper import StepMul
from arena.wrappers.basic_env_wrapper import VecRwd
from arena.wrappers.basic_env_wrapper import EarlyTerminate
from arena.interfaces.common import ConcatVecWrapper
from arena.utils.spaces import NoneSpace
from arena.wrappers.sc2stat_wrapper import StatAllAction
from arena.wrappers.sc2stat_wrapper import StatZStatFn
from arena.wrappers.basic_env_wrapper import OppoObsAsObs


AGENT_INTERFACE = 'feature'
VISUALIZE = False
SCORE_INDEX = -1
# TODO(pengsun): these settings should be set by parsing arena_id
DEFAULT_MAP_NAME = 'AbyssalReef' # 'AbyssalReef' | 'KairosJunction'
DISABLE_FOG = False
SCORE_MULTIPLIER = None
MAX_RESET_NUM = 20
MAX_STEPS_PER_EPISODE = 48000
SCREEN_RESOLUTION = 64

RACES = {
    "R": sc2_env.Race.random,
    "P": sc2_env.Race.protoss,
    "T": sc2_env.Race.terran,
    "Z": sc2_env.Race.zerg,
}
DIFFICULTIES = {
    1: sc2_env.Difficulty.very_easy,
    2: sc2_env.Difficulty.easy,
    3: sc2_env.Difficulty.medium,
    4: sc2_env.Difficulty.medium_hard,
    5: sc2_env.Difficulty.hard,
    6: sc2_env.Difficulty.harder,
    7: sc2_env.Difficulty.very_hard,
    8: sc2_env.Difficulty.cheat_vision,
    9: sc2_env.Difficulty.cheat_money,
    10: sc2_env.Difficulty.cheat_insane,
}


def _get_sc2_version():
  import importlib
  cfg = importlib.import_module('tleague.envs.sc2.dft_config')
  return cfg.game_version


class SeqLevDist_with_Coord(object):
  ## Compute LevDist (modified) using sequential added (item, pos) pair ##
  def __init__(self, target_order, target_boc):
    self.len = int(np.sum(target_order))
    self.target_order = target_order[:self.len]
    self.target_boc = target_boc[:self.len]
    self.dist = np.zeros(self.len + 1)
    self.index = 0

  @staticmethod
  def binary2int(bx):
    x = 0
    for i in bx:
      x = 2*x + i
    return x

  def pos_dist(self, pos1, pos2):
    MAX_DIST = 6 # 2 gateways
    max_d = 0.8 # scaled to [0, max_d]
    assert len(pos1) == len(pos2)
    LEN = int(len(pos1)/2)
    if len(pos1) > 2: # binary encoding
      pos1 = [self.binary2int(pos1[:LEN]) / 2,
              self.binary2int(pos1[LEN:]) / 2, ]
      pos2 = [self.binary2int(pos2[:LEN]) / 2,
              self.binary2int(pos2[LEN:]) / 2, ]
    sq_dist = ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    return max_d * min(1, sq_dist / MAX_DIST)

  def lev_dist(self, order, boc):
    index_now = int(np.sum(order))
    assert index_now >= self.index
    for index in range(self.index+1, index_now+1):
      new_dist = index * np.ones(self.len + 1)
      for i in range(1, self.len + 1):
        if all(order[index-1] == self.target_order[i-1]):
          new_dist[i] = self.dist[i - 1] + self.pos_dist(boc[index-1], self.target_boc[i-1])
        else:
          new_dist[i] = self.dist[i - 1] + 1
        new_dist[i] = min(new_dist[i-1]+1, self.dist[i]+1, new_dist[i])
      self.dist = new_dist
    self.index = index_now
    return self.dist[-1]

  def min_dist_possible(self):
    add_on = np.abs(np.arange(self.len + 1) - self.index)
    return np.min(self.dist + add_on)


class SC2ZergZStatVecReward(RewardWrapper):
  """ Reward shaping for zstat. Works for player 0, and fill
  arbitrary value for player 1. Output vectorized reward. """
  def __init__(self, env, dict_space=True, version='v1', compensate_win=True):
    super(SC2ZergZStatVecReward, self).__init__(env)
    self.dict_space = dict_space
    self.version = version
    self.compensate_win = compensate_win
    self.rwd_dim = 4 + (version == 'v3') + 2*(version == 'v4')

  def reset(self, **kwargs):
    obs = super(SC2ZergZStatVecReward, self).reset(**kwargs)
    self._potential_func = self._create_potential_func(obs[0], self.dict_space, self.version)
    self._last_potential = self._potential_func(self.unwrapped._obs[0], obs, False)
    if self.version in ['v4']:
      self._last_potential = self._last_potential + [self._count_creep(self.unwrapped._obs[0])]
    return obs

  @staticmethod
  def _create_potential_func(obs, dict_space, version):
    if version == 'v1':
      from timitate.lib4.pb2zstat_converter import PB2ZStatConverterV2
      from timitate.lib4.zstat_utils import BUILD_ORDER_CANDIDATES, RESEARCH_ABILITY_CANDIDATES
      pb2zstat_cvt = PB2ZStatConverterV2()
      pb2zstat_cvt.reset()
      split_indices = [len(BUILD_ORDER_CANDIDATES), -len(RESEARCH_ABILITY_CANDIDATES)]
    elif version in ['v2', 'v3', 'v4']:
      from timitate.lib5.zstat_utils import BUILD_ORDER_OBJECT_CANDIDATES, RESEARCH_ABILITY_CANDIDATES
      split_indices = [len(BUILD_ORDER_OBJECT_CANDIDATES), -len(RESEARCH_ABILITY_CANDIDATES)]
    def _get_zstat(ob, prefix='Z_'):
      order_bt, boc_bt = None, None
      if dict_space:
        uc = np.split(ob[prefix + 'UNIT_COUNT'] > 0, split_indices)
        order = ob[prefix + 'BUILD_ORDER']
        boc = ob[prefix + 'BUILD_ORDER_COORD']
        if version in ['v3', 'v4']:
          order_bt = ob[prefix + 'BUILD_ORDER_BT']
          boc_bt = ob[prefix + 'BUILD_ORDER_COORD_BT']
      else:
        uc = np.split(ob[15] > 0, split_indices)
        order = ob[16]
        boc = ob[17]
        if version in ['v3', 'v4']:
          order_bt = ob[18]
          boc_bt = ob[19]
      return uc, order, boc, order_bt, boc_bt
    target_uc, target_order, target_boc, target_order_bt, target_boc_bt = _get_zstat(obs, 'Z_')
    SeqLevDist = SeqLevDist_with_Coord(target_order, target_boc)
    if version in ['v3', 'v4']:
      SeqLevDist_bt = SeqLevDist_with_Coord(target_order_bt, target_boc_bt)
    use_zstat = np.sum(target_order) > 0
    def potential_func(raw_pb, ob, win):
      if use_zstat:
        if version == 'v1':
          zstat = pb2zstat_cvt.convert(raw_pb)
          uc = np.split(zstat[0] > 0, split_indices)
          order = zstat[1]
          boc = zstat[2]
          order_bt, boc_bt = None, None
        else:
          uc, order, boc, order_bt, boc_bt = _get_zstat(obs, 'IMM_')
        d_order = [- SeqLevDist.lev_dist(order, boc) / float(target_order.shape[0])]
        # negative Hamming Distance as potential for binary unit count
        d_uc = [ - sum(u != target_u) / float(len(target_u)) for u, target_u in zip(uc, target_uc)]
        if win:
          d_order = [- SeqLevDist.min_dist_possible() / float(target_order.shape[0])]
          d_uc = [- sum(u & ~target_u) / float(len(target_u)) for u, target_u in zip(uc, target_uc)]
        d_order_bt = []
        if version in ['v3', 'v4']:
          d_order_bt = [- SeqLevDist_bt.lev_dist(order_bt, boc_bt) / float(target_order_bt.shape[0])]
          if win:
            d_order_bt = [- SeqLevDist_bt.min_dist_possible() / float(target_order_bt.shape[0])]
        return d_order + d_order_bt + d_uc
      else:
        return [0] * (4 + (version == 'v3') + 2*(version == 'v4'))
    return potential_func

  @staticmethod
  def _get_creep(raw_obs):
    from timitate.lib5.pb2feature_converter import norm_img, bitmap2array
    if hasattr(raw_obs.observation, 'feature_minimap'):
      creep = raw_obs.observation.feature_minimap.creep
      creep = norm_img(np.array(creep, np.int32))
    elif hasattr(raw_obs.observation, 'feature_layer_data'):
      creep = raw_obs.observation.feature_layer_data.minimap_renders.creep
      creep = norm_img(bitmap2array(creep))
    else:
      raise KeyError('obs.observation has no feature_minimap or feature_layer_data!')
    return creep

  @staticmethod
  def _get_tumor_creep(raw_obs, creep):
    from pysc2.lib.typeenums import UNIT_TYPEID
    tumor_radius = 5
    units = raw_obs.observation.raw_data.units
    tumors = [u for u in units if u.alliance == 1 and u.unit_type in [
      UNIT_TYPEID.ZERG_CREEPTUMORQUEEN.value,
      UNIT_TYPEID.ZERG_CREEPTUMOR.value,
      UNIT_TYPEID.ZERG_CREEPTUMORBURROWED.value]]
    tumor_pos = [(u.pos.x, u.pos.y) for u in tumors]
    y, x = creep.shape
    tumor_creep = np.zeros_like(creep)
    for pos in tumor_pos:
      mat_r = y-1-int(pos[1])
      mat_c = int(pos[0])
      # for efficiency, fill a square, which will be further refined by '* creep'
      tumor_creep[max(mat_r-tumor_radius, 0):min(mat_r+tumor_radius, y-1),
                  max(mat_c-tumor_radius, 0):min(mat_c+tumor_radius, x-1)] = 1
    tumor_creep = tumor_creep * creep

    ## check the images using the following codes
    # from PIL import Image
    # array = np.array(creep * 255, dtype=np.int8)
    # im = Image.fromarray(array, mode='L')
    # im = im.convert('RGB')
    # im.save('creep.jpg')
    # array = np.array(tumor_creep * 255, dtype=np.int8)
    # im = Image.fromarray(array, mode='L')
    # im = im.convert('RGB')
    # im.save('tumor_creep.jpg')

    return tumor_creep

  def _count_creep(self, raw_obs):
    # get creep
    creep = self._get_creep(raw_obs)
    tumor_creep = self._get_tumor_creep(raw_obs, creep)
    creep_cnt = np.sum(tumor_creep) / float(creep.shape[0] * creep.shape[1])
    return creep_cnt

  def step(self, actions):
    obs, rwd, done, info = self.env.step(actions)
    win = done and (rwd[0] > 0) and self.compensate_win
    potential = self._potential_func(self.unwrapped._obs[0], obs, win)
    if self.version in ['v4']:
      creep_cnt = self._count_creep(self.unwrapped._obs[0])
      potential = potential + [creep_cnt]
    game_loop = self.unwrapped._obs[0].observation.game_loop
    ratio = np.ones([self.rwd_dim])
    if game_loop > 22.4 * 60 * 8:
      ratio[-4:] *= 0.5
    elif game_loop > 22.4*60*12:
      ratio[-4:] *= 0.5
    elif game_loop > 22.4*60*16:
      ratio[-4:] = 0
    r = [(p-lp) * rr for p, lp, rr in zip(potential, self._last_potential, ratio)]
    self._last_potential = potential
    rwd[0] = list(rwd[0]) + r if isinstance(rwd[0], (list, tuple)) else [rwd[0]] + r
    if len(rwd) == 2:
      rwd[1] = list(rwd[1]) + [0] * self.rwd_dim if isinstance(rwd[1], (list, tuple))\
        else [rwd[1]] + [0] * self.rwd_dim
    return obs, rwd, done, info


class SC2ZergUnitProductVecReward(RewardWrapper):
  """ Reward shaping for Unit Production. Works for player 0, and fill
  arbitrary value for player 1. Output vectorized reward. """

  def __init__(self, env, watched_ability_ids=None, scale=None):
    super(SC2ZergUnitProductVecReward, self).__init__(env)
    self._watched_ability_ids = watched_ability_ids or [
      ABILITY_ID.TRAIN_ZERGLING.value,
      ABILITY_ID.TRAIN_BANELING.value,
      ABILITY_ID.TRAIN_HYDRALISK.value,
      ABILITY_ID.TRAIN_ROACH.value,
      ABILITY_ID.TRAIN_QUEEN.value,
    ]
    self._ability_id_to_ind = {id: ind for ind, id in
                               enumerate(self._watched_ability_ids)}
    self.reward_len = len(self._watched_ability_ids)
    self.scale = scale
    if self.scale:
      assert len(self.scale) == len(self._watched_ability_ids)
    # self.reward_len = len(self._watched_ability_ids)
    # box = spaces.Box(-200, 200, (self.reward_len + 1,))
    # self.reward_space = spaces.Tuple([box, box])

  def step(self, actions):
    obs, rwd, done, info = self.env.step(actions)
    # only reward shaping for player 0
    rwd[0] = self._calc_player_shaped_rwd(rwd[0], actions[0])
    # arbitrary for player 1
    if len(rwd) == 2:
      rwd[1] = self._calc_player_shaped_rwd(rwd[1], [])
    return obs, rwd, done, info

  def _calc_player_shaped_rwd(self, rwd, pb_acts):
    vec_r = [0] * self.reward_len
    for a in pb_acts:
      ability_id = a.action_raw.unit_command.ability_id
      if ability_id in self._watched_ability_ids:
        ind = self._ability_id_to_ind[ability_id]
        vec_r[ind] += len(a.action_raw.unit_command.unit_tags)
    if self.scale:
      vec_r = [r * scale for r, scale in zip(vec_r, self.scale)]
    rwd = [rwd] + vec_r if not isinstance(rwd, (list, tuple)) else list(
      rwd) + vec_r
    return rwd


class HackActInt(Interface):
  """ Transform action from (a,b,(c,d)) to (a,b,c,d) """

  @property
  def action_space(self):
    if isinstance(self.inter.action_space, NoneSpace):
      return NoneSpace()
    nvec = self.inter.action_space.nvec
    sps = [spaces.Discrete(nvec[0]),
           spaces.Discrete(nvec[1]),
           spaces.MultiDiscrete(tuple(nvec[2:]))]
    sps[2].dtype = np.dtype(np.int32)
    return spaces.Tuple(sps)

  def act_trans(self, action):
    flat_action = list(action[0:2]) + list(action[2])
    return self.inter.act_trans(flat_action)


class ActDtypeInt(Interface):
  @property
  def action_space(self):
    if isinstance(self.inter.action_space, NoneSpace):
      return NoneSpace()
    elif isinstance(self.inter.action_space, spaces.MultiDiscrete):
      ac_space = self.inter.action_space
      ac_space.dtype = np.dtype(np.int32)
      return ac_space
    elif isinstance(self.inter.action_space, spaces.Tuple):
      sps = list(self.inter.action_space.spaces)
      for sp in sps:
        if isinstance(sp, spaces.MultiDiscrete):
          sp.dtype = np.dtype(np.int32)
      return spaces.Tuple(sps)


class OppoObsComponentsV1:
  def __init__(self, game_version, dict_space):
    from timitate.lib5.pb2feature_converter import GlobalFeatMaker, ImmZStatMaker
    self._dict_space = dict_space
    self._components = [GlobalFeatMaker(game_version), ImmZStatMaker()]

  def make(self, pb):
    if self._dict_space:
      dict_features = OrderedDict()
      for com in self._components:
        dict_features.update(
          OrderedDict(['OPPO_'+k, v] for k, v in zip(com.tensor_names, com.make(pb))))
      return dict_features
    else:
      tuple_features = ()
      for com in self._components:
        tuple_features += com.make(pb)

  @property
  def space(self):
    if self._dict_space:
      items = []
      for com in self._components:
        items += zip(com.tensor_names, com.space.spaces)
      return spaces.Dict(OrderedDict(['OPPO_'+k, v] for k, v in items))
    else:
      items = ()
      for com in self._components:
        items += com.space.spaces
      return spaces.Tuple(list(items))


class OppoObsComponentsV2(OppoObsComponentsV1):
  def __init__(self, game_version, dict_space, max_bo_count,
               max_bobt_count, zmaker_version, lib='lib5', **kwargs):
    if lib == 'lib5':
      from timitate.lib5.pb2feature_converter import GlobalFeatMaker, \
        ImmZStatMaker
    elif lib == 'lib6':
      from timitate.lib6.pb2feature_converter import GlobalFeatMaker, \
        ImmZStatMaker
    else:
      raise KeyError(f'lib={lib} not defined')
    self._dict_space = dict_space
    self._components = [GlobalFeatMaker(game_version),
                        ImmZStatMaker(max_bo_count=max_bo_count,
                                      max_bobt_count=max_bobt_count,
                                      zstat_version=zmaker_version)]


class SC2ZergOppoObsAsObs(OppoObsAsObs):
  """ SC2Zerg specific parse func V1; may design other versions in future """
  def __init__(self, env, dict_space, attachment):
    super(SC2ZergOppoObsAsObs, self).__init__(env)
    self._attachment = attachment
    self._dict_space = dict_space
    self._expand_obs_space()

  def _expand_obs_space(self):
    space_old = self.observation_space.spaces[self._me_id]
    if self._dict_space:
      assert isinstance(space_old, spaces.Dict)
      self.observation_space.spaces[self._me_id] = \
          spaces.Dict(OrderedDict(list(space_old.spaces.items()) +
                                  list(self._attachment.space.spaces.items())))
    else:
      assert not isinstance(space_old, spaces.Dict)
      if isinstance(self._attachment.space, spaces.Tuple):
          self.observation_space.spaces[self._me_id] = \
              spaces.Tuple(tuple(space_old.spaces) + tuple(self._attachment.space.spaces))
      else:
          self.observation_space.spaces[self._me_id] = \
              spaces.Tuple(tuple(space_old.spaces) + (self._attachment.space,))

  def _parse_oppo_obs(self, raw_oppo_obs):
    # raw_oppo_obs should be a timestep
    pb = raw_oppo_obs, None
    return self._attachment.make(pb)


class StatZStatDist(Wrapper):
  """Statistics for ZStat Distance between target and imm"""
  def step(self, actions):
    obs, reward, done, info = self.env.step(actions)
    if done:
      required_keys = ['Z_BUILD_ORDER', 'Z_BUILD_ORDER_COORD',
                       'IMM_BUILD_ORDER', 'IMM_BUILD_ORDER_COORD']
      for ind, ob in enumerate(obs):
        key = f'agt{ind}-zstat-dist'
        if isinstance(ob, dict) and set(required_keys) <= set(ob):
          if np.sum(ob['Z_BUILD_ORDER']) > 0:
            SeqLevDist = SeqLevDist_with_Coord(ob['Z_BUILD_ORDER'],
                                               ob['Z_BUILD_ORDER_COORD'])
            info[key] = SeqLevDist.lev_dist(ob['IMM_BUILD_ORDER'],
                                            ob['IMM_BUILD_ORDER_COORD'])
        else:
          print(f"WARN: cannot find the fields {required_keys} for agent{ind}")
    return obs, reward, done, info


# TODO(pengsun): strict base env creation and interface creation and remove
#  those create_sc2*_env functions
def make_sc2_base_env(n_players=2, step_mul=8, version='4.7.0',
                      screen_resolution=SCREEN_RESOLUTION, screen_ratio=1.33,
                      camera_width_world_units=24, map_name=DEFAULT_MAP_NAME,
                      replay_dir=None, use_pysc2_feature=True,
                      crop_to_playable_area=False):
  players = [sc2_env.Agent(sc2_env.Race.zerg) for _ in range(n_players)]
  if replay_dir:
    os.makedirs(replay_dir, exist_ok=True)
  save_replay_episodes = 1 if replay_dir else 0
  return SC2BaseEnv(
    players=players,
    agent_interface=AGENT_INTERFACE,
    map_name=map_name,
    screen_resolution=screen_resolution,
    screen_ratio=screen_ratio,
    camera_width_world_units=camera_width_world_units,
    visualize=VISUALIZE,
    step_mul=step_mul,
    disable_fog=DISABLE_FOG,
    score_index=SCORE_INDEX,
    score_multiplier=SCORE_MULTIPLIER,
    max_reset_num=MAX_RESET_NUM,
    max_steps_per_episode=MAX_STEPS_PER_EPISODE,
    version=version,
    replay_dir=replay_dir,
    save_replay_episodes=save_replay_episodes,
    use_pysc2_feature=use_pysc2_feature,
    crop_to_playable_area=crop_to_playable_area,
    minimap_resolution=(camera_width_world_units, screen_resolution),
  )


def make_sc2vsbot_base_env(n_players=2, step_mul=8, race="Z", bot_race='Z',
                           difficulty=0, version='4.7.0',
                           map_name=DEFAULT_MAP_NAME):
  players = [sc2_env.Agent(RACES[race]),
             sc2_env.Bot(RACES[bot_race], DIFFICULTIES[difficulty])]
  return SC2BaseEnv(
    players=players,
    agent_interface=AGENT_INTERFACE,
    map_name=map_name,
    screen_resolution=SCREEN_RESOLUTION,
    visualize=VISUALIZE,
    step_mul=step_mul,
    disable_fog=DISABLE_FOG,
    score_index=SCORE_INDEX,
    score_multiplier=SCORE_MULTIPLIER,
    max_reset_num=MAX_RESET_NUM,
    max_steps_per_episode=MAX_STEPS_PER_EPISODE,
    version=version
  )


def make_interface(auto_resource=False, n_action=1, action_mask=True,
                   keep_order=True, sub_actions=(0, 1, 2, 3, 16, 17, 18, 19),
                   scout=False, config_path=None, prod_step_mul=4,
                   combat_step_mul=4, res_step_mul=4, feat_step_mul=4):
  """ x_step_mul means each step is x_step_mul * step_mul game frames for x mgr,
   while micro action is still executed every step_mul game frames """
  inter = RawInt()
  inter = ZergDataInt(inter, config_path=config_path)
  inter = ZergProdActInt(inter, append=False, keep_order=keep_order,
                         action_mask=action_mask,
                         auto_supply=False, step_mul=prod_step_mul)
  inter = ZergResourceActInt(inter, append=True, auto_resource=auto_resource,
                             step_mul=res_step_mul)
  inter = ZergCombatActInt(inter, append=True, sub_actions=sub_actions,
                           step_mul=combat_step_mul)
  inter = ZergAreaObsInt(inter, override=True)
  inter = ZergNonspatialObsInt(inter, override=False,
                               use_features=(True, True, True, True, True),
                               n_action=n_action, step_mul=feat_step_mul)
  inter = ZergTechObsInt(inter, override=False)
  inter = ZergUnitProgObsInt(inter, override=False)
  if scout:
    inter = ZergScoutInt(inter)
  if action_mask:
    inter = AppendMaskInt(inter)
  inter = ConcatVecWrapper(inter)
  inter = ActDtypeInt(inter)
  return inter


def make_sc2full_interface():
  from arena.interfaces.sc2full_formal.obs_int import FullObsInt
  from arena.interfaces.sc2full_formal.act_int import FullActInt
  from arena.interfaces.sc2full_formal.noop_int import NoopMaskInt
  inter = RawInt()
  inter = FullObsInt(inter)
  inter = FullActInt(inter)
  inter = NoopMaskInt(inter, max_noop_dim=10)
  return inter


def make_sc2full_v2_interface():
  from arena.interfaces.sc2full_formal.obs_int import FullObsIntV2
  from arena.interfaces.sc2full_formal.act_int import FullActIntV2, CameraActInt, NoopActIntV2
  from arena.interfaces.sc2full_formal.noop_int import NoopMaskInt
  max_noop_num = 10
  inter = RawInt()
  inter = FullObsIntV2(inter)
  inter = FullActIntV2(inter, max_noop_num=max_noop_num)
  inter = NoopActIntV2(inter)
  inter = NoopMaskInt(inter, max_noop_dim=max_noop_num)
  inter = CameraActInt(inter)
  return inter


def make_sc2full_v3_interface():
  from arena.interfaces.sc2full_formal.obs_int import FullObsIntV3, LastTarTagAsObsV3
  from arena.interfaces.sc2full_formal.act_int import FullActIntV3, CameraActInt, NoopActIntV3
  from arena.interfaces.common import ActAsObsV2
  # can skip game_loop 1, 2, 4, ...
  noop_nums = (1, 2, 4, 8, 16, 32, 64, 128)
  inter = RawInt()
  inter = FullObsIntV3(inter)
  inter = LastTarTagAsObsV3(inter)
  inter = FullActIntV3(inter, max_noop_num=len(noop_nums))
  inter = ActAsObsV2(inter)
  inter = NoopActIntV3(inter, noop_nums=noop_nums)
  inter = CameraActInt(inter)
  return inter


def make_sc2full_nolastaction_v3_interface():
  from arena.interfaces.sc2full_formal.obs_int import FullObsIntV3
  from arena.interfaces.sc2full_formal.act_int import FullActIntV3, CameraActInt, NoopActIntV3
  from arena.interfaces.common import ActAsObsV2
  # can skip game_loop 1, 2, 4, ...
  noop_nums = (1, 2, 4, 8, 16, 32, 64, 128)
  inter = RawInt()
  inter = FullObsIntV3(inter)
  inter = FullActIntV3(inter, max_noop_num=len(noop_nums))
  #inter = ActAsObsV2(inter)
  inter = NoopActIntV3(inter, noop_nums=noop_nums)
  inter = CameraActInt(inter)
  return inter


def make_sc2full_v4_interface(zstat_data_src='', mmr=3500, dict_space=False):
  from arena.interfaces.sc2full_formal.obs_int import FullObsIntV4
  from arena.interfaces.sc2full_formal.act_int import FullActIntV4, CameraActInt, NoopActIntV4
  from arena.interfaces.raw_int import RawInt
  from arena.interfaces.common import ActAsObsV2
  noop_nums = [i+1 for i in range(128)]
  inter = RawInt()
  inter = FullObsIntV4(inter, zstat_data_src=zstat_data_src, mmr=mmr, dict_space=dict_space)
  inter = FullActIntV4(inter, max_noop_num=len(noop_nums), dict_space=dict_space)
  inter = ActAsObsV2(inter)
  inter = CameraActInt(inter)
  noop_func = lambda x: x['A_NOOP_NUM'] if dict_space else x[1]
  inter = NoopActIntV4(inter, noop_nums=noop_nums, noop_func=noop_func)
  return inter


def make_sc2full_v5_interface(zstat_data_src='', mmr=3500, max_bo_count=50,
                              dict_space=False):
  from arena.interfaces.sc2full_formal.obs_int import FullObsIntV5
  from arena.interfaces.sc2full_formal.act_int import FullActIntV4, CameraActInt, NoopActIntV4
  from arena.interfaces.raw_int import RawInt
  from arena.interfaces.common import ActAsObsV2
  noop_nums = [i+1 for i in range(128)]
  inter = RawInt()
  inter = FullObsIntV5(inter, zstat_data_src=zstat_data_src, mmr=mmr,
                       max_bo_count=max_bo_count, dict_space=dict_space)
  inter = FullActIntV4(inter, max_noop_num=len(noop_nums), dict_space=dict_space)
  inter = ActAsObsV2(inter)
  inter = CameraActInt(inter)
  noop_func = lambda x: x['A_NOOP_NUM'] if dict_space else x[1]
  inter = NoopActIntV4(inter, noop_nums=noop_nums, noop_func=noop_func)
  return inter


def make_sc2full_v6_interface(zstat_data_src='', mmr=3500, max_bo_count=50,
                              dict_space=False, correct_executors=True,
                              raw_selection=False, verbose=30,
                              zstat_presort_order_name=None):
  from arena.interfaces.sc2full_formal.obs_int import FullObsIntV5
  from arena.interfaces.sc2full_formal.act_int import FullActIntV5, CameraActInt, NoopActIntV4
  from arena.interfaces.raw_int import RawInt
  from arena.interfaces.common import ActAsObsV2
  noop_nums = [i+1 for i in range(128)]
  inter = RawInt()
  # this obs inter requires game core 4.10.0
  inter = FullObsIntV5(inter, zstat_data_src=zstat_data_src, mmr=mmr,
                       max_bo_count=max_bo_count, dict_space=dict_space,
                       zstat_presort_order_name=zstat_presort_order_name,
                       game_version='4.10.0')
  inter = FullActIntV5(inter, max_noop_num=len(noop_nums), dict_space=dict_space,
                       correct_executors=correct_executors, raw_selection=raw_selection,
                       verbose=verbose)
  inter = ActAsObsV2(inter)
  if not raw_selection:
    inter = CameraActInt(inter)
  noop_func = lambda x: x['A_NOOP_NUM'] if dict_space else x[1]
  inter = NoopActIntV4(inter, noop_nums=noop_nums, noop_func=noop_func)
  return inter


def make_sc2full_v7_interface(zstat_data_src='',
                              mmr=3500,
                              max_bo_count=50,
                              max_bobt_count=50,
                              dict_space=False,
                              correct_executors=True,
                              raw_selection=False,
                              verbose=30,
                              zstat_presort_order_name=None,
                              zmaker_version='v4',
                              delete_useless_selection=False,
                              **kwargs):
  from arena.interfaces.sc2full_formal.obs_int import FullObsIntV6, PostRuleMask
  from arena.interfaces.sc2full_formal.act_int import (FullActIntV5,
                                                       CameraActInt,
                                                       NoopActIntV4)
  from arena.interfaces.raw_int import RawInt
  from arena.interfaces.common import ActAsObsV2
  noop_nums = [i+1 for i in range(128)]
  inter = RawInt()
  # this obs inter requires game core 4.10.0
  inter = FullObsIntV6(inter, zstat_data_src=zstat_data_src,
                       mmr=mmr,
                       max_bo_count=max_bo_count,
                       max_bobt_count=max_bobt_count,
                       dict_space=dict_space,
                       zstat_presort_order_name=zstat_presort_order_name,
                       game_version='4.10.0',
                       zmaker_version=zmaker_version)
  if delete_useless_selection:
    assert dict_space, 'delete_useless_selection must be used with dict_space.'
    inter = PostRuleMask(inter)
  inter = FullActIntV5(inter, max_noop_num=len(noop_nums),
                       dict_space=dict_space,
                       correct_executors=correct_executors,
                       raw_selection=raw_selection,
                       verbose=verbose)
  inter = ActAsObsV2(inter)
  if not raw_selection:
    inter = CameraActInt(inter)
  noop_func = lambda x: x['A_NOOP_NUM'] if dict_space else x[1]
  inter = NoopActIntV4(inter, noop_nums=noop_nums, noop_func=noop_func)
  return inter


def make_sc2full_v8_interface(zstat_data_src='',
                              mmr=3500,
                              max_bo_count=50,
                              max_bobt_count=50,
                              dict_space=False,
                              verbose=0,
                              zstat_presort_order_name=None,
                              correct_pos_radius=2.0,
                              correct_building_pos=False,
                              zmaker_version='v4',
                              inj_larv_rule=False,
                              ban_zb_rule=False,
                              ban_rr_rule=False,
                              rr_food_cap=40,
                              zb_food_cap=10,
                              mof_lair_rule=False,
                              hydra_spire_rule=False,
                              overseer_rule=False,
                              expl_map_rule=False,
                              add_cargo_to_units=False,
                              output_map_size=(128, 128),
                              crop_to_playable_area=False,
                              ab_dropout_list=None,
                              **kwargs):
  from arena.interfaces.sc2full_formal.obs_int import FullObsIntV7
  from arena.interfaces.sc2full_formal.act_int import FullActIntV6, NoopActIntV4
  from arena.interfaces.raw_int import RawInt
  from arena.interfaces.sc2full_formal.obs_int import ActAsObsSC2
  noop_nums = [i+1 for i in range(128)]
  inter = RawInt()
  # this obs inter requires game core 4.10.0
  inter = FullObsIntV7(inter, zstat_data_src=zstat_data_src,
                       mmr=mmr,
                       max_bo_count=max_bo_count,
                       max_bobt_count=max_bobt_count,
                       dict_space=dict_space,
                       zstat_presort_order_name=zstat_presort_order_name,
                       game_version='4.10.0',
                       zmaker_version=zmaker_version,
                       inj_larv_rule=inj_larv_rule,
                       ban_zb_rule=ban_zb_rule,
                       ban_rr_rule=ban_rr_rule,
                       rr_food_cap=rr_food_cap,
                       zb_food_cap=zb_food_cap,
                       mof_lair_rule=mof_lair_rule,
                       hydra_spire_rule=hydra_spire_rule,
                       overseer_rule=overseer_rule,
                       expl_map_rule=expl_map_rule,
                       add_cargo_to_units=add_cargo_to_units,
                       output_map_resolution=output_map_size,
                       crop_to_playable_area=crop_to_playable_area,
                       ab_dropout_list=ab_dropout_list)
  inter = FullActIntV6(inter, max_noop_num=len(noop_nums),
                       correct_pos_radius=correct_pos_radius,
                       correct_building_pos=correct_building_pos,
                       map_resolution=output_map_size,
                       crop_to_playable_area=crop_to_playable_area,
                       dict_space=dict_space,
                       verbose=verbose)
  inter = ActAsObsSC2(inter)
  noop_func = lambda x: x['A_NOOP_NUM'] if dict_space else x[1]
  inter = NoopActIntV4(inter, noop_nums=noop_nums, noop_func=noop_func)
  return inter


def create_sc2_vanilla_env():
  env = make_sc2_base_env(n_players=2, step_mul=8)
  env = SC2ZergUnitProductVecReward(env, watched_ability_ids=[
    ABILITY_ID.TRAIN_ZERGLING.value,
    ABILITY_ID.TRAIN_BANELING.value,
    ABILITY_ID.TRAIN_QUEEN.value,
    ABILITY_ID.TRAIN_ROACH.value,
    ABILITY_ID.MORPH_RAVAGER.value,
    ABILITY_ID.TRAIN_INFESTOR.value,
    ABILITY_ID.TRAIN_HYDRALISK.value,
    ABILITY_ID.TRAIN_MUTALISK,
    ABILITY_ID.MORPH_BROODLORD.value,
    ABILITY_ID.TRAIN_CORRUPTOR.value,
  ])
  inter1 = make_interface(config_path='tleague.envs.sc2.dft_config',
                          prod_step_mul=4, combat_step_mul=4,
                          res_step_mul=4, feat_step_mul=4)
  inter2 = make_interface(config_path='tleague.envs.sc2.dft_config',
                          prod_step_mul=4, combat_step_mul=4,
                          res_step_mul=4, feat_step_mul=4)
  env = EnvIntWrapper(env, [inter1, inter2])
  env = StepMul(env, 4)
  return env


def create_sc2_env_micro(use_micro=True, unit_rwd=False):
  n_action = 10
  if use_micro:
    env = make_sc2_base_env(n_players=2, step_mul=8)
    inter1 = make_interface(config_path='tleague.envs.sc2.dft_config',
                            prod_step_mul=4, combat_step_mul=4,
                            res_step_mul=4, feat_step_mul=4,
                            n_action=n_action)
    inter2 = make_interface(config_path='tleague.envs.sc2.dft_config',
                            prod_step_mul=4, combat_step_mul=4,
                            res_step_mul=4, feat_step_mul=4,
                            n_action=n_action)
  else:
    env = make_sc2_base_env(n_players=2, step_mul=32)
    inter1 = make_interface(config_path='tleague.envs.sc2.dft_config',
                            prod_step_mul=1, combat_step_mul=1,
                            res_step_mul=1, feat_step_mul=1,
                            n_action=n_action)
    inter2 = make_interface(config_path='tleague.envs.sc2.dft_config',
                            prod_step_mul=1, combat_step_mul=1,
                            res_step_mul=1, feat_step_mul=1,
                            n_action=n_action)
  inter1 = HackActInt(inter1)
  inter2 = HackActInt(inter2)
  if unit_rwd:
    env = SC2ZergUnitProductVecReward(
      env,
      watched_ability_ids=[
        ABILITY_ID.TRAIN_ZERGLING.value,
        ABILITY_ID.TRAIN_BANELING.value,
        ABILITY_ID.TRAIN_QUEEN.value,
        ABILITY_ID.TRAIN_ROACH.value,
        ABILITY_ID.MORPH_RAVAGER.value,
        ABILITY_ID.TRAIN_INFESTOR.value,
        ABILITY_ID.TRAIN_HYDRALISK.value,
        ABILITY_ID.TRAIN_MUTALISK.value,
        ABILITY_ID.MORPH_BROODLORD.value,
        ABILITY_ID.TRAIN_CORRUPTOR.value,
        ABILITY_ID.TRAIN_ULTRALISK.value,
        ABILITY_ID.TRAIN_VIPER.value,
        ABILITY_ID.MORPH_LURKER.value,
      ],
      scale=[1.0 / 50, 1.0 / 50, 2.0 / 50,
             2.0 / 50, 2.0 / 50,
             2.0 / 50, 2.0 / 50, 2.0 / 50,
             2.0 / 50, 2.0 / 50,
             6.0 / 50, 3.0 / 50, 2.0 / 50]
    )
  env = VecRwd(env, append=True)
  env = EnvIntWrapper(env, [inter1, inter2])
  if use_micro:
    env = StepMul(env, 4)
  return env


def create_sc2vsbot_env_micro(use_micro=True, unit_rwd=False, difficulty=0):
  n_action = 10
  if use_micro:
    env = make_sc2vsbot_base_env(n_players=2, step_mul=8, difficulty=difficulty)
    inter1 = make_interface(config_path='tleague.envs.sc2.dft_config',
                            prod_step_mul=4, combat_step_mul=4,
                            res_step_mul=4, feat_step_mul=4,
                            n_action=n_action)
  else:
    env = make_sc2vsbot_base_env(n_players=2, step_mul=32, difficulty=difficulty)
    inter1 = make_interface(config_path='tleague.envs.sc2.dft_config',
                            prod_step_mul=1, combat_step_mul=1,
                            res_step_mul=1, feat_step_mul=1,
                            n_action=n_action)
  inter1 = HackActInt(inter1)
  if unit_rwd:
    env = SC2ZergUnitProductVecReward(
      env,
      watched_ability_ids=[
        ABILITY_ID.TRAIN_ZERGLING.value,
        ABILITY_ID.TRAIN_BANELING.value,
        ABILITY_ID.TRAIN_QUEEN.value,
        ABILITY_ID.TRAIN_ROACH.value,
        ABILITY_ID.MORPH_RAVAGER.value,
        ABILITY_ID.TRAIN_INFESTOR.value,
        ABILITY_ID.TRAIN_HYDRALISK.value,
        ABILITY_ID.TRAIN_MUTALISK.value,
        ABILITY_ID.MORPH_BROODLORD.value,
        ABILITY_ID.TRAIN_CORRUPTOR.value,
        ABILITY_ID.TRAIN_ULTRALISK.value,
        ABILITY_ID.TRAIN_VIPER.value,
        ABILITY_ID.MORPH_LURKER.value,
      ],
      scale=[1.0 / 50, 1.0 / 50, 2.0 / 50,
             2.0 / 50, 2.0 / 50,
             2.0 / 50, 2.0 / 50, 2.0 / 50,
             2.0 / 50, 2.0 / 50,
             6.0 / 50, 3.0 / 50, 2.0 / 50]
    )
  env = VecRwd(env, append=True)
  env = EnvIntWrapper(env, [inter1])
  if use_micro:
    env = StepMul(env, 4)
  return env


def create_sc2_battle_env():
  from arena.env.sc2_battle_env import create_sc2_battle_env as battle_sc2_env
  env = battle_sc2_env(agent_interface='feature',
                       map_name='5ImmortalNoReset',
                       visualize=False,
                       max_steps_per_episode=10000)
  return env


def create_sc2_battle_fixed_oppo_env():
  from arena.env.sc2_battle_env import create_sc2_battle_env_with_fixed_oppo \
    as battle_sc2_env
  env = battle_sc2_env(agent_interface='feature',
                       map_name='5ImmortalNoReset',
                       visualize=False,
                       max_steps_per_episode=10000)
  return env


def create_sc2full_formal_env(inter_fun,
                              inter_config=None,
                              vec_rwd=True,
                              unit_rwd=True,
                              astar_rwd=False,
                              astar_rwd_version='v1',
                              compensate_win=True,
                              early_term=True,
                              stat_action=True,
                              stat_zstat_fn=True,
                              version='4.7.0',
                              map_name=DEFAULT_MAP_NAME,
                              replay_dir=None,
                              step_mul=None,
                              centralized_value=False,
                              use_trt=False,
                              crop_to_playable_area=False,
                              stat_zstat_dist=True,
                              skip_noop=False):
  if inter_fun == make_sc2full_interface:
    # default to 8
    step_mul = 8 if step_mul is None else step_mul
    env = make_sc2_base_env(n_players=2, step_mul=step_mul, version=version,
                            map_name=map_name, replay_dir=replay_dir,
                            use_pysc2_feature=False)
  elif inter_fun == make_sc2full_v2_interface:
    # default to 8
    step_mul = 8 if step_mul is None else step_mul
    env = make_sc2_base_env(n_players=2, step_mul=step_mul, version=version,
                            screen_resolution=168, screen_ratio=0.905,
                            camera_width_world_units=152, map_name=map_name,
                            replay_dir=replay_dir,
                            use_pysc2_feature=False)
  elif inter_fun in [make_sc2full_v3_interface,
                     make_sc2full_nolastaction_v3_interface]:
    # default to 2
    step_mul = 2 if step_mul is None else step_mul
    env = make_sc2_base_env(n_players=2, step_mul=step_mul, version=version,
                            screen_resolution=168, screen_ratio=0.905,
                            camera_width_world_units=152, map_name=map_name,
                            replay_dir=replay_dir,
                            use_pysc2_feature=False)
  elif inter_fun in [make_sc2full_v4_interface, make_sc2full_v5_interface,
                     make_sc2full_v6_interface, make_sc2full_v7_interface,
                     make_sc2full_v8_interface]:
    # default to 1
    step_mul = 1 if step_mul is None else step_mul
    # TODO(pengsun): double-check the screen_xxx stuff
    env = make_sc2_base_env(n_players=2, step_mul=step_mul, version=version,
                            screen_resolution=168, screen_ratio=0.905,
                            camera_width_world_units=152, map_name=map_name,
                            replay_dir=replay_dir, use_pysc2_feature=False,
                            crop_to_playable_area=crop_to_playable_area)
  else:
    raise ValueError('Unknown interface fun {}'.format(inter_fun))
  if early_term:
    env = EarlyTerminate(env)
  if stat_action:
    env = StatAllAction(env)
  if unit_rwd:
    env = SC2ZergUnitProductVecReward(
      env,
      watched_ability_ids=[
        ABILITY_ID.TRAIN_ZERGLING.value,
        ABILITY_ID.TRAIN_BANELING.value,
        ABILITY_ID.TRAIN_QUEEN.value,
        ABILITY_ID.TRAIN_ROACH.value,
        ABILITY_ID.MORPH_RAVAGER.value,
        ABILITY_ID.TRAIN_INFESTOR.value,
        ABILITY_ID.TRAIN_HYDRALISK.value,
        ABILITY_ID.TRAIN_MUTALISK.value,
        ABILITY_ID.MORPH_BROODLORD.value,
        ABILITY_ID.TRAIN_CORRUPTOR.value,
        ABILITY_ID.TRAIN_ULTRALISK.value,
        ABILITY_ID.TRAIN_VIPER.value,
        ABILITY_ID.MORPH_LURKER.value,
      ],
      scale=[1.0 / 50, 1.0 / 50, 2.0 / 50,
             2.0 / 50, 2.0 / 50,
             2.0 / 50, 2.0 / 50, 2.0 / 50,
             2.0 / 50, 2.0 / 50,
             6.0 / 50, 3.0 / 50, 2.0 / 50]
    )
  if vec_rwd:
    env = VecRwd(env, append=True)
  if use_trt:
    # install Tower-Rush-Trick env wrapper
    from arena.wrappers.basic_env_wrapper import OppoTRTNoOut
    env = OppoTRTNoOut(env)

  # parse interface config
  inter_config = {} if inter_config is None else inter_config
  # really install the interfaces
  inter1 = inter_fun(**inter_config)
  inter2 = inter_fun(**inter_config)
  if skip_noop and inter_fun == make_sc2full_v8_interface:
    noop_func = lambda x: x['A_NOOP_NUM'] + 1
    env = SC2EnvIntWrapper(env, [inter1, inter2], noop_func)
  else:
    env = EnvIntWrapper(env, [inter1, inter2])

  # Note: add it AFTER EnvIntWrapper
  if stat_zstat_fn:
    env = StatZStatFn(env)

  if stat_zstat_dist:
    env = StatZStatDist(env)

  # install other possible env wrappers
  if astar_rwd:
    env = SC2ZergZStatVecReward(env, version=astar_rwd_version, compensate_win=compensate_win)
  if centralized_value:
    dict_space = False if 'dict_space' not in inter_config else inter_config['dict_space']
    if inter_fun in [make_sc2full_v6_interface]:
      env = SC2ZergOppoObsAsObs(env, dict_space, OppoObsComponentsV1(version, dict_space))
    elif inter_fun in [make_sc2full_v7_interface, make_sc2full_v8_interface]:
      zmaker_version = 'v4' if 'zmaker_version' not in inter_config else inter_config['zmaker_version']
      max_bo_count = 50 if 'max_bo_count' not in inter_config else inter_config['max_bo_count']
      max_bobt_count = 50 if 'max_bobt_count' not in inter_config else inter_config['max_bobt_count']
      lib = 'lib5' if inter_fun == make_sc2full_v7_interface else 'lib6'
      env = SC2ZergOppoObsAsObs(env, dict_space, OppoObsComponentsV2(
        version, dict_space, max_bo_count, max_bobt_count, zmaker_version, lib))
    else:
      raise NotImplemented('Unknown interface using centralized value.')
  return env


def create_sc2_env(arena_id, env_config=None, inter_config=None):
  # parsing env config
  env_config = {} if env_config is None else env_config
  difficulty = (0 if 'difficulty' not in env_config
                else env_config['difficulty'])
  replay_dir = (None if 'replay_dir' not in env_config
                else env_config['replay_dir'])
  step_mul = None if 'step_mul' not in env_config else env_config['step_mul']
  map_name = ('KairosJunction' if 'map_name' not in env_config
              else env_config['map_name'])
  use_trt = False if 'use_trt' not in env_config else env_config['use_trt']
  astar_rwd_version = ('v3' if 'astar_rwd_version' not in env_config
                       else env_config['astar_rwd_version'])
  compensate_win = (True if 'compensate_win' not in env_config
                    else env_config['compensate_win'])
  skip_noop = False if 'skip_noop' not in env_config else env_config['skip_noop']
  crop_to_playable_area = (False if 'crop_to_playable_area' not in env_config
                           else env_config['crop_to_playable_area'])
  early_term = (True if 'early_term' not in env_config
                else env_config['early_term'])  # enabling by default
  if arena_id == 'sc2':
    return create_sc2_vanilla_env()
  elif arena_id == 'sc2_with_micro':
    return create_sc2_env_micro(use_micro=True)
  elif arena_id == 'sc2_without_micro':
    return create_sc2_env_micro(use_micro=False)
  elif arena_id == 'sc2_unit_rwd_micro':
    return create_sc2_env_micro(use_micro=True, unit_rwd=True)
  elif arena_id == 'sc2_unit_rwd_no_micro':
    return create_sc2_env_micro(use_micro=False, unit_rwd=True)
  elif arena_id == 'sc2vsbot_unit_rwd_no_micro':

    return create_sc2vsbot_env_micro(use_micro=False, unit_rwd=True,
                                     difficulty=difficulty)
  elif arena_id == 'sc2_battle':
    return create_sc2_battle_env()
  elif arena_id == 'sc2_battle_fixed_oppo':
    return create_sc2_battle_fixed_oppo_env()
  elif arena_id == 'sc2full_formal':
    return create_sc2full_formal_env(map_name='AbyssalReef',
                                     early_term=early_term,
                                     inter_fun=make_sc2full_interface,
                                     version='4.7.0',
                                     replay_dir=replay_dir,
                                     step_mul=step_mul)
  elif arena_id == 'sc2full_formal2':
    return create_sc2full_formal_env(map_name='KairosJunction',
                                     early_term=early_term,
                                     inter_fun=make_sc2full_v2_interface,
                                     version='4.7.1',
                                     replay_dir=replay_dir,
                                     step_mul=step_mul)
  elif arena_id == 'sc2full_formal3':
    return create_sc2full_formal_env(map_name='KairosJunction',
                                     early_term=early_term,
                                     inter_fun=make_sc2full_v3_interface,
                                     version='4.7.1',
                                     replay_dir=replay_dir,
                                     step_mul=step_mul)
  elif arena_id == 'sc2full_formal_nolastaction3':
    return create_sc2full_formal_env(
      early_term=early_term,
      map_name='KairosJunction',
      inter_fun=make_sc2full_nolastaction_v3_interface,
      version='4.7.1',
      replay_dir=replay_dir,
      step_mul=step_mul
    )
  elif arena_id == 'sc2full_formal4':
    return create_sc2full_formal_env(
      inter_fun=make_sc2full_v4_interface,
      inter_config=inter_config,
      early_term=early_term,
      map_name=map_name,
      version='4.7.1',
      replay_dir=replay_dir,
      step_mul=1  # enforce step_mul=1 TODO(pengsun): double-check this
    )
  elif arena_id == 'sc2full_formal4_dict':
    inter_config = {} if inter_config is None else inter_config
    inter_config['dict_space'] = True
    return create_sc2full_formal_env(
      inter_fun=make_sc2full_v4_interface,
      inter_config=inter_config,
      vec_rwd=False,
      unit_rwd=False,
      astar_rwd=True,
      early_term=early_term,
      map_name=map_name,
      version='4.7.1',
      replay_dir=replay_dir,
      step_mul=1  # enforce step_mul=1 TODO(pengsun): double-check this
    )
  elif arena_id == 'sc2full_formal5_dict':
    inter_config = {} if inter_config is None else inter_config
    inter_config['dict_space'] = True
    return create_sc2full_formal_env(
      inter_fun=make_sc2full_v5_interface,
      inter_config=inter_config,
      vec_rwd=False,
      unit_rwd=False,
      astar_rwd=True,
      early_term=early_term,
      map_name=map_name,
      version='4.7.1',
      replay_dir=replay_dir,
      step_mul=1,  # enforce step_mul=1 TODO(pengsun): double-check this
      centralized_value=True,
      astar_rwd_version='v2',
    )
  elif arena_id == 'sc2full_formal6_dict':
    inter_config = {} if inter_config is None else inter_config
    inter_config['dict_space'] = True
    return create_sc2full_formal_env(
      inter_fun=make_sc2full_v6_interface,
      inter_config=inter_config,
      vec_rwd=False,
      unit_rwd=False,
      astar_rwd=True,
      early_term=early_term,
      map_name=map_name,
      version='4.10.0',  # required game core version 4.10.0
      replay_dir=replay_dir,
      step_mul=1,  # enforce step_mul=1 TODO(pengsun): double-check this
      centralized_value=True,
      astar_rwd_version='v2',
    )
  elif arena_id == 'sc2full_formal7_dict':
    inter_config = {} if inter_config is None else inter_config
    inter_config['dict_space'] = True
    return create_sc2full_formal_env(
      inter_fun=make_sc2full_v7_interface,
      inter_config=inter_config,
      vec_rwd=False,
      unit_rwd=False,
      astar_rwd=True,
      astar_rwd_version=astar_rwd_version,
      early_term=early_term,
      map_name=map_name,
      version='4.10.0',  # required game core version 4.10.0
      replay_dir=replay_dir,
      step_mul=1,  # enforce step_mul=1 TODO(pengsun): double-check this
      centralized_value=True,
      use_trt=use_trt
    )
  elif arena_id == 'sc2full_formal8_dict':
    inter_config = {} if inter_config is None else inter_config
    inter_config['dict_space'] = True
    return create_sc2full_formal_env(
      inter_fun=make_sc2full_v8_interface,
      inter_config=inter_config,
      vec_rwd=False,
      unit_rwd=False,
      astar_rwd=True,
      astar_rwd_version=astar_rwd_version,
      compensate_win=compensate_win,
      early_term=early_term,
      map_name=map_name,
      version='4.10.0',  # required game core version 4.10.0
      replay_dir=replay_dir,
      step_mul=1,  # enforce step_mul=1 TODO(pengsun): double-check this
      centralized_value=True,
      use_trt=use_trt,
      crop_to_playable_area=crop_to_playable_area,
      skip_noop=skip_noop,
    )
  else:
    raise Exception('Unknown arena_id {}'.format(arena_id))


def sc2_env_space(arena_id, env_config=None, inter_config=None):
  # parsing env config
  env_config = {} if env_config is None else env_config
  inter_config = {} if inter_config is None else inter_config
  if 'max_bo_count' not in inter_config:
    inter_config['max_bo_count'] = 50
  if 'max_bobt_count' not in inter_config:
    inter_config['max_bobt_count'] = 20
  if 'zmaker_version' not in inter_config:
    inter_config['zmaker_version'] = 'v5'
  centralized_value = True if 'centralized_value' not in env_config else env_config['centralized_value']
  import distutils
  version = _get_sc2_version()
  version_ge_414 = (distutils.version.LooseVersion(version)
                    >= distutils.version.LooseVersion('4.1.4'))
  if arena_id == 'sc2':
    n_action = 1  # num of historical actions in observation
    ac_space = spaces.MultiDiscrete((46 + version_ge_414, 3, 8, 2))
    ac_space.dtype = np.dtype(np.int32)
    ob_space = spaces.Box(low=-1, high=1, dtype=np.float32,
                          shape=[1685 + (n_action + 1) * version_ge_414, ])
  elif arena_id in ['sc2_with_micro', 'sc2_without_micro',
                    'sc2_unit_rwd_micro', 'sc2_unit_rwd_no_micro']:
    n_action = 10  # num of historical actions in observation
    ac_space = spaces.Tuple([spaces.Discrete(46 + version_ge_414)] +
                            [spaces.Discrete(3)] +
                            [spaces.MultiDiscrete((8, 2))])
    ac_space.spaces[0].dtype = np.dtype(np.int32)
    ac_space.spaces[1].dtype = np.dtype(np.int32)
    ac_space.spaces[2].dtype = np.dtype(np.int32)
    ob_space = spaces.Box(low=-1, high=1, dtype=np.float32,
                          shape=[2216 + (n_action + 1) * version_ge_414, ])
  elif arena_id in ['sc2_battle', 'sc2_battle_fixed_oppo']:
    ac_space = spaces.Tuple([spaces.Discrete(2),  # move / atk
                             spaces.Discrete(10),  # tar_unit, max num of units
                             spaces.Discrete(8),  # x
                             spaces.Discrete(8),  # y
                             spaces.Box(low=0, high=1, shape=(10,),
                                        dtype=np.int32),  # m_select
                             spaces.Box(low=0, high=1, shape=(10,),
                                        dtype=np.int32)  # a_select
                             ])
    ob_space = spaces.Tuple([
      spaces.Box(low=0, high=1, shape=(10, 6), dtype=np.float32),
      spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
      spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
      spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
    ])
  elif arena_id in ['sc2full_formal']:
    inter = make_sc2full_interface()
    ob_space = inter.observation_space
    ac_space = inter.action_space
  elif arena_id in ['sc2full_formal2']:
    inter = make_sc2full_v2_interface()
    ob_space = inter.observation_space
    ac_space = inter.action_space
  elif arena_id in ['sc2full_formal3']:
    inter = make_sc2full_v3_interface()
    ob_space = inter.observation_space
    ac_space = inter.action_space
  elif arena_id in ['sc2full_formal_nolastaction3']:
    inter = make_sc2full_nolastaction_v3_interface()
    ob_space = inter.observation_space
    ac_space = inter.action_space
  elif arena_id in ['sc2full_formal4']:
    inter = make_sc2full_v4_interface()
    ob_space = inter.observation_space
    ac_space = inter.action_space
  elif arena_id in ['sc2full_formal4_dict']:
    inter = make_sc2full_v4_interface(dict_space=True)
    ob_space = inter.observation_space
    ac_space = inter.action_space
  elif arena_id in ['sc2full_formal5_dict']:
    inter = make_sc2full_v5_interface(dict_space=True)
    ob_space = inter.observation_space
    ac_space = inter.action_space
    if centralized_value:
      com = OppoObsComponentsV1(game_version='4.7.1', dict_space=True)
      ob_space = spaces.Dict(OrderedDict(list(ob_space.spaces.items()) +
                                         list(com.space.spaces.items())))
  elif arena_id in ['sc2full_formal6_dict']:
    inter = make_sc2full_v6_interface(dict_space=True)
    ob_space = inter.observation_space
    ac_space = inter.action_space
    if centralized_value:
      com = OppoObsComponentsV1(game_version='4.10.0', dict_space=True)
      ob_space = spaces.Dict(OrderedDict(list(ob_space.spaces.items()) +
                                         list(com.space.spaces.items())))
  elif arena_id in ['sc2full_formal7_dict']:
    # TODO(pengsun): need interface_config here?? now it's hard-coding
    inter_config['dict_space'] = True
    inter = make_sc2full_v7_interface(
      **inter_config
    )
    ob_space = inter.observation_space
    ac_space = inter.action_space
    # TODO(pengsun): need interface_config here?? now it's hard-coding
    if centralized_value:
      com = OppoObsComponentsV2(game_version='4.10.0',
                                lib='lib5',
                                **inter_config)
      ob_space = spaces.Dict(OrderedDict(list(ob_space.spaces.items()) +
                                         list(com.space.spaces.items())))
  elif arena_id in ['sc2full_formal8_dict']:
    # TODO(pengsun): need interface_config here?? now it's hard-coding
    inter_config['dict_space'] = True
    inter = make_sc2full_v8_interface(
      **inter_config
    )
    ob_space = inter.observation_space
    ac_space = inter.action_space
    # TODO(pengsun): need interface_config here?? now it's hard-coding
    if centralized_value:
      com = OppoObsComponentsV2(game_version='4.10.0',
                                lib='lib6',
                                **inter_config)
      ob_space = spaces.Dict(OrderedDict(list(ob_space.spaces.items()) +
                                         list(com.space.spaces.items())))
  else:
    env = create_sc2_env(arena_id, env_config, inter_config)
    env.reset()
    ac_space = env.action_space.spaces[0]
    ob_space = env.observation_space.spaces[0]
    env.close()
  return ob_space, ac_space
