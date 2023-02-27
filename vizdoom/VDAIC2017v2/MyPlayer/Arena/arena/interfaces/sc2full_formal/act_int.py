""" Gym env wrappers """
from copy import deepcopy

import numpy as np
from gym import spaces
from timitate.lib.action2pb_converter import Action2PBConverter
from timitate.lib.pb2action_converter import PB2ActionConverter
from timitate.utils.commands import cmd_camera, cmd_with_pos, cmd_with_tar, noop
from timitate.lib2.action2pb_converter import Action2PBConverter as Action2PBConverterV2
from timitate.lib2.pb2action_converter import PB2ActionConverter as PB2ActionConverterV2
from timitate.lib3.action2pb_converter import Action2PBConverter as Action2PBConverterV3
from timitate.lib3.pb2action_converter import PB2ActionConverter as PB2ActionConverterV3
from timitate.lib4.action2pb_converter import Action2PBConverter as Action2PBConverterV4
from timitate.lib5.action2pb_converter import Action2PBConverter as Action2PBConverterV5
from timitate.lib5.zstat_utils import UNIT_TYPEID as UNIT_TYPEIDV5
from timitate.lib6.action2pb_converter import Action2PBConverter as Action2PBConverterV6


from arena.interfaces.interface import Interface


class FullActInt(Interface):
    def __init__(self, inter):
        super(self.__class__, self).__init__(inter)
        self.gameinfo = None
        self.act2pb = Action2PBConverter()
        self.space = PB2ActionConverter().space
        self.map_r_max = 176
        self.map_c_max = 200
        self.noop_cnt = 0

    def reset(self, obs, **kwargs):
        super(self.__class__, self).reset(obs, **kwargs)

    @property
    def action_space(self):
        il_ac_space = self.space.spaces
        ac_space = spaces.Tuple([il_ac_space[0],
                                 il_ac_space[1],
                                 il_ac_space[2],
                                 il_ac_space[3],
                                 spaces.Discrete(il_ac_space[4].n * il_ac_space[5].n),
                                 il_ac_space[6],
                                 il_ac_space[7]
                                 ])
        ac_space.spaces[4].dtype = np.int32
        return ac_space

    def act_trans(self, action):
        if self.inter.pb:
            x, y = self._loc_to_xy(action[4])
            action_reog = [action[0], action[1], action[2], action[3],
                           x, y, action[5], action[6]]
            pb_action = self.act2pb.convert(action_reog, self.inter.pb)
            if len(pb_action) == 0:
                # action_reog[2] starts from 0, so cnt needs +1
                self.noop_cnt = deepcopy(action_reog[2]) + 1
            if self.noop_cnt > 0:
                self.noop_cnt -= 1
                return []
            return pb_action
        else:
            raise BaseException

    def _loc_to_xy(self, loc):
        r, c = self._loc_to_rc(loc)
        x, y = self._rc_to_xy(r, c)
        return x, y

    def _xy_to_loc(self, x, y):
        r, c = self._xy_to_rc(x, y)
        loc = self._rc_to_loc(r, c)
        return loc

    def _loc_to_rc(self, loc):
        r = loc // self.map_c_max
        c = loc % self.map_c_max
        return r, c

    def _rc_to_loc(self, r, c):
        loc = r * self.map_c_max + c
        return loc

    def _xy_to_rc(self, x, y):
        r = self.map_r_max - 1 - y
        c = x
        return r, c

    def _rc_to_xy(self, r, c):
        x = c
        y = self.map_r_max - 1 - r
        return x, y


class CameraActInt(Interface):
    # Move camera to center of map if it is not
    def __init__(self, inter, map_map_resolution=(152,168)):
        self.map_c_max, self.map_r_max = map_map_resolution
        super(CameraActInt, self).__init__(inter)

    def act_trans(self, act):
        act = self.inter.act_trans(act)
        if self.unwrapped()._obs.observation.raw_data.player.camera.x != self.map_c_max // 2 or \
              self.unwrapped()._obs.observation.raw_data.player.camera.y != self.map_r_max // 2:
            print('return camera center.')
            act = [cmd_camera(self.map_c_max // 2, self.map_r_max // 2)] + act
        return act


class FullActIntV2(Interface):
    def __init__(self, inter, map_resolution=(152, 168), max_noop_num=10):
        super(self.__class__, self).__init__(inter)
        self.act2pb = Action2PBConverterV2(map_padding_size=map_resolution)
        self.space = PB2ActionConverterV2(map_resolution=map_resolution,
                                          max_noop_num=max_noop_num).space

    @property
    def action_space(self):
        return self.space

    def act_trans(self, action):
        if self.inter.pb:
            pb_action = self.act2pb.convert(action, self.inter.pb)
        else:
            raise BaseException
        return [pb_action]


class NoopActIntV2(Interface):
    def __init__(self, inter):
        super(self.__class__, self).__init__(inter)
        self.noop_cnt = 0

    def act_trans(self, action):
        if action[0] == 0:
            # action_reog[1] starts from 0, so cnt needs +1
            self.noop_cnt = deepcopy(action[1]) + 1
        if self.noop_cnt > 0:
            self.noop_cnt -= 1
            return []
        else:
            return self.inter.act_trans(action)


class FullActIntV3(Interface):
    def __init__(self, inter, map_resolution=(152, 168), max_noop_num=8):
        super(self.__class__, self).__init__(inter)
        self.act2pb = Action2PBConverterV3(map_padding_size=map_resolution)
        self.space = PB2ActionConverterV3(map_resolution=map_resolution,
                                          max_noop_num=max_noop_num).space

    @property
    def action_space(self):
        return self.space

    def act_trans(self, action):
        if self.inter.pb:
            pb_action = self.act2pb.convert(action, self.inter.pb)
        else:
            raise BaseException
        return [pb_action]


class NoopActIntV3(Interface):
    def __init__(self, inter, noop_nums=(1,2,4,8,16,32,64,128)):
        super(self.__class__, self).__init__(inter)
        self.noop_nums = noop_nums
        self.target_game_loop = 0

    def obs_trans(self, obs):
        obs = self.inter.obs_trans(obs)
        game_loop = self.unwrapped()._obs.observation.game_loop
        if game_loop < self.target_game_loop:
            return None
        else:
            return obs

    def act_trans(self, action):
        game_loop = self.unwrapped()._obs.observation.game_loop
        if game_loop < self.target_game_loop:
            return []
        else:
            if action[0] == 0:
                self.target_game_loop = game_loop + self.noop_nums[int(action[1])]
            act = self.inter.act_trans(action)
            return act

    def reset(self, obs, **kwargs):
        super(NoopActIntV3, self).reset(obs, **kwargs)
        self.target_game_loop = 0


class FullActIntV4(Interface):
    def __init__(self, inter, map_resolution=(128, 128), max_noop_num=128, dict_space=False):
        super(self.__class__, self).__init__(inter)
        self.act2pb = Action2PBConverterV4(map_padding_size=map_resolution,
                                           max_noop_num=max_noop_num,
                                           dict_space=dict_space)

    @property
    def action_space(self):
        return self.act2pb.space

    def act_trans(self, action):
        if self.inter:
            action = self.inter.act_trans(action)
        if self.inter.pb:
            pb_action = self.act2pb.convert(action, self.inter.pb)
        else:
            raise BaseException
        return [pb_action]


class FullActIntV5(Interface):
    def __init__(self, inter, map_resolution=(128, 128), max_noop_num=128,
                 dict_space=False, correct_executors=True,
                 raw_selection=False, verbose=30):
        super(self.__class__, self).__init__(inter)
        self.act2pb = Action2PBConverterV5(map_padding_size=map_resolution,
                                           max_noop_num=max_noop_num,
                                           dict_space=dict_space,
                                           correct_executors=correct_executors,
                                           raw_selection=raw_selection,
                                           verbose=verbose)

    @property
    def action_space(self):
        return self.act2pb.space

    def act_trans(self, action):
        if self.inter:
            action = self.inter.act_trans(action)
        if self.inter.pb:
            pb_action = self.act2pb.convert(action, self.inter.pb)
        else:
            raise BaseException
        if isinstance(pb_action, list):
            return pb_action
        else:
            return [pb_action]


class NoopActIntV4(Interface):
    def __init__(self, inter, noop_nums=(i+1 for i in range(128)),
                 noop_func=lambda x: x[1]):
        super(self.__class__, self).__init__(inter)
        self.noop_nums = list(noop_nums)
        self.target_game_loop = 0
        self.noop_func = noop_func

    def obs_trans(self, obs):
        game_loop = obs.observation.game_loop
        if game_loop < self.target_game_loop:
            return None
        else:
            obs = self.inter.obs_trans(obs)
            return obs

    def act_trans(self, action):
        game_loop = self.unwrapped()._obs.observation.game_loop
        if game_loop < self.target_game_loop:
            return []
        else:
            self.target_game_loop = game_loop + self.noop_nums[int(self.noop_func(action))]
            act = self.inter.act_trans(action)
            return act

    def reset(self, obs, **kwargs):
        super(NoopActIntV4, self).reset(obs, **kwargs)
        self.target_game_loop = 0


class TRTActInt(Interface):
    # Tower rush trick (TRT) action interface, only for KairosJunction
    def __init__(self, inter):
        super(TRTActInt, self).__init__(inter)
        self.use_trt = False
        self._started_print = False
        self._completed_print = False
        self._executors = []
        self._n_drones_trt = 2
        self._base_pos = [(31.5, 140.5), (120.5, 27.5)]

    def _dist(self, x1, y1, x2, y2):
        return ((x1-x2)**2+(y1-y2)**2)**0.5

    def _ready_to_go(self):
        units = self.unwrapped()._obs.observation.raw_data.units
        drones = [u for u in units if u.alliance == 1 and
                  u.unit_type == UNIT_TYPEIDV5.ZERG_DRONE.value]
        if len(self._executors) >= self._n_drones_trt:
            # update executors' attributes
            for i, e in enumerate(self._executors):
                if e is None:
                    continue
                is_alive = False
                for d in drones:
                    if d.tag == e.tag:
                        self._executors[i] = d
                        is_alive = True
                        break
                if not is_alive:
                    self._executors[i] = None
            return True
        for u in units:
            # once self spawning pool is on building after 0.3 progresses
            if u.alliance == 1 and u.unit_type == UNIT_TYPEIDV5.ZERG_SPAWNINGPOOL.value \
              and u.build_progress > 0.3 and len(drones) > 0 \
              and len(self._executors) < self._n_drones_trt:
                for d in drones:
                    if d not in self._executors:
                        self._executors.append(d)
                        if len(self._executors) >= self._n_drones_trt:
                            return True
        return False

    def _mission_completed(self):
        # if there had been executors and now the executors have been eliminated (failed or success)
        if len(self._executors) == 0:
            return False
        for e in self._executors:
            if e is not None:
                return False
        return True

    def _target_pos(self):
        units = self.unwrapped()._obs.observation.raw_data.units
        self_h = [u for u in units if u.alliance == 1 and u.unit_type == UNIT_TYPEIDV5.ZERG_HATCHERY.value]
        if len(self_h) == 0:
            return 0, 0
        self_h0 = None
        for h in self_h:
            if min(self._dist(h.pos.x, h.pos.y,
                              self._base_pos[0][0], self._base_pos[0][1]),
                   self._dist(h.pos.x, h.pos.y,
                              self._base_pos[1][0], self._base_pos[1][1])) < 1:
                self_h0 = h
        if self_h0 is None:
            raise BaseException('Not KJ map in TRTActInt.')
        if self._dist(self_h0.pos.x, self_h0.pos.y,
                      self._base_pos[0][0], self._base_pos[0][1]) < \
           self._dist(self_h0.pos.x, self_h0.pos.y,
                      self._base_pos[1][0], self._base_pos[1][1]):
            return self._base_pos[1]
        else:
            return self._base_pos[0]

    def _drone_micro(self):
        def _build_spinecrawler(drone):
            detect_range = 10
            order_ab_ids = [o.ability_id for o in drone.orders]
            if 1166 not in order_ab_ids:
                random_r1 = np.random.random()
                random_r2 = np.random.random()
                build_pos = (random_r1*(drone.pos.x-detect_range/2)+
                             (1-random_r1)*(drone.pos.x+detect_range/2),
                             random_r2*(drone.pos.y-detect_range/2)+
                             (1-random_r2)*(drone.pos.y+detect_range/2))
                return cmd_with_pos(ability_id=1166,
                                    x=build_pos[0],
                                    y=build_pos[1],
                                    tags=[drone.tag],
                                    shift=False)
            else:
                return noop()

        def _move_to_tar(drone, tar_pos):
            order_ab_ids = [o.ability_id for o in drone.orders]
            if 1 not in order_ab_ids:
                return cmd_with_pos(ability_id=1,
                                    x=tar_pos[0],
                                    y=tar_pos[1],
                                    tags=[drone.tag],
                                    shift=False)
            else:
                return noop()

        def _atk_tar(drone, tar_tag):
            order_ab_ids = [o.ability_id for o in drone.orders]
            if 23 not in order_ab_ids:
                return cmd_with_tar(ability_id=23,
                                    target_tag=tar_tag,
                                    tags=[drone.tag],
                                    shift=False)
            else:
                return noop()

        units = self.unwrapped()._obs.observation.raw_data.units
        enemy_d = [u for u in units if u.alliance == 4 and
                   u.unit_type == UNIT_TYPEIDV5.ZERG_DRONE.value]
        pb_actions = []
        tar_pos = self._target_pos()
        for i, drone in enumerate(self._executors):
            if drone is None:
                continue
            if i % 2 == 0:
                if self._dist(drone.pos.x, drone.pos.y, tar_pos[0], tar_pos[1]) < 10:
                    # enemy hatchery in drone's detect range; why drone's detect_range = 0?
                    pb_actions.append(_build_spinecrawler(drone))
                else:
                    pb_actions.append(_move_to_tar(drone, tar_pos))
            else:
                if len(enemy_d) > 0:
                    # enemy drones in drone's detect range
                    enemy_d_tags = [u.tag for u in enemy_d]
                    pb_actions.append(_atk_tar(drone, min(enemy_d_tags)))
                else:
                    pb_actions.append(_move_to_tar(drone, tar_pos))
        return pb_actions

    def _trt_act(self):
        if self._ready_to_go():
            if not self._started_print:
                print('Launch tower rush trick.')
                self._started_print = True
            if not self._mission_completed():
                return self._drone_micro()
            else:
                if not self._completed_print:
                    print('Tower rush trick completed.')
                    self._completed_print = True
        return []

    def _get_cmd_tags(self, raw_acts):
        all_tags = []
        for a in raw_acts:
            if hasattr(a, 'action_raw') and hasattr(a.action_raw, 'unit_command') \
              and hasattr(a.action_raw.unit_command, 'unit_tags'):
                all_tags += a.action_raw.unit_command.unit_tags
        return all_tags

    def _remove_tags(self, ori_act, tags):
        if len(tags) == 0:
            return ori_act
        for a in ori_act:
            if hasattr(a, 'action_raw') and hasattr(a.action_raw, 'unit_command') \
              and hasattr(a.action_raw.unit_command, 'unit_tags'):
                a_tags = set(a.action_raw.unit_command.unit_tags)
                for t in tags:
                    if t in a_tags:
                        a_tags.remove(t)
                # protobuff repeated only support pop()
                while len(a.action_raw.unit_command.unit_tags) > 0:
                    a.action_raw.unit_command.unit_tags.pop()
                for t in a_tags:
                    a.action_raw.unit_command.unit_tags.append(t)
        return ori_act

    def act_trans(self, act):
        # TODO: assert act is raw_action
        ori_act = act
        if self.use_trt:
            trt_act = self._trt_act()
            trt_a_tags = self._get_cmd_tags(trt_act)
            ori_act = self._remove_tags(ori_act, trt_a_tags)
            # Note: the order of the added items matters;
            # if ori_act is placed before trt_act, the remained selection is determined
            # by trt_act and then the model will be confused.
            act = trt_act + ori_act
        return act

    def reset(self, obs, **kwargs):
        super(TRTActInt, self).reset(obs, **kwargs)
        if 'use_trt' in kwargs:
            self.use_trt = kwargs['use_trt']
        else:
            self.use_trt = False
        self._started_print = False
        self._completed_print = False
        self._executors = []


class FullActIntV6(Interface):
    def __init__(self, inter, map_resolution=(128, 128), max_noop_num=128,
                 max_unit_num=600, correct_pos_radius=2.0, dict_space=False,
                 correct_building_pos=False, crop_to_playable_area=False,
                 verbose=30):
        super(self.__class__, self).__init__(inter)
        self.act2pb = Action2PBConverterV6(map_padding_size=map_resolution,
                                           max_noop_num=max_noop_num,
                                           dict_space=dict_space,
                                           verbose=verbose,
                                           max_unit_num=max_unit_num,
                                           correct_building_pos=correct_building_pos,
                                           crop_to_playable_area=crop_to_playable_area,
                                           correct_pos_radius=correct_pos_radius)

    @property
    def action_space(self):
        return self.act2pb.space

    def act_trans(self, action):
        if self.inter:
            action = self.inter.act_trans(action)
        if self.inter.pb:
            pb_action = self.act2pb.convert(action, self.inter.pb)
        else:
            raise BaseException
        if isinstance(pb_action, list):
            return pb_action
        else:
            return [pb_action]

    def reset(self, obs, **kwargs):
        self.act2pb.reset(obs, **kwargs)
        super(FullActIntV6, self).reset(obs, **kwargs)
