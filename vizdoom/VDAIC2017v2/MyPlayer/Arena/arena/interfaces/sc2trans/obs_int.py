""" Gym env wrappers """
from gym import spaces
import numpy as np
from pysc2.lib.typeenums import UNIT_TYPEID
import math
from arena.interfaces.interface import Interface


class ListMiniObsInt(Interface):
    def __init__(self, inter, gameinfo, max_list_len=10):
        super(self.__class__, self).__init__(inter)
        self.gameinfo = gameinfo
        self.max_list_len = max_list_len
        self.obs_spec = spaces.Tuple([
            spaces.Box(low=0, high=1, shape=(self.max_list_len, 6), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.max_list_len,), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.max_list_len,), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.max_list_len,), dtype=np.float32),
        ])
        self.self_tag = None
        self.all_tag = None

    def reset(self, obs, **kwargs):
        super(self.__class__, self).reset(obs, **kwargs)

    @property
    def observation_space(self):
        return spaces.Tuple([
            spaces.Box(low=0, high=1, shape=(self.max_list_len, 6), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.max_list_len,), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.max_list_len,), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.max_list_len,), dtype=np.float32),
        ])

    def obs_trans(self, raw_obs):
        """
        :param raw observation:
        :return: a list of units
        """
        if self.inter:
            obs = self.inter.obs_trans(raw_obs)
        else:
            obs = raw_obs

        timestep = obs
        units = timestep.observation.raw_data.units

        self_tag = []
        self_list = []
        enemy_tag = []
        enemy_list = []
        masks = {}

        for u in units:
            # Only for ImmortalZealot map
            if u.unit_type == UNIT_TYPEID.PROTOSS_PYLON.value:
                continue

            pos = {'x': u.pos.x, 'y': u.pos.y}
            hp = float(u.health) / u.health_max
            cd = float(u.weapon_cooldown) / 100.0
            shield = float(u.shield) / (u.shield_max + 1e-9)
            # damage = 1.0 if u.unit_type == UNIT_TYPEID.PROTOSS_IMMORTAL.value else 0.4
            feature_map_pos = self._convert_world_to_feature_map(pos)
            if u.alliance == 1:
                self_tag.append(u.tag)
                feat = [hp, cd, shield, feature_map_pos[0] / 8.0, feature_map_pos[1] / 8.0, 0]
                self_list.append(feat)
            elif u.alliance == 4:
                enemy_tag.append(u.tag)
                feat = [hp, cd, shield, feature_map_pos[0] / 8.0, feature_map_pos[1] / 8.0, 1]
                enemy_list.append(feat)

        list_obs = self_list + enemy_list
        self_mask = [1 for _ in range(len(self_list))] + [0 for _ in range(len(enemy_list))]
        enemy_mask = [0 for _ in range(len(self_list))] + [1 for _ in range(len(enemy_list))]
        len_mask = [1 for _ in range(len(self_list)+len(enemy_list))]
        while len(list_obs) < self.max_list_len:
            list_obs.append([0 for _ in range(6)])
            self_mask.append(0)
            enemy_mask.append(0)
            len_mask.append(0)

        list_obs = np.array(list_obs, dtype=float)
        self_mask = np.array(self_mask, dtype=float)
        enemy_mask = np.array(enemy_mask, dtype=float)
        len_mask = np.array(len_mask, dtype=float)
        # masks['len'] = len_mask
        # masks['target'] = enemy_mask
        # masks['select'] = self_mask

        # For action wrapper use
        self.self_tag = self_tag
        self.all_tag = self_tag + enemy_tag

        return list_obs, len_mask, enemy_mask, self_mask

    def _convert_world_to_feature_map(self, world_pos):
        p0 = self.gameinfo.start_raw.playable_area.p0
        p1 = self.gameinfo.start_raw.playable_area.p1
        image_x = world_pos['x'] - p0.x
        image_y = world_pos['y'] - p0.y
        real_size_x = p1.x - p0.x
        real_size_y = p1.y - p0.y
        image_x = min(math.ceil(image_x / real_size_x * 8), 8 - 1)
        image_y = min(math.ceil(image_y / real_size_y * 8), 8 - 1)

        x_rotate = 8 - image_y
        y_rotate = image_x
        return [int(x_rotate), int(y_rotate)]


class TransMARLObsInt(Interface):
    def __init__(self, inter, gameinfo, max_list_len=10):
        super(self.__class__, self).__init__(inter)
        self.gameinfo = gameinfo
        self.max_list_len = max_list_len
        self.obs_spec = spaces.Tuple([
            spaces.Box(low=0, high=1, shape=(self.max_list_len, 6), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.max_list_len,), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.max_list_len,), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.max_list_len,), dtype=np.float32),
        ])
        self.self_tag = None
        self.all_tag = None

    def reset(self, obs, **kwargs):
        super(self.__class__, self).reset(obs, **kwargs)

    @property
    def observation_space(self):
        return spaces.Tuple([
            spaces.Box(low=0, high=1, shape=(self.max_list_len, 6), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.max_list_len,), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.max_list_len,), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.max_list_len,), dtype=np.float32),
        ])

    def obs_trans(self, raw_obs):
        """
        :param raw observation:
        :return: a list of units
        """
        if self.inter:
            obs = self.inter.obs_trans(raw_obs)
        else:
            obs = raw_obs

        timestep = obs
        units = timestep.observation.raw_data.units

        self_tag = []
        self_list = []

        for u in units:
            # Only for mineral map
            if u.unit_type != UNIT_TYPEID.TERRAN_MARINE.value:
                continue

            pos = {'x': u.pos.x, 'y': u.pos.y}
            feature_map_pos = self._convert_world_to_feature_map(pos)
            if u.alliance == 1:
                self_tag.append(u.tag)
                feat = [feature_map_pos[0] / 8.0, feature_map_pos[1] / 8.0]
                self_list.append(feat)

        list_obs = self_list + enemy_list
        self_mask = [1 for _ in range(len(self_list))] + [0 for _ in range(len(enemy_list))]
        enemy_mask = [0 for _ in range(len(self_list))] + [1 for _ in range(len(enemy_list))]
        len_mask = [1 for _ in range(len(self_list)+len(enemy_list))]
        while len(list_obs) < self.max_list_len:
            list_obs.append([0 for _ in range(6)])
            self_mask.append(0)
            enemy_mask.append(0)
            len_mask.append(0)

        list_obs = np.array(list_obs, dtype=float)
        self_mask = np.array(self_mask, dtype=float)
        enemy_mask = np.array(enemy_mask, dtype=float)
        len_mask = np.array(len_mask, dtype=float)
        # masks['len'] = len_mask
        # masks['target'] = enemy_mask
        # masks['select'] = self_mask

        # For action wrapper use
        self.self_tag = self_tag
        self.all_tag = self_tag + enemy_tag

        return list_obs, len_mask, enemy_mask, self_mask

    def _convert_world_to_feature_map(self, world_pos):
        p0 = self.gameinfo.start_raw.playable_area.p0
        p1 = self.gameinfo.start_raw.playable_area.p1
        image_x = world_pos['x'] - p0.x
        image_y = world_pos['y'] - p0.y
        real_size_x = p1.x - p0.x
        real_size_y = p1.y - p0.y
        image_x = min(math.ceil(image_x / real_size_x * 8), 8 - 1)
        image_y = min(math.ceil(image_y / real_size_y * 8), 8 - 1)

        x_rotate = 8 - image_y
        y_rotate = image_x
        return [int(x_rotate), int(y_rotate)]


class TransTeamObsInt(Interface):
    def __init__(self, inter, gameinfo):
        """ use 3I2Z map """
        super(self.__class__, self).__init__(inter)
        self.gameinfo = gameinfo
        self.max_num_zealot = 2
        self.max_num_immortal = 3
        self.feat_dim = 5

        self.obs_spec = spaces.Tuple([
            spaces.Box(low=0, high=1, shape=(self.max_num_zealot, self.feat_dim), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.max_num_immortal, self.feat_dim), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(8, 8, 2*2), dtype=np.float32),
        ])
        self.zealot_tags = None
        self.immortal_tags = None

    def reset(self, obs, **kwargs):
        super(self.__class__, self).reset(obs, **kwargs)

    @property
    def observation_space(self):
        return spaces.Tuple([
            spaces.Box(low=0, high=1, shape=(self.max_num_zealot, self.feat_dim), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(self.max_num_immortal, self.feat_dim), dtype=np.float32),
            spaces.Box(low=0, high=1, shape=(8, 8, 2*2), dtype=np.float32),
        ])

    def obs_trans(self, raw_obs):
        """
        :param raw observation:
        :return: a list of units
        """
        if self.inter:
            obs = self.inter.obs_trans(raw_obs)
        else:
            obs = raw_obs

        timestep = obs
        units = timestep.observation.raw_data.units

        zealot_tags = []
        immortal_tags = []

        zealot_team_feats = []
        immortal_team_feats = []

        global_img_feat = np.zeros([8, 8, 2+4*2])  # enemy needs hp, cd, shield, pos (pos is not necessary but used for discriminating HWC)

        for u in units:
            if u.unit_type not in [UNIT_TYPEID.PROTOSS_ZEALOT.value,
                                   UNIT_TYPEID.PROTOSS_IMMORTAL.value]:
                continue

            if u.alliance == 1:
                if u.unit_type == UNIT_TYPEID.PROTOSS_ZEALOT.value:
                    feat, pos = self._get_u_feats(u)
                    zealot_tags.append(u.tag)
                    zealot_team_feats.append(feat)
                    global_img_feat[pos[0], pos[1], 0] = 1
                if u.unit_type == UNIT_TYPEID.PROTOSS_IMMORTAL.value:
                    feat, pos = self._get_u_feats(u)
                    immortal_tags.append(u.tag)
                    immortal_team_feats.append(feat)
                    global_img_feat[pos[0], pos[1], 1] = 1

            if u.alliance == 4:
                if u.unit_type == UNIT_TYPEID.PROTOSS_ZEALOT.value:
                    feat, pos = self._get_u_feats(u)
                    global_img_feat[pos[0], pos[1], 2] = feat[0]
                    global_img_feat[pos[0], pos[1], 3] = feat[1]
                    global_img_feat[pos[0], pos[1], 4] = feat[2]
                if u.unit_type == UNIT_TYPEID.PROTOSS_IMMORTAL.value:
                    feat, pos = self._get_u_feats(u)
                    global_img_feat[pos[0], pos[1], 5] = feat[0]
                    global_img_feat[pos[0], pos[1], 6] = feat[1]
                    global_img_feat[pos[0], pos[1], 7] = feat[2]

        zealot_mask = [1 for _ in range(len(zealot_tags))]
        immortal_mask = [1 for _ in range(len(immortal_tags))]

        while len(zealot_tags) < self.max_num_zealot:
            zealot_team_feats.append([0 for _ in range(self.feat_dim)])
            zealot_mask.append(0)
            zealot_tags.append(0)

        while len(immortal_tags) < self.max_num_immortal:
            immortal_team_feats.append([0 for _ in range(self.feat_dim)])
            immortal_mask.append(0)
            immortal_tags.append(0)

        zealot_team_feats = np.array(zealot_team_feats, dtype=np.float32)
        zealot_mask = np.array(zealot_mask, dtype=np.float32)

        immortal_team_feats = np.array(immortal_team_feats, dtype=np.float32)
        immortal_mask = np.array(immortal_mask, dtype=np.float32)

        # For action wrapper use
        self.zealot_tags = zealot_tags
        self.immortal_tags = immortal_tags

        team_x_list = [zealot_team_feats, immortal_team_feats]
        team_masks = [zealot_mask, immortal_mask]

        return team_x_list, team_masks

    def _get_u_feats(self, u):
        pos = {'x': u.pos.x, 'y': u.pos.y}
        feature_map_pos = self._convert_world_to_feature_map(pos)
        hp = float(u.health) / u.health_max
        cd = float(u.weapon_cooldown) / 100.0
        shield = float(u.shield) / (u.shield_max + 1e-9)
        feat = [hp, cd, shield, feature_map_pos[0] / 8.0, feature_map_pos[1] / 8.0]
        return feat, feature_map_pos

    def _convert_world_to_feature_map(self, world_pos):
        p0 = self.gameinfo.start_raw.playable_area.p0
        p1 = self.gameinfo.start_raw.playable_area.p1
        image_x = world_pos['x'] - p0.x
        image_y = world_pos['y'] - p0.y
        real_size_x = p1.x - p0.x
        real_size_y = p1.y - p0.y
        image_x = min(math.ceil(image_x / real_size_x * 8), 8 - 1)
        image_y = min(math.ceil(image_y / real_size_y * 8), 8 - 1)

        x_rotate = 8 - image_y
        y_rotate = image_x
        return [int(x_rotate), int(y_rotate)]
