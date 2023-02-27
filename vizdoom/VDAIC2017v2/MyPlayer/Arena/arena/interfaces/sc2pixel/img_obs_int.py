""" Gym env wrappers """
from gym import spaces
from arena.utils.unit_util import find_units_by_tag
import numpy as np
import scipy.misc
from pysc2.lib.typeenums import UNIT_TYPEID
import copy
import math
from arena.interfaces.interface import Interface

class ImgObsInt(Interface):
    def __init__(self, inter, gameinfo):
        super(self.__class__, self).__init__(inter)
        self.gameinfo = gameinfo
        self.self_tag_2d = None

    def reset(self, obs, **kwargs):
        super(self.__class__, self).reset(obs, **kwargs)

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=1, shape=(8, 32, 32), dtype=np.float32)

    def obs_trans(self, raw_obs):
        """
        :param raw observation:
        :return:
        (minimap_size_x, minimap_size_y, channel_num)
        channels:
        1 - self_pos
        2 - enemy_pos
        3 - self_hp
        4 - enemy_hp
        5 - self_cd
        6 - enemy_cd
        # 7 - self_energy
        # 8 - enemy_energy
        9 - self_damage
        10 - enemy_damage
        """
        if self.inter:
            obs = self.inter.obs_trans(raw_obs)
        timestep = obs
        img_obs = []
        units = timestep.observation.raw_data.units
        player_relative = np.zeros(shape=(self.observation_space.shape[1],
                                          self.observation_space.shape[2]))

        self_tag_2d = np.zeros_like(player_relative, dtype=int)
        self_hp_2d = np.zeros_like(player_relative, dtype=float)
        self_cd_2d = np.zeros_like(player_relative, dtype=float)
        self_shield_2d = np.zeros_like(player_relative, dtype=float)
        self_damage_2d = np.zeros_like(player_relative, dtype=float)

        enemy_tag_2d = np.zeros_like(player_relative, dtype=int)
        enemy_hp_2d = np.zeros_like(player_relative, dtype=float)
        enemy_cd_2d = np.zeros_like(player_relative, dtype=float)
        enemy_shield_2d = np.zeros_like(player_relative, dtype=float)
        enemy_damage_2d = np.zeros_like(player_relative, dtype=float)

        for u in units:
            # Only for ImmortalZealot map
            if u.unit_type == UNIT_TYPEID.PROTOSS_PYLON.value:
                continue

            pos = {'x': u.pos.x, 'y': u.pos.y}
            hp = float(u.health) / u.health_max
            cd = float(u.weapon_cooldown)
            shield = float(u.shield) / (u.shield_max + 1e-9)
            damage = 1.0 if u.unit_type == UNIT_TYPEID.PROTOSS_IMMORTAL.value else 0.4
            feature_map_pos = self._convert_world_to_feature_map(pos)
            if u.alliance == 1:
                # self_pos_2d[feature_map_pos[0], feature_map_pos[1]] = 1
                self_tag_2d[feature_map_pos[0], feature_map_pos[1]] = u.tag
                self_hp_2d[feature_map_pos[0], feature_map_pos[1]] = hp
                self_cd_2d[feature_map_pos[0], feature_map_pos[1]] = cd
                self_shield_2d[feature_map_pos[0], feature_map_pos[1]] = shield
                self_damage_2d[feature_map_pos[0], feature_map_pos[1]] = damage
            elif u.alliance == 4:
                # enemy_pos_2d[feature_map_pos[0], feature_map_pos[1]] = 1
                enemy_tag_2d[feature_map_pos[0], feature_map_pos[1]] = u.tag
                enemy_hp_2d[feature_map_pos[0], feature_map_pos[1]] = hp
                enemy_cd_2d[feature_map_pos[0], feature_map_pos[1]] = cd
                enemy_shield_2d[feature_map_pos[0], feature_map_pos[1]] = shield
                enemy_damage_2d[feature_map_pos[0], feature_map_pos[1]] = damage

        # img_obs.append(self_pos_2d)
        # img_obs.append(enemy_pos_2d)
        img_obs.append(self_hp_2d)
        img_obs.append(enemy_hp_2d)
        img_obs.append(self_cd_2d)
        img_obs.append(enemy_cd_2d)
        img_obs.append(self_shield_2d)
        img_obs.append(enemy_shield_2d)
        img_obs.append(self_damage_2d)
        img_obs.append(enemy_damage_2d)

        img_obs = np.array(img_obs, dtype=float)
        # img_obs = np.ascontiguousarray(np.transpose(img_obs, (1, 2, 0)))
        # img_obs = copy.deepcopy(img_obs)

        # For action wrapper use
        self.self_tag_2d = self_tag_2d

        # scipy.misc.imsave('self_hp_2d.jpg', self_hp_2d)
        # scipy.misc.imsave('enemy_hp_2d.jpg', enemy_hp_2d)

        return img_obs

    def _convert_world_to_feature_map(self, world_pos):
        # image_width = self.gameinfo.options.feature_layer.minimap_resolution.x
        # image_height = self.gameinfo.options.feature_layer.minimap_resolution.y
        # map_width = self.gameinfo.start_raw.map_size.x
        # map_height = self.gameinfo.start_raw.map_size.y

        # Pixels always cover a square amount of world space. The scale is determined
        # by the largest axis of the map.
        # pixel_size = max(float(map_width) / image_width, float(map_height) / image_height)

        # Origin of world space is bottom left. Origin of image space is top left.
        # Upper left corner of the map corresponds to the upper left corner of the upper
        # left pixel of the feature layer.
        # image_origin_x = 0
        # image_origin_y = map_height
        # image_relative_x = world_pos['x'] - image_origin_x
        # image_relative_y = image_origin_y - world_pos['y']

        # image_x = int(image_relative_x / pixel_size)
        # image_y = int(image_relative_y / pixel_size)

        p0 = self.gameinfo.start_raw.playable_area.p0
        p1 = self.gameinfo.start_raw.playable_area.p1
        image_x = world_pos['x'] - p0.x
        image_y = world_pos['y'] - p0.y
        real_size_x = p1.x - p0.x
        real_size_y = p1.y - p0.y
        image_x = min(math.ceil(image_x / real_size_x * self.observation_space.shape[1]),
                      self.observation_space.shape[1] - 1)
        image_y = min(math.ceil(image_y / real_size_y * self.observation_space.shape[2]),
                      self.observation_space.shape[1] - 1)

        return [int(image_x), int(self.observation_space.shape[1] - 1 - image_y)]  # reverse
