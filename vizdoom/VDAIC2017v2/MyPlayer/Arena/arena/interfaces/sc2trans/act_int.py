""" Gym env wrappers """
from gym import spaces
from arena.utils.unit_util import collect_units_by_alliance
from arena.utils.unit_util import find_units_by_tag
from arena.utils.constant import AllianceType
from arena.utils.dist_util import find_nearest
from arena.utils.action_util import hold_pos
from arena.utils.action_util import attack_unit, move_pos
from arena.interfaces.interface import Interface
import numpy as np


class PtrMiniActInt(Interface):
    def __init__(self, inter, gameinfo):
        super(self.__class__, self).__init__(inter)
        self.gameinfo = gameinfo

    def reset(self, obs, **kwargs):
        super(self.__class__, self).reset(obs, **kwargs)

    @property
    def action_space(self):
        return spaces.Tuple([spaces.Discrete(2),  # move / atk
                             spaces.Discrete(10),  # tar_unit, max number of units
                             spaces.Discrete(8),  # x
                             spaces.Discrete(8),  # y
                             spaces.Box(low=0, high=1, shape=(10,), dtype=np.int32),  # m_select
                             spaces.Box(low=0, high=1, shape=(10,), dtype=np.int32)  # a_select
                             ])

    def act_trans(self, action):
        self_tag_list = self.inter.self_tag
        all_tag_list = self.inter.all_tag
        ability, tar_unit, tar_loc_x, tar_loc_y, m_select, a_select = action
        if not isinstance(ability, int):
            ability = ability[0]
            tar_unit = tar_unit[0]
            tar_loc_x = tar_loc_x[0]
            tar_loc_y = tar_loc_y[0]
            m_select = m_select[0]
            a_select = a_select[0]

        m_nnz_idx = np.nonzero(m_select)
        m_nnz_idx = m_nnz_idx[0]
        a_nnz_idx = np.nonzero(a_select)
        a_nnz_idx = a_nnz_idx[0]

        raw_actions = []
        if ability == 0:  # move
            if len(m_nnz_idx) == 0:
                return []

            move_agents = []
            for i in m_nnz_idx:
                if i >= len(self_tag_list):
                    # print('WARN: selected unit idx not in self units')
                    continue
                else:
                    move_agents.append(self_tag_list[i])
            raw_actions.append(move_pos(u=move_agents,
                                        pos=self._convert_feature_to_world_map(
                                            [tar_loc_x, tar_loc_y]
                                        )))
        else:  # attack
            if len(a_nnz_idx) == 0:
                return []

            atk_agents = []
            for i in a_nnz_idx:
                if i >= len(self_tag_list):
                    # print('WARN: selected unit idx not in self units')
                    continue
                else:
                    atk_agents.append(self_tag_list[i])

            if tar_unit < len(all_tag_list):
                raw_actions.append(
                    attack_unit(u=atk_agents,
                                target_unit=all_tag_list[tar_unit]))
            else:
                pass
                # print('WARN: selected tar unit idx not in enemy units')

        return raw_actions

    def _convert_feature_to_world_map(self, feature_pos):
        p0 = self.gameinfo.start_raw.playable_area.p0
        p1 = self.gameinfo.start_raw.playable_area.p1

        real_size_x = p1.x - p0.x
        real_size_y = p1.y - p0.y

        image_x = feature_pos[1]
        image_y = 8 - feature_pos[0]

        image_x = image_x / 8.0 * real_size_x
        image_y = image_y / 8.0 * real_size_y

        world_pos_x = image_x + p0.x
        world_pos_y = image_y + p0.y

        return [world_pos_x, world_pos_y]


class TransMARLActInt(Interface):
    def __init__(self, inter, gameinfo):
        super(self.__class__, self).__init__(inter)
        self.gameinfo = gameinfo

    def reset(self, obs, **kwargs):
        super(self.__class__, self).reset(obs, **kwargs)

    @property
    def action_space(self):
        return spaces.MultiDiscrete(nvec=tuple([9 for _ in range(10)]))

    def act_trans(self, action):
        self_tag_list = self.inter.self_tag
        actions = []

        for i, tag in enumerate(self_tag_list):
            actions.append(self._discrete_action(tag, action[i]))

        return actions

    def _discrete_action(self, unit_tag, act_id):
        timestep = self.unwrapped()._obs
        units = timestep.observation.raw_data.units
        u = find_units_by_tag(units, unit_tag)
        if not u:
            return []
        u = u[0]

        pos_x, pos_y = u.pos.x, u.pos.y
        dx, dy = [0, 0, -1, 1, -1, 1, 1, -1], [1, -1, 0, 0, 1, 1, -1, -1]
        pos = [pos_x + dx[act_id], pos_y + dy[act_id]]
        action = move_pos(u, pos)

        return action


class TransTeamActInt(Interface):
    def __init__(self, inter, gameinfo):
        super(self.__class__, self).__init__(inter)
        self.gameinfo = gameinfo

    def reset(self, obs, **kwargs):
        super(self.__class__, self).reset(obs, **kwargs)

    @property
    def action_space(self):
        return spaces.Tuple([
            spaces.Discrete(9),
            spaces.Discrete(9),
        ])

    def act_trans(self, action):
        zealot_tags = self.inter.zealot_tags
        immortal_tags = self.inter.immortal_tags

        zealot_action = action[0]
        immortal_action = action[1]

        raw_actions = []

        for i, tag in enumerate(zealot_tags):
            if tag != 0:
                raw_actions.append(self._discrete_action(tag, zealot_action[i]))

        for i, tag in enumerate(immortal_tags):
            if tag != 0:
                raw_actions.append(self._discrete_action(tag, immortal_action[i]))

        return raw_actions

    def _discrete_action(self, unit_tag, act_id):
        timestep = self.unwrapped()._obs
        units = timestep.observation.raw_data.units
        u = find_units_by_tag(units, unit_tag)
        if not u:
            return []
        u = u[0]

        pos_x, pos_y = u.pos.x, u.pos.y
        dx, dy = [0, 0, -1, 1, -1, 1, 1, -1], [1, -1, 0, 0, 1, 1, -1, -1]
        pos = [pos_x + dx[act_id], pos_y + dy[act_id]]
        action = move_pos(u, pos)

        return action
