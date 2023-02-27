from pysc2.lib.typeenums import UNIT_TYPEID
from pysc2.lib.typeenums import UPGRADE_ID
from pysc2.lib.typeenums import ABILITY_ID
from s2clientprotocol import sc2api_pb2 as sc_pb


def move_pos(u, pos):
    """
    :param u: either an unit or a list of unit tags
    :param pos: pos[0], pos[1]
    :return: raw action
    """
    action = sc_pb.Action()
    action.action_raw.unit_command.ability_id = ABILITY_ID.MOVE.value
    action.action_raw.unit_command.target_world_space_pos.x = pos[0]
    action.action_raw.unit_command.target_world_space_pos.y = pos[1]
    if isinstance(u, list):
        for u_tag in u:
            action.action_raw.unit_command.unit_tags.append(u_tag)
    else:
        action.action_raw.unit_command.unit_tags.append(u.tag)
    return action


def attack_pos(u, pos):
    """
    :param u: either an unit or a list of unit tags
    :param pos: pos[0], pos[1]
    :return: raw action
    """
    action = sc_pb.Action()
    action.action_raw.unit_command.ability_id = ABILITY_ID.ATTACK_ATTACK.value
    action.action_raw.unit_command.target_world_space_pos.x = pos[0]
    action.action_raw.unit_command.target_world_space_pos.y = pos[1]
    if isinstance(u, list):
        for u_tag in u:
            action.action_raw.unit_command.unit_tags.append(u_tag)
    else:
        action.action_raw.unit_command.unit_tags.append(u.tag)
    return action


def attack_unit(u, target_unit):
    """
    :param u: either an unit or a list of unit tags
    :param target_unit: either a target unit or a target unit_tag (int)
    :return: raw action
    """
    action = sc_pb.Action()
    action.action_raw.unit_command.ability_id = ABILITY_ID.ATTACK_ATTACK.value
    if hasattr(target_unit, 'tag'):
        action.action_raw.unit_command.target_unit_tag = target_unit.tag
    else:
        action.action_raw.unit_command.target_unit_tag = target_unit
    if isinstance(u, list):
        for u_tag in u:
            action.action_raw.unit_command.unit_tags.append(u_tag)
    else:
        action.action_raw.unit_command.unit_tags.append(u.tag)
    return action


def attack_unitpos(u, target_unit):
    action = sc_pb.Action()
    action.action_raw.unit_command.ability_id = ABILITY_ID.ATTACK_ATTACK.value
    action.action_raw.unit_command.target_world_space_pos = target_unit.pos
    action.action_raw.unit_command.unit_tags.append(u.tag)
    return action


def hold_pos(u):
    action = sc_pb.Action()
    action.action_raw.unit_command.ability_id = ABILITY_ID.HOLDPOSITION.value
    action.action_raw.unit_command.unit_tags.append(u.tag)
    return action


def run_away_from_closest_enemy(u, closest_enemy_unit, run_away_factor=0.2):
    action = sc_pb.Action()
    action.action_raw.unit_command.ability_id = ABILITY_ID.SMART.value
    action.action_raw.unit_command.target_world_space_pos.x = \
        u.pos.x + (u.pos.x - closest_enemy_unit.pos.x) * run_away_factor
    action.action_raw.unit_command.target_world_space_pos.y = \
        u.pos.y + (u.pos.y - closest_enemy_unit.pos.y) * run_away_factor
    action.action_raw.unit_command.unit_tags.append(u.tag)
    return action
