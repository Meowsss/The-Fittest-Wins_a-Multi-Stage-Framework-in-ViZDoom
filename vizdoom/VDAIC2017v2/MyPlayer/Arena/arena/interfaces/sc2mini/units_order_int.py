""" Gym env wrappers """
from arena.utils.unit_util import collect_units_by_alliance
from arena.utils.unit_util import collect_units_by_types
from arena.utils.constant import AllianceType
from arena.utils.constant import COMBAT_UNITS
from arena.interfaces.interface import Interface

def UnitOrder(timestep):
    units_orders = []
    for alliance in [AllianceType.SELF.value, AllianceType.ENEMY.value]:
        units = timestep.observation.raw_data.units
        my_units = collect_units_by_alliance(units, alliance)
        my_combat_units = collect_units_by_types(my_units, COMBAT_UNITS)
        my_combat_units.sort(key=lambda u: u.tag)
        my_combat_units.sort(key=lambda u: u.unit_type)
        tags = [u.tag for u in my_combat_units]
        units_orders.append(tuple(tags))
    return tuple(units_orders)


class UnitOrderInt(Interface):
    def __init__(self, interface=None):
        super(UnitOrderInt, self).__init__(interface)
        self.unwrapped().units_order = None

    def reset(self, obs, **kwargs):
        super(UnitOrderInt, self).reset(obs, **kwargs)
        self.unwrapped().units_order = UnitOrder(obs)

