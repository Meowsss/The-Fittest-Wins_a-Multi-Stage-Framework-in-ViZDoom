from pysc2.lib.typeenums import UNIT_TYPEID
from pysc2.lib.typeenums import UPGRADE_ID
import numpy as np


def distance(u1, u2):
    return ((u1.pos.x - u2.pos.x) ** 2 +
            (u1.pos.y - u2.pos.y) ** 2) ** 0.5


def find_nearest(units, unit):
    """ find the nearest one to 'unit' within the list 'units' """
    if not units:
        return None
    dd = np.asarray([distance(unit, u) for u in units])
    return units[dd.argmin()]
