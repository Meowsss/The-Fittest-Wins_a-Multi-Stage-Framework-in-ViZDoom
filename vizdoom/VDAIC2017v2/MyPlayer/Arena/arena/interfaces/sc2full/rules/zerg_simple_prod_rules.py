"""Production Strategy Manager"""
import random
from gym import spaces
import distutils.version
from ..tstarbot_rules.production_strategy.build_cmd import *
from pysc2.lib.typeenums import UNIT_TYPEID, UPGRADE_ID
from arena.utils.constant import MAXIMUM_NUM
from ..tstarbot_rules.production_strategy.base_zerg_production_mgr import ZergBaseProductionMgr
from ..tstarbot_rules.production_strategy.util import unique_unit_count


class ZergSimpleProdMgr(ZergBaseProductionMgr):
    def __init__(self, dc, auto_resource=True, auto_larva=True, auto_pre=True,
                 auto_supply=True, keep_order=False):
        super(ZergSimpleProdMgr, self).__init__(dc)
        self.auto_resource = auto_resource  # use rules for resource_mgr
        self.auto_larva = auto_larva  # use rules to spawn larva
        self.auto_supply = auto_supply  # use rules to produce supply in advance
        self.auto_pre = auto_pre  # whether use rule to add prerequisite
        self.keep_order = keep_order and self.auto_pre  # whether to maintain a build_order
        self.verbose = 0
        self._build_actions = [
            UNIT_TYPEID.INVALID,  # Do not produce any unit.
            UNIT_TYPEID.ZERG_DRONE,
            UNIT_TYPEID.ZERG_OVERLORD,
            UNIT_TYPEID.ZERG_QUEEN,
            UNIT_TYPEID.ZERG_ROACH,
            UNIT_TYPEID.ZERG_ZERGLING,
            UNIT_TYPEID.ZERG_EXTRACTOR,
            UNIT_TYPEID.ZERG_SPAWNINGPOOL,
            UNIT_TYPEID.ZERG_ROACHWARREN,
        ]
        self._resource_actions = [
            'none',
            'gas_first',
            'mineral_first',
        ]
        self._other_actions = [
            'spawn_larva',
        ]
        nvec = len(self._build_actions)
        self.action_space = spaces.Discrete(nvec)

    def reset(self, obs):
        self.onStart = True
        self.build_order.clear_all()
        self.obs = obs.observation
        self.cut_in_item = []
        self.born_pos = None

    def update(self, dc, am, action):
        if self.onStart:
            bases = dc.dd.base_pool.bases
            self.born_pos = [[bases[tag].unit.float_attr.pos_x,
                              bases[tag].unit.float_attr.pos_y]
                             for tag in bases][0]
            self.onStart = False
        self.obs = dc.sd.obs

        if self.build_order.is_empty() and action != 0:
            build_item = self._build_actions[action]
            self.build_order.queue_as_highest(build_item)

        if self.auto_supply:
            # will produce overlord automatically
            self.check_supply()

        if not self.build_order.is_empty():
            if self.auto_pre:
                while self.detect_dead_lock(dc):
                    pass
            current_item = self.build_order.current_item()
            if current_item and self.can_build(current_item, dc):
                # print(current_item.unit_id)
                if current_item.isUnit:
                    if self.set_build_base(current_item, dc):
                        self.build_order.remove_current_item()
                        if current_item.unit_id in self.cut_in_item:
                            self.cut_in_item.remove(current_item.unit_id)
                        if self.verbose > 0:
                            print('Produce: {}'.format(current_item.unit_id))
                else:  # Upgrade
                    if self.upgrade(current_item, dc):
                        self.build_order.remove_current_item()
                        if current_item.unit_id in self.cut_in_item:
                            self.cut_in_item.remove(current_item.unit_id)
                        if self.verbose > 0:
                            print('Upgrade: {}'.format(current_item.unit_id))

        if not self.keep_order:
            self.build_order.clear_all()
            self.cut_in_item = []

        if self.auto_resource:
            # will collect resource automatically
            if self.should_gas_first(dc):
                dc.dd.build_command_queue.put(BuildCmdHarvest(gas_first=True))
            if self.should_mineral_first(dc):
                dc.dd.build_command_queue.put(BuildCmdHarvest(gas_first=False))
        else:
            if self._resource_actions[action[1]] == 'gas_first':
                dc.dd.build_command_queue.put(BuildCmdHarvest(gas_first=True))
            elif self._resource_actions[action[1]] == 'mineral_first':
                dc.dd.build_command_queue.put(BuildCmdHarvest(gas_first=False))

        if self.auto_larva:
            # will use queen to spawn larva automatically
            self.spawn_larva(dc)

    def detect_dead_lock(self, data_context):
        current_item = self.build_order.current_item()
        builder = None
        for unit_type in current_item.whatBuilds:
            if (self.has_unit(unit_type) or self.unit_in_progress(unit_type)
                    or unit_type == UNIT_TYPEID.ZERG_LARVA.value):
                builder = unit_type
                break
        if builder is None and len(current_item.whatBuilds) > 0:
            builder_id = [unit_id for unit_id in UNIT_TYPEID
                          if unit_id.value == current_item.whatBuilds[0]]
            self.build_order.queue_as_highest(builder_id[0])
            if self.verbose > 0:
                print('Cut in: {}'.format(builder_id[0]))
            return True
        required_unit = None
        for unit_type in current_item.requiredUnits:
            if self.has_unit(unit_type) or self.unit_in_progress(unit_type):
                required_unit = unit_type
                break
        if required_unit is None and len(current_item.requiredUnits) > 0:
            required_id = [unit_id for unit_id in UNIT_TYPEID
                           if unit_id.value == current_item.requiredUnits[0]]
            self.build_order.queue_as_highest(required_id[0])
            if self.verbose > 0:
                print('Cut in: {}'.format(required_id[0]))
            return True
        required_upgrade = None
        for upgrade_type in current_item.requiredUpgrades:
            if (not self.has_upgrade(data_context, [upgrade_type])
                    and not self.upgrade_in_progress(upgrade_type)):
                required_upgrade = upgrade_type
                break
        if required_upgrade is not None:
            required_id = [up_id for up_id in UPGRADE_ID
                           if up_id.value == required_upgrade]
            self.build_order.queue_as_highest(required_id[0])
            if self.verbose > 0:
                print('Cut in: {}'.format(required_id[0]))
            return True
        if current_item.gasCost > 0:
            n_g = len([u for u in self.obs['units']
                       if u.unit_type == UNIT_TYPEID.ZERG_EXTRACTOR.value
                       and u.int_attr.vespene_contents > 100
                       and u.int_attr.alliance == 1])
            if n_g == 0:
                self.build_order.queue_as_highest(self.gas_unit())
                if self.verbose > 0:
                    print('Cut in: {}'.format(self.gas_unit()))
                return True
        return False

    def can_build(self, build_item, dc):  # check resource requirement
        if not (self.has_building_built(build_item.whatBuilds)
                and self.has_building_built(build_item.requiredUnits)
                and self.has_upgrade(dc, build_item.requiredUpgrades)
                and not self.expand_waiting_resource(dc)):
            return False
        play_info = self.obs["player"]
        if (build_item.supplyCost > 0
            and play_info[3] + build_item.supplyCost > play_info[4]):
            return False
        if build_item.unit_id == UNIT_TYPEID.ZERG_HATCHERY:
            return self.obs['player'][1] >= build_item.mineralCost - 100
        return (self.obs['player'][1] >= build_item.mineralCost
                and self.obs['player'][2] >= build_item.gasCost)

    def check_avail(self, build_item, dc):
        if build_item in MAXIMUM_NUM:
            if self.unit_count[build_item.value] >= MAXIMUM_NUM[build_item]:
                return 0
        if build_item == UNIT_TYPEID.ZERG_OVERLORD:
            if (dc.sd.obs['player'][3] + 20 < dc.sd.obs['player'][4]
                    or dc.sd.obs['player'][4] >= 200):  # enough supply
                return 0
        if build_item == UNIT_TYPEID.ZERG_EXTRACTOR:
            tag = self.find_base_to_build_extractor(dc.dd.base_pool.bases)
            if tag is None:
                return 0
        if build_item == UNIT_TYPEID.ZERG_HATCHERY:
            if not dc.dd.build_command_queue.empty():
                return 0
        if type(build_item) == UNIT_TYPEID:
            build_item = self.TT.getUnitData(build_item.value)
        elif type(build_item) == UPGRADE_ID:
            build_item = self.TT.getUpgradeData(build_item.value)
        has_spare_builder = False
        for unit_type in build_item.whatBuilds:
            builder = self.TT.getUnitData(unit_type)
            if builder.isBuilding:  # building must be available and spare
                has_spare_builder = self.find_spare_building([unit_type]) is not None
                break
            else:  # unit always be available
                has_spare_builder = self.has_unit(unit_type)
                break
        if not has_spare_builder:
            can_build = False
        elif self.keep_order:
            can_build = (self.has_building_built(build_item.requiredUnits)
                         and self.has_upgrade(dc, build_item.requiredUpgrades)
                         and dc.sd.obs['player'][3] + build_item.supplyCost <= dc.sd.obs['player'][4])
        else:
            can_build = (self.has_building_built(build_item.requiredUnits)
                         and self.has_upgrade(dc, build_item.requiredUpgrades)
                         and dc.sd.obs['player'][3] + build_item.supplyCost <= dc.sd.obs['player'][4]
                         and dc.sd.obs['player'][1] >= build_item.mineralCost
                         and dc.sd.obs['player'][2] >= build_item.gasCost)
        return int(can_build)

    def action_avail(self, dc):
        self.obs = dc.sd.obs
        self.unit_count = unique_unit_count(dc.sd.obs['units'], dc.sd.TT)
        action_avail = [1]  # none is always available
        if self.keep_order and not self.build_order.is_empty():
            action_avail.extend([0] * (len(self._build_actions) - 1))
        else:
            for i in range(1, len(self._build_actions)):
                build_item = self._build_actions[i]
                if type(build_item) == list:  # multi-level upgrade
                    up_list = [up_type in dc.sd.obs['raw_data'].player.upgrade_ids
                               for up_type in build_item]
                    if len(up_list) < len(build_item):
                        build_item = build_item[len(up_list)]
                    else:  # upgrades already exists
                        action_avail.append(0)
                        continue
                action_avail.append(self.check_avail(build_item, dc))
        if not self.auto_resource:
            action_avail.extend([1] * 3)
        return action_avail
