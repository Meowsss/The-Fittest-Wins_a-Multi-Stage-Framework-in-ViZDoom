from enum import Enum
from pysc2.lib.typeenums import UNIT_TYPEID


class AllianceType(Enum):
    SELF = 1
    ALLY = 2
    NEUTRAL = 3
    ENEMY = 4


COMBAT_UNITS = set([
    # Zerg
    UNIT_TYPEID.ZERG_BANELING.value,
    UNIT_TYPEID.ZERG_BANELINGBURROWED.value,
    UNIT_TYPEID.ZERG_BROODLING.value,
    UNIT_TYPEID.ZERG_BROODLORD.value,
    UNIT_TYPEID.ZERG_CHANGELING.value,
    UNIT_TYPEID.ZERG_CHANGELINGMARINE.value,
    UNIT_TYPEID.ZERG_CHANGELINGMARINESHIELD.value,
    UNIT_TYPEID.ZERG_CHANGELINGZEALOT.value,
    UNIT_TYPEID.ZERG_CHANGELINGZERGLING.value,
    UNIT_TYPEID.ZERG_CHANGELINGZERGLINGWINGS.value,
    UNIT_TYPEID.ZERG_CORRUPTOR.value,
    UNIT_TYPEID.ZERG_HYDRALISK.value,
    UNIT_TYPEID.ZERG_HYDRALISKBURROWED.value,
    UNIT_TYPEID.ZERG_INFESTOR.value,
    UNIT_TYPEID.ZERG_INFESTORBURROWED.value,
    UNIT_TYPEID.ZERG_MUTALISK.value,
    UNIT_TYPEID.ZERG_NYDUSCANAL.value,
    UNIT_TYPEID.ZERG_OVERLORD.value,
    UNIT_TYPEID.ZERG_OVERSEER.value,
    UNIT_TYPEID.ZERG_QUEEN.value,
    UNIT_TYPEID.ZERG_QUEENBURROWED.value,
    UNIT_TYPEID.ZERG_RAVAGER.value,
    UNIT_TYPEID.ZERG_ROACH.value,
    UNIT_TYPEID.ZERG_ROACHBURROWED.value,
    UNIT_TYPEID.ZERG_SPINECRAWLER.value,
    UNIT_TYPEID.ZERG_SPINECRAWLERUPROOTED.value,
    UNIT_TYPEID.ZERG_SPORECRAWLER.value,
    UNIT_TYPEID.ZERG_SPORECRAWLERUPROOTED.value,
    UNIT_TYPEID.ZERG_SWARMHOSTMP.value,
    UNIT_TYPEID.ZERG_ULTRALISK.value,
    UNIT_TYPEID.ZERG_ZERGLING.value,
    UNIT_TYPEID.ZERG_ZERGLINGBURROWED.value,
    UNIT_TYPEID.ZERG_LURKERMP.value,
    UNIT_TYPEID.ZERG_LURKERMPBURROWED.value,
    UNIT_TYPEID.ZERG_VIPER.value,

    # TERRAN
    UNIT_TYPEID.TERRAN_SCV.value,
    UNIT_TYPEID.TERRAN_GHOST.value,
    UNIT_TYPEID.TERRAN_MARAUDER.value,
    UNIT_TYPEID.TERRAN_MARINE.value,
    UNIT_TYPEID.TERRAN_REAPER.value,
    UNIT_TYPEID.TERRAN_HELLION.value,
    UNIT_TYPEID.TERRAN_CYCLONE.value,
    UNIT_TYPEID.TERRAN_SIEGETANK.value,
    UNIT_TYPEID.TERRAN_THOR.value,
    UNIT_TYPEID.TERRAN_WIDOWMINE.value,
    UNIT_TYPEID.TERRAN_NUKE.value,
    UNIT_TYPEID.TERRAN_BANSHEE.value,
    UNIT_TYPEID.TERRAN_BATTLECRUISER.value,
    UNIT_TYPEID.TERRAN_LIBERATOR.value,
    UNIT_TYPEID.TERRAN_VIKINGFIGHTER.value,
    UNIT_TYPEID.TERRAN_RAVEN.value,
    UNIT_TYPEID.TERRAN_MEDIVAC.value,
    UNIT_TYPEID.TERRAN_MULE.value,

# Protoss
    UNIT_TYPEID.PROTOSS_PROBE.value,
    UNIT_TYPEID.PROTOSS_MOTHERSHIPCORE.value,
    UNIT_TYPEID.PROTOSS_ZEALOT.value,
    UNIT_TYPEID.PROTOSS_SENTRY.value,
    UNIT_TYPEID.PROTOSS_STALKER.value,
    UNIT_TYPEID.PROTOSS_HIGHTEMPLAR.value,
    UNIT_TYPEID.PROTOSS_DARKTEMPLAR.value,
    UNIT_TYPEID.PROTOSS_ADEPT.value,
    UNIT_TYPEID.PROTOSS_COLOSSUS.value,
    UNIT_TYPEID.PROTOSS_DISRUPTOR.value,
    UNIT_TYPEID.PROTOSS_WARPPRISM.value,
    UNIT_TYPEID.PROTOSS_OBSERVER.value,
    UNIT_TYPEID.PROTOSS_IMMORTAL.value,
    UNIT_TYPEID.PROTOSS_CARRIER.value,
    UNIT_TYPEID.PROTOSS_ORACLE.value,
    UNIT_TYPEID.PROTOSS_PHOENIX.value,
    UNIT_TYPEID.PROTOSS_VOIDRAY.value,
    UNIT_TYPEID.PROTOSS_TEMPEST.value,
    UNIT_TYPEID.PROTOSS_INTERCEPTOR.value,
    UNIT_TYPEID.PROTOSS_ORACLESTASISTRAP.value
])

ZERG_COMBAT_UNITS = set([
    # Zerg
    UNIT_TYPEID.ZERG_BANELING.value,
    # UNIT_TYPEID.ZERG_BANELINGBURROWED.value,
    UNIT_TYPEID.ZERG_BROODLING.value,
    UNIT_TYPEID.ZERG_BROODLORD.value,
    # UNIT_TYPEID.ZERG_CHANGELING.value,
    # UNIT_TYPEID.ZERG_CHANGELINGMARINE.value,
    # UNIT_TYPEID.ZERG_CHANGELINGMARINESHIELD.value,
    # UNIT_TYPEID.ZERG_CHANGELINGZEALOT.value,
    # UNIT_TYPEID.ZERG_CHANGELINGZERGLING.value,
    # UNIT_TYPEID.ZERG_CHANGELINGZERGLINGWINGS.value,
    UNIT_TYPEID.ZERG_CORRUPTOR.value,
    UNIT_TYPEID.ZERG_HYDRALISK.value,
    # UNIT_TYPEID.ZERG_HYDRALISKBURROWED.value,
    UNIT_TYPEID.ZERG_INFESTOR.value,
    # UNIT_TYPEID.ZERG_INFESTORBURROWED.value,
    UNIT_TYPEID.ZERG_MUTALISK.value,
    # UNIT_TYPEID.ZERG_NYDUSCANAL.value,
    UNIT_TYPEID.ZERG_OVERLORD.value,
    UNIT_TYPEID.ZERG_OVERSEER.value,
    UNIT_TYPEID.ZERG_QUEEN.value,
    # UNIT_TYPEID.ZERG_QUEENBURROWED.value,
    UNIT_TYPEID.ZERG_RAVAGER.value,
    UNIT_TYPEID.ZERG_ROACH.value,
    # UNIT_TYPEID.ZERG_ROACHBURROWED.value,
    UNIT_TYPEID.ZERG_SPINECRAWLER.value,
    # UNIT_TYPEID.ZERG_SPINECRAWLERUPROOTED.value,
    UNIT_TYPEID.ZERG_SPORECRAWLER.value,
    # UNIT_TYPEID.ZERG_SPORECRAWLERUPROOTED.value,
    # UNIT_TYPEID.ZERG_SWARMHOSTMP.value,
    UNIT_TYPEID.ZERG_ULTRALISK.value,
    UNIT_TYPEID.ZERG_ZERGLING.value,
    # UNIT_TYPEID.ZERG_ZERGLINGBURROWED.value,
    UNIT_TYPEID.ZERG_LURKERMP.value,
    # UNIT_TYPEID.ZERG_LURKERMPBURROWED.value,
    UNIT_TYPEID.ZERG_VIPER.value,
])

ZERG_BUILDING_UNITS = set([
  # Zerg
  UNIT_TYPEID.ZERG_HATCHERY.value,
  # UNIT_TYPEID.ZERG_SPINECRAWLER.value,
  # UNIT_TYPEID.ZERG_SPORECRAWLER.value,
  UNIT_TYPEID.ZERG_EXTRACTOR.value,
  UNIT_TYPEID.ZERG_SPAWNINGPOOL.value,
  UNIT_TYPEID.ZERG_EVOLUTIONCHAMBER.value,
  UNIT_TYPEID.ZERG_ROACHWARREN.value,
  UNIT_TYPEID.ZERG_BANELINGNEST.value,
  # UNIT_TYPEID.ZERG_CREEPTUMOR.value,
  UNIT_TYPEID.ZERG_LAIR.value,
  UNIT_TYPEID.ZERG_HYDRALISKDEN.value,
  UNIT_TYPEID.ZERG_LURKERDENMP.value,
  UNIT_TYPEID.ZERG_SPIRE.value,
  UNIT_TYPEID.ZERG_SWARMHOSTBURROWEDMP.value,
  # UNIT_TYPEID.ZERG_NYDUSNETWORK.value,
  UNIT_TYPEID.ZERG_INFESTATIONPIT.value,
  UNIT_TYPEID.ZERG_HIVE.value,
  UNIT_TYPEID.ZERG_GREATERSPIRE.value,
  UNIT_TYPEID.ZERG_ULTRALISKCAVERN.value,
])

NOR_CONST = 50.0

ZERG_COMBAT_UNITS_FEAT_NOR = {
    UNIT_TYPEID.ZERG_BANELING.value: 1.0 / NOR_CONST,
    # UNIT_TYPEID.ZERG_BANELINGBURROWED.value,
    UNIT_TYPEID.ZERG_BROODLING.value: 0.2 / NOR_CONST,
    UNIT_TYPEID.ZERG_BROODLORD.value: 4.0 / NOR_CONST,
    # UNIT_TYPEID.ZERG_CHANGELING.value,
    # UNIT_TYPEID.ZERG_CHANGELINGMARINE.value,
    # UNIT_TYPEID.ZERG_CHANGELINGMARINESHIELD.value,
    # UNIT_TYPEID.ZERG_CHANGELINGZEALOT.value,
    # UNIT_TYPEID.ZERG_CHANGELINGZERGLING.value,
    # UNIT_TYPEID.ZERG_CHANGELINGZERGLINGWINGS.value,
    UNIT_TYPEID.ZERG_CORRUPTOR.value: 2.0 / NOR_CONST,
    UNIT_TYPEID.ZERG_HYDRALISK.value: 2.0 / NOR_CONST,
    # UNIT_TYPEID.ZERG_HYDRALISKBURROWED.value,
    UNIT_TYPEID.ZERG_INFESTOR.value: 2.0 / NOR_CONST,
    # UNIT_TYPEID.ZERG_INFESTORBURROWED.value,
    UNIT_TYPEID.ZERG_MUTALISK.value: 2.0 / NOR_CONST,
    # UNIT_TYPEID.ZERG_NYDUSCANAL.value,
    UNIT_TYPEID.ZERG_OVERLORD.value: 1.0 / 30.0,
    UNIT_TYPEID.ZERG_OVERSEER.value: 1.0 / 30.0,
    UNIT_TYPEID.ZERG_QUEEN.value: 2.0 / NOR_CONST,
    # UNIT_TYPEID.ZERG_QUEENBURROWED.value,
    UNIT_TYPEID.ZERG_RAVAGER.value: 3.0 / NOR_CONST,
    UNIT_TYPEID.ZERG_ROACH.value: 2.0 / NOR_CONST,
    # UNIT_TYPEID.ZERG_ROACHBURROWED.value,
    UNIT_TYPEID.ZERG_SPINECRAWLER.value: 1.0 / 3.0,
    # UNIT_TYPEID.ZERG_SPINECRAWLERUPROOTED.value,
    UNIT_TYPEID.ZERG_SPORECRAWLER.value: 1.0 / 3.0,
    # UNIT_TYPEID.ZERG_SPORECRAWLERUPROOTED.value,
    # UNIT_TYPEID.ZERG_SWARMHOSTMP.value,
    UNIT_TYPEID.ZERG_ULTRALISK.value: 6.0 / NOR_CONST,
    UNIT_TYPEID.ZERG_ZERGLING.value: 1.0 / NOR_CONST,
    # UNIT_TYPEID.ZERG_ZERGLINGBURROWED.value,
    UNIT_TYPEID.ZERG_LURKERMP.value: 3.0 / NOR_CONST,
    # UNIT_TYPEID.ZERG_LURKERMPBURROWED.value,
    UNIT_TYPEID.ZERG_VIPER.value: 3.0 / NOR_CONST,
    UNIT_TYPEID.ZERG_DRONE.value: 1.0 / NOR_CONST
}

MAIN_BASE_BUILDS = set([
  # Zerg
  UNIT_TYPEID.ZERG_SPAWNINGPOOL.value,
  UNIT_TYPEID.ZERG_ROACHWARREN.value,
  UNIT_TYPEID.ZERG_BANELINGNEST.value,
])

MINERAL_UNITS = set([UNIT_TYPEID.NEUTRAL_RICHMINERALFIELD.value,
                     UNIT_TYPEID.NEUTRAL_RICHMINERALFIELD750.value,
                     UNIT_TYPEID.NEUTRAL_MINERALFIELD.value,
                     UNIT_TYPEID.NEUTRAL_MINERALFIELD750.value,
                     UNIT_TYPEID.NEUTRAL_LABMINERALFIELD.value,
                     UNIT_TYPEID.NEUTRAL_LABMINERALFIELD750.value,
                     UNIT_TYPEID.NEUTRAL_PURIFIERRICHMINERALFIELD.value,
                     UNIT_TYPEID.NEUTRAL_PURIFIERRICHMINERALFIELD750.value,
                     UNIT_TYPEID.NEUTRAL_PURIFIERMINERALFIELD.value,
                     UNIT_TYPEID.NEUTRAL_PURIFIERMINERALFIELD750.value,
                     UNIT_TYPEID.NEUTRAL_BATTLESTATIONMINERALFIELD.value,
                     UNIT_TYPEID.NEUTRAL_BATTLESTATIONMINERALFIELD750.value])

MAXIMUM_NUM = {
    UNIT_TYPEID.ZERG_SPAWNINGPOOL: 1,
    UNIT_TYPEID.ZERG_ROACHWARREN: 1,
    UNIT_TYPEID.ZERG_HYDRALISKDEN: 1,
    UNIT_TYPEID.ZERG_HATCHERY: 4,
    UNIT_TYPEID.ZERG_EVOLUTIONCHAMBER: 2,
    UNIT_TYPEID.ZERG_BANELINGNEST: 1,
    UNIT_TYPEID.ZERG_INFESTATIONPIT: 1,
    UNIT_TYPEID.ZERG_SPIRE: 1,
    UNIT_TYPEID.ZERG_ULTRALISKCAVERN: 1,
    UNIT_TYPEID.ZERG_NYDUSNETWORK: 1,
    UNIT_TYPEID.ZERG_LURKERDENMP: 1,
    UNIT_TYPEID.ZERG_LAIR: 1,
    UNIT_TYPEID.ZERG_HIVE: 1,
    UNIT_TYPEID.ZERG_GREATERSPIRE: 1,
    UNIT_TYPEID.ZERG_OVERSEER: 10,
    UNIT_TYPEID.ZERG_QUEEN: 4,
    UNIT_TYPEID.ZERG_CORRUPTOR: 6,
    UNIT_TYPEID.ZERG_INFESTOR: 3,
    UNIT_TYPEID.ZERG_RAVAGER: 5,
    UNIT_TYPEID.ZERG_ULTRALISK: 6,
    UNIT_TYPEID.ZERG_MUTALISK: 6,
    UNIT_TYPEID.ZERG_BROODLORD: 4,
    UNIT_TYPEID.ZERG_DRONE: 66,
    UNIT_TYPEID.ZERG_VIPER: 3,
    UNIT_TYPEID.ZERG_LURKERMP: 6,
    UNIT_TYPEID.ZERG_ZERGLING: 30,
    UNIT_TYPEID.ZERG_ROACH: 30,
    UNIT_TYPEID.ZERG_HYDRALISK: 20,
    UNIT_TYPEID.ZERG_BANELING: 5,
    UNIT_TYPEID.ZERG_SPINECRAWLER: 6,
    UNIT_TYPEID.ZERG_SPORECRAWLER: 2,
}