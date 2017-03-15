from networktables import NetworkTable, NetworkTables
from enum import Enum
import logging
import time
import config
from config import States, Modes, NetworkTablesKeys


logging.basicConfig(stream=config.LOG_STREAM, level=config.LOG_LEVEL)
log = logging.getLogger(__name__)

# location of the NetworkTables server & table to use
__server_url = config.NETWORKTABLES_SERVER
__table_name = config.NETWORKTABLES_TABLE_NAME

# init and retrieve table from network
NetworkTables.initialize(__server_url)
__table = NetworkTable.getTable(__table_name)

# return the current time (in a function so that the format can be changed if need be)
def __time():
    return int(time.time() * 1000)


def __send_number(k, v, timestamp):
    assert isinstance(k, NetworkTablesKeys), 'Key is not a NetworkTableKey'
    k = k.value
    log.debug('Sent value %s to %s', v, k)
    t = __time()
    r = __table.putNumber(k, v)
    if timestamp:
        log.debug('Sent value %s to %s', v, k + NetworkTablesKeys.TIMESTAMP.value)
        r = r and __table.putNumber(k + NetworkTablesKeys.TIMESTAMP.value, t)
    return r

def set_high_goal_state(state):
    assert isinstance(state, States), 'Value is not a valid jetson state'
    last = None
    if NetworkTablesKeys.HIGH_GOAL_STATE in __table.getKeys():
        last = __table.getNumber(NetworkTablesKeys.HIGH_GOAL_STATE)
    if state.value != last:
        log.info('Set state %s', state.name)
        return __send_number(NetworkTablesKeys.HIGH_GOAL_STATE, state.value, timestamp=True)
    else:
        return True


def set_gear_state(state):
    assert isinstance(state, States), 'Value is not a valid jetson state'
    last = None
    if NetworkTablesKeys.GEAR_STATE in __table.getKeys():
        last = __table.getNumber(NetworkTablesKeys.GEAR_STATE)
    if state.value != last:
        log.info('Set state %s', state.name)
        return __send_number(NetworkTablesKeys.GEAR_STATE, state.value, timestamp=True)
    else:
        return True


def set_high_goal(angle, distance):
    log.debug('Sent high goal angle %s', angle)
    log.debug('Sent high goal distance %s', distance)
    return __send_number(NetworkTablesKeys.HIGH_GOAL_ANGLE, angle, timestamp=True) & \
           __send_number(NetworkTablesKeys.HIGH_GOAL_DISTANCE, distance, timestamp=True)


def get_turret_angle():
    if NetworkTablesKeys.TURRET_ANGLE in __table.getKeys():
        return __table.getNumber(NetworkTablesKeys.TURRET_ANGLE)
    return None


def set_gear(rotation, horizontal, forward):
    log.debug('Sent gear angle %s', rotation)
    log.debug('Sent gear horizontal move %s', horizontal)
    log.debug('Sent gear forward move %s', forward)
    return __send_number(NetworkTablesKeys.GEARS_ROTATION, rotation, timestamp=True) & \
    __send_number(NetworkTablesKeys.GEARS_HORIZONTAL, horizontal, timestamp=True) & \
    __send_number(NetworkTablesKeys.GEARS_FORWARD, forward, timestamp=True)


def get_mode():
    if NetworkTablesKeys.MODE in __table.getKeys():
        return __table.getNumber(NetworkTablesKeys.MODE)
    return Modes.HIGH_GOAL
