from networktables import NetworkTable, NetworkTables
from enum import Enum
import logging
import time
import config
from config import States, Modes


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


def __log_value(k, v):
    log.debug('Sent value %s to %s', v, k)


def set_high_goal_state(state):
    assert isinstance(state, States), 'Value is not a valid jetson state'
    last = None
    if config.NETWORKTABLES_HIGH_GOAL_STATE_ID in __table.getKeys():
        last = __table.getNumber(config.NETWORKTABLES_HIGH_GOAL_STATE_ID)
    if state.value != last:
        log.info('Set state %s', state.name)
        t = __time()
        __log_value(config.NETWORKTABLES_HIGH_GOAL_STATE_ID, state.value)
        __log_value(config.NETWORKTABLES_HIGH_GOAL_STATE_TIMESTAMP_ID, t)
        return __table.putNumber(config.NETWORKTABLES_HIGH_GOAL_STATE_ID, state.value) & \
               __table.putNumber(config.NETWORKTABLES_HIGH_GOAL_STATE_TIMESTAMP_ID, t)
    else:
        return True


def set_gear_state(state):
    assert isinstance(state, States), 'Value is not a valid jetson state'
    last = None
    if config.NETWORKTABLES_GEAR_STATE_ID in __table.getKeys():
        last = __table.getNumber(config.NETWORKTABLES_GEAR_STATE_ID)
    if state.value != last:
        log.info('Set state %s', state.name)
        t = __time()
        __log_value(config.NETWORKTABLES_GEAR_STATE_ID, state.value)
        __log_value(config.NETWORKTABLES_GEAR_STATE_TIMESTAMP_ID, t)
        return __table.putNumber(config.NETWORKTABLES_GEAR_STATE_ID, state.value) & \
               __table.putNumber(config.NETWORKTABLES_GEAR_STATE_TIMESTAMP_ID, t)
    else:
        return True


def set_high_goal(angle, distance):
    log.debug('Sent high goal angle %s', angle)
    t = __time()
    __log_value(config.NETWORKTABLES_GOAL_ANGLE_ID, angle)
    __log_value(config.NETWORKTABLES_GOAL_ANGLE_TIMESTAMP_ID, t)
    __log_value(config.NETWORKTABLES_GOAL_DISTANCE_ID, distance)
    __log_value(config.NETWORKTABLES_GOAL_DISTANCE_TIMESTAMP_ID, t)
    return __table.putNumber(config.NETWORKTABLES_GOAL_ANGLE_ID, angle) & \
           __table.putNumber(config.NETWORKTABLES_GOAL_ANGLE_TIMESTAMP_ID, t) & \
           __table.putNumber(config.NETWORKTABLES_GOAL_DISTANCE_ID, distance) & \
           __table.putNumber(config.NETWORKTABLES_GEARS_DISTANCE_TIMESTAMP_ID, t)


def get_turret_angle():
    if config.NETWORKTABLES_TURRET_ANGLE_ID in __table.getKeys():
        return __table.getNumber(config.NETWORKTABLES_TURRET_ANGLE_ID)
    return None


def set_gear(angle, distance):
    log.debug('Sent gear angle %s', angle)
    log.debug('Sent gear distance %s', distance)
    t = __time()
    __log_value(config.NETWORKTABLES_GEARS_ANGLE_ID, angle)
    __log_value(config.NETWORKTABLES_GEARS_ANGLE_TIMESTAMP_ID, t)
    __log_value(config.NETWORKTABLES_GEARS_DISTANCE_ID, distance)
    __log_value(config.NETWORKTABLES_GEARS_DISTANCE_TIMESTAMP_ID, t)
    return __table.putNumber(config.NETWORKTABLES_GEARS_ANGLE_ID, angle) & \
           __table.putNumber(config.NETWORKTABLES_GEARS_ANGLE_TIMESTAMP_ID, t) & \
           __table.putNumber(config.NETWORKTABLES_GEARS_DISTANCE_ID, distance) & \
          -__table.putNumber(config.NETWORKTABLES_GEARS_DISTANCE_TIMESTAMP_ID, t)


def get_mode():
    if config.NETWORKTABLES_MODE_ID in __table.getKeys():
        return __table.getNumber(config.NETWORKTABLES_MODE_ID)
    return Modes.HIGH_GOAL
