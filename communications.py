from networktables import NetworkTable, NetworkTables
from enum import Enum
import logging
import time
import config


logging.basicConfig(stream=config.LOG_STREAM, level=config.LOG_LEVEL)
log = logging.getLogger(__name__)

# location of the NetworkTables server & table to use
__server_url = config.NETWORKTABLES_SERVER
__table_name = config.NETWORKTABLES_TABLE_NAME

# init and retrieve table from network
NetworkTables.initialize(__server_url)
__table = NetworkTable.getTable(__table_name)

# hardcoded key/values
__state_id = 'jetson_state'
__state_timestamp_id = 'jetson_state_time'
class States(Enum):
    POWERED_ON = 0
    CAMERA_ERROR = 1
    TARGET_FOUND = 2
    TARGET_NOT_FOUND = 3
    POWERED_OFF = 4

__mode_id = 'jetson_mode'
class Modes(Enum):
    HIGH_GOAL = 0
    GEARS = 1
    BOTH = 2
    NOT_YET_SET = 3

__goal_id = 'high_goal'
__goal_timestamp_id = 'high_goal_time'
__gears_rvecs_id = 'gear_rvecs'
__gears_tvecs_id = 'gear_tvecs'
__gears_timestamp = 'gear_time'


# return the current time (in a function so that the format can be changed if need be)
def __time():
    return int(time.time() * 1000)

def __log_value(k, v):
    log.debug('Sent value %s to %s', v, k)

def set_state(state):
    assert isinstance(state, States), 'Value is not a valid jetson state'
    last = None
    if __state_id in __table.getKeys():
        last = __table.getNumber(__state_id)
    if state.value != last:
        log.info('Set state %s', state.name)
        t = __time()
        __log_value(__state_id, state.value)
        __log_value(__state_timestamp_id, t)
        return __table.putNumber(__state_id, state.value) & \
               __table.putNumber(__state_timestamp_id, t)
    else:
        return True

def set_high_goal(angle):
    log.debug('Sent high goal angle %s', angle)
    t = __time()
    __log_value(__goal_id, angle)
    __log_value(__goal_timestamp_id, t)
    return __table.putNumber(__goal_id, angle) & \
           __table.putNumber(__goal_timestamp_id, t)

def set_gear(rvecs, tvecs):
    log.debug('Sent gear rvecs %s', rvecs)
    t = __time()
    __log_value(__gears_rvecs_id, rvecs)
    __log_value(__gears_tvecs_id, tvecs)
    __log_value(__gears_timestamp, t)
    return __table.putNumberArray(__gears_rvecs_id, rvecs) & \
           __table.putNumberArray(__gears_tvecs_id, tvecs) & \
           __table.putNumber(__gears_timestamp, t)

def get_mode():
    if __mode_id in __table.getKeys():
        return __table.getNumber(__mode_id)
    return Modes.NOT_YET_SET