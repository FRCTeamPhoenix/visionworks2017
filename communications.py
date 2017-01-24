from networktables import NetworkTable, NetworkTables
from enum import Enum
import logging
import time
import config


logging.basicConfig(stream=config.LOG_STREAM, level=config.LOG_LEVEL)
log = logging.getLogger(__name__)

# location of the NetworkTables server & table to use
__server_url__ = config.NETWORKTABLES_SERVER
__table_name__ = config.NETWORKTABLES_TABLE_NAME

# init and retrieve table from network
NetworkTables.initialize(__server_url__)
__table__ = NetworkTable.getTable(__table_name__)

# hardcoded key/values
__state_id__ = 'jetson_state'
class States(Enum):
    POWERED_ON = '0'
    CAMERA_ERROR = '1'
    TARGET_FOUND = '2'
    TARGET_NOT_FOUND = '3'
    POWERED_OFF = '4'

__targeting_id__ = 'target_angle'

# delimiter ( timestamp$DELIMETERvalue )
__delimiter__ = ';'

# return the current time (in a function so that the format can be changed if need be)
def __time__():
    return int(time.time() * 1000)

def set_state(state):
    assert isinstance(state, States), 'Value is not a valid state'
    last = None
    if __state_id__ in __table__.getKeys():
        last = __table__.getString(__state_id__).split(__delimiter__)[1]
    if state.value != last:
        log.info('Set state %s', state.name)
        s = str(__time__()) + __delimiter__ + str(state.value)
        log.debug('Sent value %s to %s', s, __state_id__)
        return __table__.putString(__state_id__, s)
    else:
        return True

def set_targeting(targeting_info):
    log.debug('Sent targeting info %s', targeting_info)
    s = str(__time__()) + __delimiter__ + str(targeting_info)
    log.debug('Sent value %s to %s', s, __targeting_id__)
    return __table__.putString(__targeting_id__, s)
