import logging
import sys
import v4l2ctl
import collections
import numpy as np
from enum import Enum
from datetime import datetime

class Modes(Enum):
    HIGH_GOAL = 0
    GEARS = 1
    BOTH = 2
    NOT_YET_SET = 3
class States(Enum):
    POWERED_ON = 0
    CAMERA_ERROR = 1
    TARGET_FOUND = 2
    TARGET_NOT_FOUND = 3
    POWERED_OFF = 4


# General
WAIT_TIME = 25 # wait time in the main loop
START_FRAME = 0 # for video replays


# GUI configuration
GUI_SHOW = False
GUI_DEBUG_DRAW = True
GUI_WAIT_FOR_CONTINUE = False
GUI_EXIT_KEY = 27
GUI_WAIT_FOR_CONTINUE_KEY = 32

# HTTP configuration
USE_HTTP_SERVER = True
SERVER_MODE = Modes.HIGH_GOAL

# communications/networktables
NETWORKTABLES_SERVER = 'roboRIO-2342-FRC.local'
NETWORKTABLES_TABLE_NAME = 'datatable'
NETWORKTABLES_STATE_ID = 'jetson_state'
NETWORKTABLES_STATE_TIMESTAMP_ID = 'jetson_state_time'
NETWORKTABLES_MODE_ID = 'jetson_mode'
NETWORKTABLES_GOAL_ID = 'high_goal'
NETWORKTABLES_GOAL_TIMESTAMP_ID = 'high_goal_time'
NETWORKTABLES_GEARS_ANGLE_ID = 'gear_angle'
NETWORKTABLES_GEARS_DISTANCE_ID = 'gear_distance'
NETWORKTABLES_GEARS_ANGLE_TIMESTAMP_ID = 'gear_angle_time'
NETWORKTABLES_GEARS_DISTANCE_TIMESTAMP_ID = 'gear_distance_time'
NETWORKTABLES_TURRET_ANGLE_ID = 'turret_angle'

# logging config
LOG_LEVEL = logging.INFO

filename = datetime.now().strftime('%Y%m%d-%H:%M') + '.log'
LOG_STREAM = open(filename, 'w+')
sys.stdout = LOG_STREAM
sys.stderr = LOG_STREAM

# camera configuration can be a camera index or a filename
#   turret - /dev/video10
#   gear - /dev/video11
VIDEO_SOURCE_TURRET = 10
VIDEO_SOURCE_GEAR = 11
RESOLUTION_X = 640
RESOLUTION_Y = 480
CAMERA_V4L_SETTINGS = collections.OrderedDict([
    (v4l2ctl.PROP_EXPOSURE_AUTO, 1),
    (v4l2ctl.PROP_EXPOSURE_AUTO_PRIORITY, 0),
    (v4l2ctl.PROP_EXPOSURE_ABS, 10),
    (v4l2ctl.PROP_WHITE_BALANCE_TEMP_AUTO, 0),
    (v4l2ctl.PROP_FOCUS_AUTO, 0)
])

# camera specs (used for pose estimation) (these ones are for the logitech c910 cams)
CAMERA_MATRIX = np.array([[ 771.,      0.,    float(RESOLUTION_X / 2)],
                            [   0.,    771.,    float(RESOLUTION_Y / 2)],
                            [   0.,      0.,      1.]])
CAMERA_DISTORTION_MATRIX = np.array([[ 0.03236637, -0.03763916, -0.00569912, -0.00091719, -0.008543  ]])
CAMERA_DIAG_FOV = 83

# processing tuning
SHOOTER_THRESH_LOW = np.array([70, 100, 85])
SHOOTER_THRESH_HIGH = np.array([80, 255, 255])
GEAR_THRESH_LOW = np.array([70, 100, 100])
GEAR_THRESH_HIGH = np.array([80, 255, 255])


MORPH_KERNEL_WIDTH = 3
MORPH_KERNEL_HEIGHT = 3
POLY_EPS = 0.1

# pose estimation
MIN_NORM_TVECS = 0.0001
MAX_NORM_TVECS = 1000
GEARS_OBJP = np.array([[-5.125, -2.5, 0],
                       [5.125, -2.5, 0],
                       [5.125, 2.5, 0],
                       [-5.125, 2.5, 0]])

# bounds for target size
MIN_SHOOTER_AREA = 0.00003 * RESOLUTION_X * RESOLUTION_Y
MAX_SHOOTER_AREA = 0.3 * RESOLUTION_X * RESOLUTION_Y
MIN_GEARS_AREA = 0.0002 * RESOLUTION_X * RESOLUTION_Y
MAX_GEARS_AREA = 0.4 * RESOLUTION_X * RESOLUTION_Y
RELATIVE_SPLIT_CONTOUR_EPSILON = 1
