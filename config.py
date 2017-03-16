import logging
import sys
import v4l2ctl
import collections
import numpy as np
from enum import Enum
from datetime import datetime
import math

class Mode(Enum):
    HIGH_GOAL = 0
    GEARS = 1
    BOTH = 2
    NOT_YET_SET = 3
class State(Enum):
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
SERVER_MODE = Mode.HIGH_GOAL

# communications/networktables
NETWORKTABLES_SERVER = 'roboRIO-2342-FRC.local'
NETWORKTABLES_TABLE_NAME = 'datatable'
class NetworkTableKey(Enum):
    MODE = 'jetson_mode'
    HIGH_GOAL_STATE = 'high_goal_state'
    GEAR_STATE = 'gear_state'
    HIGH_GOAL_ANGLE = 'high_goal_angle'
    HIGH_GOAL_DISTANCE = 'high_goal_distance'
    TURRET_ANGLE = 'turret_angle' # current real life turret position (sent from RIO)
    GEARS_ROTATION = 'gears_rotation'
    GEARS_HORIZONTAL = 'gears_horizontal'
    GEARS_FORWARD = 'gears_forward'
    TIMESTAMP = '_time'

# logging config
LOG_LEVEL = logging.INFO

filename = datetime.now().strftime('%Y%m%d-%H:%M') + '.log'
LOG_STREAM = sys.stdout#open(filename, 'w+')
#sys.stdout = LOG_STREAM
#sys.stderr = LOG_STREAM

# camera configuration can be a camera index or a filename
#   turret - /dev/video10
#   gear - /dev/video11
#   configured under /etc/udev/rules.d/name-video-devices.rules on both jetsons
def path_to_index(path):
    import subprocess
    import os
    import re

    if os.path.exists(path):
        cmd = "udevadm info " + path
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        out = process.communicate()[0].split('\n')
        dev_path = '/sys' + out[7].split('=')[1]
        for root, dirs, files in os.walk(dev_path):
            m = re.search('\'video[0-9]\'', str(dirs))
            if m:
                return int(m.group(0)[-2])
    return None

VIDEO_SOURCE_TURRET = path_to_index('/dev/turret_cam')
VIDEO_SOURCE_GEAR = path_to_index('/dev/gear_cam')
RESOLUTION_X = 640
RESOLUTION_Y = 480
ASPECT = float(RESOLUTION_Y) / RESOLUTION_X
CAMERA_V4L_SETTINGS = collections.OrderedDict([
    (v4l2ctl.PROP_EXPOSURE_AUTO, 1),
    (v4l2ctl.PROP_EXPOSURE_AUTO_PRIORITY, 0),
    (v4l2ctl.PROP_EXPOSURE_ABS, 20),
    (v4l2ctl.PROP_WHITE_BALANCE_TEMP_AUTO, 0),
    (v4l2ctl.PROP_FOCUS_AUTO, 0)
])

# camera specs (used for pose estimation) (these ones are for the logitech c910 cams)
CAMERA_MATRIX = np.array([[ 5.6284447817984869e+02,      0.,    float(RESOLUTION_X / 2)],
                            [   0.,    5.6284447817984869e+02,    float(RESOLUTION_Y / 2)],
                            [   0.,      0.,      1.]])
CAMERA_DISTORTION_MATRIX = np.array([[3.9443937614041871e-02, -5.3261951371308369e-01, 0., 0., 1.4406927368950202e+00]])
CAMERA_DIAG_FOV = 83
CAMERA_HORIZ_FOV = math.sqrt(CAMERA_DIAG_FOV ** 2 / ((ASPECT ** 2) + 1))
CAMERA_VERT_FOV = ASPECT * CAMERA_HORIZ_FOV
CAMERA_HEIGHT = 21.5 # height of the camera off the ground (inches)
CAMERA_ANGLE = 38.4 # angle of the camera

# processing tuning
SHOOTER_THRESH_LOW = np.array([40, 20, 150])
SHOOTER_THRESH_HIGH = np.array([85, 255, 255])
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
MIN_SHOOTER_AREA = 0.00002 * RESOLUTION_X * RESOLUTION_Y
MAX_SHOOTER_AREA = 0.3 * RESOLUTION_X * RESOLUTION_Y
MIN_GEARS_AREA = 0.0002 * RESOLUTION_X * RESOLUTION_Y
MAX_GEARS_AREA = 0.4 * RESOLUTION_X * RESOLUTION_Y
RELATIVE_SPLIT_CONTOUR_EPSILON = 1

# constants for steamworks game
STEAMWORKS_HIGH_GOAL_CENTER_HEIGHT = 83 # inches
