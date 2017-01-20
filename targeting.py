# TO DO
#   - Implement consistent origin checks (check)
#   - Reprojected points error checking (not worth it, we got rid of that stuff in the cpp proj
#   - Fancy debug features (yeah boy)
#   - Clean code (pretty much)
#   - Only do calculations when the RIO wants them/reset origin when we start aiming
#   - 3D transformation matrix
#   - Random crashing when sides/corners are covered


import cv2
from cv2 import cv
import time
import numpy as np
from enum import Enum
import platform
import math
import sys
import logging
from networktables import NetworkTable

#####################
##### CONSTANTS #####
#####################

# mode
LIVE = True
SHOW_IMAGE = True
DRAW = True
WAIT_FOR_CONTINUE = False
WAIT_TIME = 25
START_FRAME = 0
WINDOW_NAME = "Debug Display"

# controls (ascii)
EXIT = 27
CONTINUE = 32

# logging
STREAM = sys.stdout
LEVEL = logging.DEBUG

# networktables
TABLE_NAME = "jetson_table"
STATE_JETSON = "jetson_state"
class Jetson(Enum):
    POWERED_ON = 1
    CAMERA_ERROR = 2
    TARGET_FOUND = 3
    TARGET_NOT_FOUND = 4

# camera settings
CAM_INDEX = 0
RESOLUTION_X = 640
RESOLUTION_Y = 480
EXPOSURE_ABS = 10

# image processing settings
THRESH_LOW = np.array([70, 75, 75])
THRESH_HIGH = np.array([80, 255, 255])
MORPH_KERNEL_WIDTH = 3
MORPH_KERNEL_HEIGHT = 3
POLY_EPS = 0.1

# accepted error
MIN_TARGET_AREA = 0.003 * RESOLUTION_X * RESOLUTION_Y
MAX_TARGET_AREA = 0.3 * RESOLUTION_X * RESOLUTION_Y
MIN_NORM_TVECS = 0.0001
MAX_NORM_TVECS = 1000
BAD_ESTIMATE = 8

# drawing settings
LINE_WIDTH = 2
TEXT_SIZE = 1
TEXT_STROKE = 2

# colors
WHITE = (255, 255, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)

# experimentally determined camera (intrinsic) and distortion matrices, converted to numpy arrays
mtx = [[ 771.,      0.,    320.],
       [   0.,    771.,    240.],
       [   0.,      0.,      1.]]
mtx = np.asarray(mtx)
dist = [[ 0.03236637, -0.03763916, -0.00569912, -0.00091719, -0.008543  ]]
dist = np.asarray(dist)

# object points array
objp = np.array([[12, -12,  0],
                 [12,  12,  0],
                 [-12, 12,  0],
                 [-12,-12,  0]], dtype=float)

# axis for drawing the debug representation
axis = np.array([[ 0,  0,  0],
                 [12,  0,  0],
                 [ 0,  12, 0],
                 [ 0,  0, 12]], dtype=float)

newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(RESOLUTION_X, RESOLUTION_Y),1,(RESOLUTION_X, RESOLUTION_Y))

# pixels to degrees
fov = 83 # diagonal fov
ptd = fov / math.sqrt(math.pow(RESOLUTION_X, 2) + math.pow(RESOLUTION_Y, 2))



#####################
####### INIT ########
#####################

# initialize logging
logging.basicConfig(stream=STREAM, level=LEVEL)
log = logging.getLogger(__name__)
log.info("OpenCV %s", cv2.__version__)

# initialize networktables
table = NetworkTable.getTable(TABLE_NAME)
table.putString(STATE_JETSON, Jetson.POWERED_ON)
log.info("Sent powered on message on table %s", TABLE_NAME)

# capture init
cap = cv2.VideoCapture(CAM_INDEX)
log.info("Loaded capture from %s %s", "index" if LIVE else "source file", CAM_INDEX)
if not LIVE:
    cap.set(cv.CV_CAP_PROP_POS_FRAMES, START_FRAME)
    log.info("Set position to frame %s", START_FRAME)

# set the resolution
cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, RESOLUTION_X)
cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, RESOLUTION_Y)
log.info("Set resolution to %sx%s", RESOLUTION_X, RESOLUTION_Y)

# find out if the camera is actually working
if cap.isOpened():
    rval, frame = cap.read()

    # run some configuration if everything is good
    if rval:
        log.info("Read from capture successfully")
        # run config using v4l2 driver if the platform is linux and the feed is live
        if platform.system() == "Linux" and LIVE:
            log.info("Running Linux config using v4l2ctl")
            import v4l2ctl
            v4l2ctl.restore_defaults(CAM_INDEX)
            v4l2ctl.set(CAM_INDEX, v4l2ctl.PROP_EXPOSURE_AUTO, 1)
            v4l2ctl.set(CAM_INDEX, v4l2ctl.PROP_EXPOSURE_AUTO_PRIORITY, 0)
            v4l2ctl.set(CAM_INDEX, v4l2ctl.PROP_EXPOSURE_ABS, EXPOSURE_ABS)
            v4l2ctl.set(CAM_INDEX, v4l2ctl.PROP_WHITE_BALANCE_TEMP_AUTO, 0)
            v4l2ctl.set(CAM_INDEX, v4l2ctl.PROP_FOCUS_AUTO, 0)
    else:
        rval = False
        log.critical("Problem reading from capture")
        table.putString(STATE_JETSON, Jetson.CAMERA_ERROR)


else:
    rval = False
    log.critical("Problem opening capture")
    table.putString(STATE_JETSON, Jetson.CAMERA_ERROR)

# vars for calculating fps
frametimes = list()
last = time.time()


#####################
### FUNCTIONALITY ###
#####################

# draws a 3d axis on an image (calculated from pose estimation)
def draw_axis(img, rvecs, tvecs):
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    corner = tuple(np.array(imgpts[0].ravel(), dtype=int))
    img = cv2.line(img, corner, tuple(np.array(imgpts[1].ravel(), dtype=int)), CYAN, LINE_WIDTH)
    img = cv2.line(img, corner, tuple(np.array(imgpts[2].ravel(), dtype=int)), MAGENTA, LINE_WIDTH)
    img = cv2.line(img, corner, tuple(np.array(imgpts[3].ravel(), dtype=int)), YELLOW, LINE_WIDTH)


# draws a polygon on an image
def draw_poly(img, polyp):
    l = len(polyp)
    for i in range(0, l):
        if i + 1 == l:
            img = cv2.line(img, tuple(polyp[i].ravel()), tuple(polyp[0].ravel()), BLUE, LINE_WIDTH);
        else:
            img = cv2.line(img, tuple(polyp[i].ravel()), tuple(polyp[i + 1].ravel()), BLUE, LINE_WIDTH);


# draw rvecs (the numbers) on an image
def draw_rvecs(img, rvecs):
    cv2.putText(img, "rvecs:", (300, 580), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, GREEN, TEXT_STROKE, 9)
    cv2.putText(img, str(rvecs[0]), (300, 620), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, GREEN, TEXT_STROKE, 9)
    cv2.putText(img, str(rvecs[1]), (300, 660), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, GREEN, TEXT_STROKE, 9)
    cv2.putText(img, str(rvecs[2]), (300, 700), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, GREEN, TEXT_STROKE, 9)


# draw tvecs (the numbers) on an image
def draw_tvecs(img, tvecs):
    cv2.putText(img, "tvecs:", (10, 580), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, RED, TEXT_STROKE, 9)
    cv2.putText(img, str(tvecs[0]), (10, 620), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, RED, TEXT_STROKE, 9)
    cv2.putText(img, str(tvecs[1]), (10, 660), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, RED, TEXT_STROKE, 9)
    cv2.putText(img, str(tvecs[2]), (10, 700), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, RED, TEXT_STROKE, 9)


# thresholds and edge detects image; returns the result after masking and a list of contours
def process_frame(frame):
    # convert to hsv colorspace
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # threshold
    mask = cv2.inRange(hsv, THRESH_LOW, THRESH_HIGH)
    # remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_WIDTH, MORPH_KERNEL_HEIGHT))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    res = mask.copy()
    # get a list of continuous lines in the image
    contours, _ = cv2.findContours(mask, 1, 2)

    return res, contours


# finds the target in a list of contours, returns a matrix with the target polygon
def find_target(contours):
    if len(contours) > 0:
        # find the polygon with the largest area
        best_area1 = 0
        best_area2 = 0
        target1 = None
        target2 = None
        for c in contours:
            area = cv2.contourArea(c)
            if area > best_area1:
                best_area2 = best_area1
                target2 = target1
                best_area1 = area
                target1 = c
            elif area > best_area2:
                best_area2  = area
                target2 = c

        if target1 is not None and target2 is not None:
            hull = cv2.convexHull(np.concatenate((target1, target2)))
            return hull
    return None


# estimates the pose of a target, returns rvecs and tvecs
def estimate_pose(target):
    # fix array dimensions (aka unwrap the double wrapped array)
    new = []
    for r in target:
        new.append([r[0][0], r[0][1]])
    imgp = np.array(new, dtype=np.float64)

    # calculate rotation and translation matrices
    _, rvecs, tvecs = cv2.solvePnP(objp, imgp, mtx, dist)

    if cv2.norm(np.array(tvecs)) < MIN_NORM_TVECS or cv2.norm(np.array(tvecs)) > MAX_NORM_TVECS:
        tvecs = None
    if math.isnan(rvecs[0]):
        rvecs = None
    return rvecs, tvecs


#####################
##### MAIN LOOP #####
#####################

if __name__ == "__main__":
    log.info("Entering main loop")
    # loop for as long as we're still getting images
    while rval:
        # read the frame
        rval, frame = cap.read()
        # undistort
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        frame = dst


        res, contours = process_frame(frame)
        target = find_target(contours)
        if target is not None:
            cv2.drawContours(frame, [target], 0, (0, 255, 0), 3)
            M = cv2.moments(target)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.drawContours(frame, [np.array([[cx, cy]])], 0, (0, 0, 255), 10)

            distance_from_center = cx - (RESOLUTION_X / 2)
            angle = distance_from_center * ptd
            log.debug('Angle from target: ' + str(angle))
            cv2.putText(frame, str(angle), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, RED, TEXT_STROKE, 9)

        else:
            pass

        # calculate fps
        frametimes.append(time.time() - last)
        if len(frametimes) > 60:
            frametimes.pop(0)
        fps = int(1 / (sum(frametimes) / len(frametimes)))

        # draw fps
        if DRAW:
            cv2.putText(frame, str(fps), (10, 40), cv.CV_FONT_HERSHEY_SIMPLEX, TEXT_SIZE, YELLOW, TEXT_STROKE, 8)

        if SHOW_IMAGE:
            scale = 1.48
            frame = cv2.resize(frame, (int(RESOLUTION_X * (scale + 0.02)), int(RESOLUTION_Y * scale)), interpolation=cv2.INTER_CUBIC)
            cv2.imshow(WINDOW_NAME, res)
            cv2.imshow('debug', frame)
        key = cv2.waitKey(WAIT_TIME)
        if WAIT_FOR_CONTINUE:
            while key != EXIT and key != CONTINUE:
                key = cv2.waitKey(1)
        if key == EXIT:  # exit on ESC
            break

        # record time for fps calculation
        last = time.time()

    log.info("Main loop exited successfully")
    log.info("FPS at time of exit: %s", fps)
    cv2.destroyWindow(WINDOW_NAME)
