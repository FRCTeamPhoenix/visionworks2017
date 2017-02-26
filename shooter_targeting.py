import cv2
from cv2 import cv
import time
import numpy as np
import math
import logging
import communications as comms
import config
from config import Modes, States
import feed
import thread
from capture import Capture
import sys


# gui config/mode stuff (see config.py for details)
show_image = config.GUI_SHOW
draw = config.GUI_DEBUG_DRAW
wait_for_continue = config.GUI_WAIT_FOR_CONTINUE
wait_time = config.WAIT_TIME
start_frame = config.START_FRAME

# controls (ascii)
exit_key = config.GUI_EXIT_KEY
continue_key = config.GUI_WAIT_FOR_CONTINUE_KEY

# camera settings
res_x = config.RESOLUTION_X
res_y = config.RESOLUTION_Y

# image processing settings
shooter_thresh_low = config.SHOOTER_THRESH_LOW
shooter_thresh_high = config.SHOOTER_THRESH_HIGH
morph_kernel_width = config.MORPH_KERNEL_WIDTH
morph_kernel_height = config.MORPH_KERNEL_HEIGHT
poly_eps = config.POLY_EPS
rel_split_eps = config.RELATIVE_SPLIT_CONTOUR_EPSILON

# experimentally determined camera (intrinsic) and distortion matrices
mtx = config.CAMERA_MATRIX
dist = config.CAMERA_DISTORTION_MATRIX

max_norm_tvecs = config.MAX_NORM_TVECS
min_norm_tvecs = config.MIN_NORM_TVECS
max_shooter_area = config.MAX_SHOOTER_AREA
min_shooter_area = config.MIN_SHOOTER_AREA


# target camera matrix for
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (res_x, res_y), 1, (res_x, res_y))

# pixels to degrees conversion factor
ptd = config.CAMERA_DIAG_FOV / math.sqrt(math.pow(res_x, 2) + math.pow(res_y, 2))
focal_length = math.sqrt(res_x**2 + res_y**2) / 2 / math.tan(0.5 * config.CAMERA_DIAG_FOV * math.pi / 180)

# initialize logging
logging.basicConfig(stream=config.LOG_STREAM, level=config.LOG_LEVEL)
log = logging.getLogger(__name__)
log.info('OpenCV %s', cv2.__version__)

# send confirmation that we're alive
comms.set_high_goal_state(States.POWERED_ON)

# capture init
turret_cam_server = Capture(config.VIDEO_SOURCE_TURRET, Modes.HIGH_GOAL)


def high_goal_targeting(hsv, turret_angle):
    # threshold
    mask = cv2.inRange(hsv, shooter_thresh_low, shooter_thresh_high)


    # remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    res = mask.copy()

    # get a list of continuous lines in the image
    contours, _ = cv2.findContours(mask, 1, 2)

    # identify the target if there is one
    target = None
    if len(contours) > 0:
        # find the two biggest contours
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
                best_area2 = area
                target2 = c
        target = target1

    if best_area1 > 0 and best_area2 > 0:
        target = cv2.convexHull(np.concatenate((target1, target2)))

    # if we found a target
    if target is not None:
        # set state
        comms.set_high_goal_state(States.TARGET_FOUND)

        # find the centroid of the target
        M = cv2.moments(target)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # calculate the angle needed in order to align the target
        distance_from_center = (res_x / 2) - cx
        angle = distance_from_center * ptd # pixel distance * conversion factor

        # send the (absolute) angle to the RIO
        comms.set_high_goal(turret_angle + angle)

        # draw debug information about the target on the frame
        if draw:
            cv2.drawContours(frame, [target], 0, (0, 255, 0), 3)
            cv2.drawContours(frame, [np.array([[cx, cy]])], 0, (0, 0, 255), 10)
            cv2.putText(frame, str(angle), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 9)
    else:
        comms.set_high_goal_state(States.TARGET_NOT_FOUND)


def basic_frame_process(frame):
    # undistort the image
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # crop the image after undistortion
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    frame = dst

    # convert to hsv colorspace
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


# vars for calculating fps
frametimes = list()
last = time.time()
fps = 0

run_server = config.USE_HTTP_SERVER and len(sys.argv) > 1
if run_server:
    port = int(sys.argv[1])
    thread.start_new_thread(feed.init, (turret_cam_server, port))

log.info("Starting vision processing loop")
# loop for as long as we're still getting images
while True:

    try:
        if turret_cam_server.rval:
            turret_cam_server.update()
            turret_angle = comms.get_turret_angle()
            rval, frame = turret_cam_server.rval, turret_cam_server.frame

            # TESTING
            if turret_angle == None:
                turret_angle = 0
            # ENDTESTING

            if turret_angle is not None:
                hsv = basic_frame_process(frame)
                high_goal_targeting(hsv, turret_angle)
            if draw:
                cv2.putText(frame, str(fps), (10, 40), cv.CV_FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, 8)
            if config.USE_HTTP_SERVER:
                turret_cam_server.set_jpeg(frame)

        # calculate fps
        frametimes.append(time.time() - last)
        if len(frametimes) > 60:
            frametimes.pop(0)
        fps = int(1 / (sum(frametimes) / len(frametimes)))

        if show_image:
            cv2.imshow('frame', frame)

        key = cv2.waitKey(wait_time)
        if wait_for_continue:
            while key != exit_key and key != continue_key:
                key = cv2.waitKey(1)
        if key == exit_key:  # exit on ESC
            break

        # record time for fps calculation
        last = time.time()

    except Exception as e:
        # in real life situations on the field, we want to continue even if something goes really wrong.
        # just keep looping :)
        print(e)

comms.set_high_goal_state(States.POWERED_OFF)
comms.set_gear_state(States.POWERED_OFF)
log.info("Main loop exited successfully")
log.info("FPS at time of exit: %s", fps)
