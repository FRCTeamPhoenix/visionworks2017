# TODO
#   - Consider implementing minimum/maximum areas
#   - Check to see if the two contours are close enough to each other?
#   - Evaluate behaviour when the target is far away enough that there is only 1 contour


import cv2
from cv2 import cv
import time
import numpy as np
import platform
import math
import logging
import communications as comms
from communications import States
import v4l2ctl
import config
import feed
import thread
from capture import Capture


# gui config/mode stuff (see config.py for details)
show_image = config.GUI_SHOW
draw = config.GUI_DEBUG_DRAW
wait_for_continue = config.GUI_WAIT_FOR_CONTINUE
wait_time = config.WAIT_TIME
start_frame = config.START_FRAME

# controls (ascii)
EXIT = config.GUI_EXIT_KEY
CONTINUE = config.GUI_WAIT_FOR_CONTINUE_KEY

# camera settings
video_source = config.VIDEO_SOURCE
live = True if isinstance(video_source, int) else False
res_x = config.RESOLUTION_X
res_y = config.RESOLUTION_Y

# image processing settings
thresh_low = config.THRESH_LOW
thresh_high = config.THRESH_HIGH
morph_kernel_width = config.MORPH_KERNEL_WIDTH
morph_kernel_height = config.MORPH_KERNEL_HEIGHT

# experimentally determined camera (intrinsic) and distortion matrices
mtx = config.CAMERA_MATRIX
dist = config.CAMERA_DISTORTION_MATRIX

max_norm_tvecs = config.MAX_NORM_TVECS
min_norm_tvecs = config.min_norm_tvecs

max_target_area = config.MAX_TARGET_AREA
min_target_area = config.min_target_area


# target camera matrix for undistortion
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (res_x, res_y), 1, (res_x, res_y))

# pixels to degrees conversion factor
ptd = config.CAMERA_DIAG_FOV / math.sqrt(math.pow(res_x, 2) + math.pow(res_y, 2))

# initialize logging
logging.basicConfig(stream=config.LOG_STREAM, level=config.LOG_LEVEL)
log = logging.getLogger(__name__)
log.info('OpenCV %s', cv2.__version__)

# send confirmation that we're alive
comms.set_state(States.POWERED_ON)

# capture init
cam_server = Capture()
cap = cam_server.video
log.info('Loaded capture from %s', video_source)

# set the resolution
cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, res_x)
cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, res_y)
log.info("Set resolution to %sx%s", res_x, res_y)

# find out if the camera is actually working
if cap.isOpened():
    rval, frame = cap.read()

    # run some configuration if everything is good
    if rval:
        log.info("Read from capture successfully")
        # run config using v4l2 driver if the platform is linux and the feed is live
        if platform.system() == "Linux" and live:
            log.info("Running Linux config using v4l2ctl")
            v4l2ctl.restore_defaults(video_source)
            for prop in config.CAMERA_V4L_SETTINGS:
                v4l2ctl.set(video_source, prop, config.CAMERA_V4L_SETTINGS[prop])
    else:
        rval = False
        log.critical("Problem reading from capture")
        comms.set_state(States.CAMERA_ERROR)

else:
    rval = False
    log.critical("Problem opening capture")

    comms.set_state(States.CAMERA_ERROR)


# estimates the pose of a target, returns rvecs and tvecs
def estimate_pose(target):
    # fix array dimensions (aka unwrap the double wrapped array)
    new = []
    for r in target:
        new.append([r[0][0], r[0][1]])
    imgp = np.array(new, dtype=np.float64)

    # calculate rotation and translation matrices
    _, rvecs, tvecs = cv2.solvePnP(objp, imgp, mtx, dist)

    if cv2.norm(np.array(tvecs)) < min_norm_tvecs or cv2.norm(np.array(tvecs)) > max_norm_tvecs:
        tvecs = None
    if math.isnan(rvecs[0]):
        rvecs = None
    return rvecs, tvecs

# finds the target in a list of contours, returns a matrix with the target polygon
def gear_contours(contours):
    if len(contours) > 0:
        # find the polygon with the largest area
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

        if best_area1 > 0 and best_area2 > 0:
            target = cv2.convexHull(np.concatenate((target1, target2)))

        e = POLY_EPS * cv2.arcLength(target, True)
        target = cv2.approxPolyDP(target, e, True)

        if target is not None:
            correct_number_of_sides = len(target) == len(objp)
            area_within_range = best_area > min_target_area and best_area < max_target_area
            target_within_bounds = True
            for p in target:
                # note, array is double wrapped, that's why accessing x and y values here is weird
                if p[0][0] > res_x - 3 or p[0][0] <= 1 or p[0][1] > res_y - 3 or p[0][1] <= 1:
                    target_within_bounds = False
                    break

            if correct_number_of_sides and area_within_range and target_within_bounds:
                return target
    return None

def gear_targeting(frame):
    # threshold
    mask = cv2.inRange(hsv, thresh_low, thresh_high)

    # remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # get a list of continuous lines in the image
    contours, _ = cv2.findContours(mask, 1, 2)

    # there's only a target if there's 2+ contours
    target = gear_contours(contours)

    if target is not None:
        rvecs, tvecs = estimate_pose(target)
        #comms
    else:
        comms.set_state(States.TARGET_NOT_FOUND)

def shooter_contours(contours):
    # there's only a target if there's 2+ contours
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

    return target

def shooter_targeting(hsv):
    # threshold
    mask = cv2.inRange(hsv, thresh_low, thresh_high)

    # remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # get a list of continuous lines in the image
    contours, _ = cv2.findContours(mask, 1, 2)

    target = shooter_contours(contours)
    if target is not None:
        # set state
        comms.set_state(States.TARGET_FOUND)

        # find the centroid of the target
        M = cv2.moments(target)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # calculate the angle needed in order to align the target
        distance_from_center = cx - (res_x / 2)
        angle = distance_from_center * ptd # pixel distance * conversion factor

        # send the angle to the RIO
        comms.set_targeting(angle)

        # draw debug information about the target on the frame
        if draw:
            cv2.drawContours(frame, [target], 0, (0, 255, 0), 3)
            cv2.drawContours(frame, [np.array([[cx, cy]])], 0, (0, 0, 255), 10)
            cv2.putText(frame, str(angle), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 9)
    else:
        comms.set_state(States.TARGET_NOT_FOUND)


# vars for calculating fps
frametimes = list()
last = time.time()

cam_server.update()
thread.start_new_thread(feed.init, (cam_server,))

log.info("Starting vision processing loop")
# loop for as long as we're still getting images
while rval:

    # read the frame
    cam_server.update()
    rval, frame = cam_server.rval, cam_server.frame

    # undistort the image
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # crop the image after undistortion
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    frame = dst

    # convert to hsv colorspace
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    gear_targeting(hsv)

    # calculate fps
    frametimes.append(time.time() - last)
    if len(frametimes) > 60:
        frametimes.pop(0)
    fps = int(1 / (sum(frametimes) / len(frametimes)))

    # draw fps
    if draw:
        cv2.putText(frame, str(fps), (10, 40), cv.CV_FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, 8)

    cam_server.set_jpeg(frame)
    if show_image:
        #scale = 1.48
        #frame = cv2.resize(frame, (int(RESOLUTION_X * (scale + 0.02)), int(RESOLUTION_Y * scale)), interpolation=cv2.INTER_CUBIC)
        #cv2.imshow('res', res)
        cv2.imshow('frame', frame)
    key = cv2.waitKey(wait_time)
    if wait_for_continue:
        while key != EXIT and key != CONTINUE:
            key = cv2.waitKey(1)
    if key == EXIT:  # exit on ESC
        break

    # record time for fps calculation
    last = time.time()

comms.set_state(States.POWERED_OFF)
log.info("Main loop exited successfully")
log.info("FPS at time of exit: %s", fps)
