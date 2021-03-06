import cv2
from cv2 import cv
import time
import numpy as np
import math
import logging
import communications as comms
import config
from config import Mode, State
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
#shooter_thresh_low = config.SHOOTER_THRESH_LOW
#shooter_thresh_high = config.SHOOTER_THRESH_HIGH
gear_thresh_low = config.GEAR_THRESH_LOW
gear_thresh_high = config.GEAR_THRESH_HIGH
morph_kernel_width = config.MORPH_KERNEL_WIDTH
morph_kernel_height = config.MORPH_KERNEL_HEIGHT
poly_eps = config.POLY_EPS
rel_split_eps = config.RELATIVE_SPLIT_CONTOUR_EPSILON

# experimentally determined camera (intrinsic) and distortion matrices
mtx = config.CAMERA_MATRIX
dist = config.CAMERA_DISTORTION_MATRIX

max_norm_tvecs = config.MAX_NORM_TVECS
min_norm_tvecs = config.MIN_NORM_TVECS
#max_shooter_area = config.MAX_SHOOTER_AREA
#min_shooter_area = config.MIN_SHOOTER_AREA
max_gears_area = config.MAX_GEARS_AREA
min_gears_area = config.MIN_GEARS_AREA
gears_objp = config.GEARS_OBJP


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
comms.set_gear_state(State.POWERED_ON)

# capture init
gear_cam_server = Capture(config.VIDEO_SOURCE_GEAR, Mode.GEARS)


# estimates the pose of a target, returns rvecs and tvecs
def estimate_pose(target):
    # fix array dimensions (aka unwrap the double wrapped array)
    new = []
    for r in target:
        new.append([r[0][0], r[0][1]])
    imgp = np.array(new, dtype=np.float64)

    # calculate rotation and translation matrices
    _, rvecs, tvecs = cv2.solvePnP(gears_objp, imgp, mtx, dist)

    if cv2.norm(np.array(tvecs)) < min_norm_tvecs or cv2.norm(np.array(tvecs)) > max_norm_tvecs:
        tvecs = None
    if math.isnan(rvecs[0]):
        rvecs = None
    return rvecs, tvecs


def gear_targeting(hsv):
    # threshold
    mask = cv2.inRange(hsv, gear_thresh_low, gear_thresh_high)

    # remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    res = mask.copy()

    # get a list of continuous lines in the image
    contours, _ = cv2.findContours(mask, 1, 2)

    # identify the target in the image
    target = None
    if len(contours) > 0:
        # find the polygon with the largest area
        # find the two biggest contours
        areas = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > min_gears_area and area < max_gears_area:
                areas.append((area, c))
        areas.sort(key=lambda x: -x[0])

        if len(areas) == 2:
            target = cv2.convexHull(np.concatenate((areas[0][1], areas[1][1])))

        # if one of the sides is cut in half
        elif len(areas) > 2:
            half_area = areas[0][0] / 2
            upper = half_area * (1 + rel_split_eps)
            lower = half_area * (1 - rel_split_eps)
            area1_within_bounds = areas[1][0] < upper and areas[1][0] > lower
            area2_within_bounds = areas[2][0] < upper and areas[2][0] > lower
            if area1_within_bounds and area2_within_bounds:
                target = cv2.convexHull(np.concatenate((areas[0][1], areas[1][1], areas[2][1])))

        if target is not None:
            e = poly_eps * cv2.arcLength(target, True)
            target = cv2.approxPolyDP(target, e, True)
            correct_number_of_sides = len(target) == len(gears_objp)
            target_within_bounds = True
            for p in target:
                # note, array is double wrapped, that's why accessing x and y values here is weird
                if p[0][0] > res_x - 3 or p[0][0] <= 1 or p[0][1] > res_y - 3 or p[0][1] <= 1:
                    target_within_bounds = False
                    break

            if not correct_number_of_sides or not target_within_bounds:
                target = None

    if target is not None:
        # set state
        comms.set_gear_state(State.TARGET_FOUND)

        # fix origin
        index = None
        best_cost = 1000000
        for i, p in enumerate(target):
            cost = (p[0][0] + p[0][1])
            if cost < best_cost:
                best_cost = cost
                index = i

        target = np.append(target, target[:index], 0)[index:]

        #Calculate the rvecs and tvecs
        rvecs, tvecs = estimate_pose(target)

        #rlen = np.linalg.norm(rvecs)
        #rvecs *= (rlen - 12*math.pi/180) / rlen
        #Rotation matrix derived from rodrigues vector rvec
        R, _ = cv2.Rodrigues(rvecs)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        y = math.atan2(-R[2, 0], sy) #Rotation of target relative to camera

        #The angle off normal of the camera mounted on the robot (to be 12 degrees)
        cam_angle = -0 * math.pi / 180

        #Alpha is the angle that the target is observed at by the camera
        alpha = math.atan(tvecs[0] / tvecs[2])

        #tlen is the distance from the camera to the center of the target
        tlen = math.sqrt(tvecs[0]*tvecs[0] + tvecs[2]*tvecs[2])

        #calculate the shift that the gear should move to be in line with the peg
        shift = tlen * (math.sin(cam_angle + alpha) + math.cos(cam_angle + alpha) * math.tan(- y - cam_angle))

        #Additionally, shift the center of the robot to be in line with the peg (note that the gear will no longer be in line)
        shift += 14.5 * math.tan( - y - cam_angle)

        rotation = (cam_angle - y) / math.pi * 180
        #horizontal = -float(tvecs[0]) - 7
	#Add a factor for the displacement of the gear from the camera
        horizontal = shift + 7.875
        forward = float(tvecs[2]) #FIX APPROACH

        comms.set_gear(rotation, horizontal, forward)

        if draw:
            cv2.drawContours(frame, [target + 12], 0, (0, 255, 0), 3)
            M = cv2.moments(target)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.drawContours(frame, [np.array([[cx + 12, cy + 12]])], 0, (0, 0, 255), 10)

            cv2.putText(frame, "rotation: " + str(rotation), (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 9)
            cv2.putText(frame, "alpha: " + str(alpha / math.pi * 180), (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 9)
            cv2.putText(frame, "horizontal: " + str(horizontal), (100, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 9)
            cv2.putText(frame, "tlen: " + str(tlen), (100, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, 9)
    else:
        comms.set_gear_state(State.TARGET_NOT_FOUND)


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


#config.SERVER_MODE = int(sys.argv[2])
run_server = config.USE_HTTP_SERVER and len(sys.argv) > 1
if run_server:
    port = int(sys.argv[1])
    thread.start_new_thread(feed.init, (gear_cam_server, port))

log.info("Starting vision processing loop")
# loop for as long as we're still getting images
while True:

    try:
        if gear_cam_server.rval:
            gear_cam_server.update()
            rval, frame = gear_cam_server.rval, gear_cam_server.frame
            hsv = basic_frame_process(frame)
            gear_targeting(hsv)
            if draw:
                cv2.putText(frame, str(fps), (10, 40), cv.CV_FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, 8)
            if run_server:
                gear_cam_server.set_jpeg(frame)

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

comms.set_high_goal_state(State.POWERED_OFF)
comms.set_gear_state(State.POWERED_OFF)
log.info("Main loop exited successfully")
log.info("FPS at time of exit: %s", fps)
