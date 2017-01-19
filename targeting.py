import cv2
import platform


index = 1
cap = cv2.VideoCapture(index)
if cap.isOpened():
    rval, frame = cap.read()

    # run some configuration if everything is good
    if rval:
        # run config using v4l2 driver if the platform is linux and the feed is live
        if platform.system() == "Linux":
            import v4l2ctl
            v4l2ctl.restore_defaults(index)
            v4l2ctl.set(index, v4l2ctl.PROP_EXPOSURE_AUTO, 1)
            v4l2ctl.set(index, v4l2ctl.PROP_EXPOSURE_AUTO_PRIORITY, 0)
            v4l2ctl.set(index, v4l2ctl.PROP_EXPOSURE_ABS, 50)
            v4l2ctl.set(index, v4l2ctl.PROP_WHITE_BALANCE_TEMP_AUTO, 0)
            v4l2ctl.set(index, v4l2ctl.PROP_FOCUS_AUTO, 0)
    else:
        rval = False

while rval:
    rval, frame = cap.read()

    cv2.imshow("dbg", frame)
    key = cv2.waitKey(10)
    if key == 27:  # exit on ESC
        break