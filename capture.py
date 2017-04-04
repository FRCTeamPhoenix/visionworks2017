import cv2
from cv2 import cv
import config
import logging
import communications as comms
from communications import State, Mode
import platform
import v4l2ctl

logging.basicConfig(stream=config.LOG_STREAM, level=config.LOG_LEVEL)
log = logging.getLogger(__name__)

class Capture(object):
    frame = None
    jpeg = None
    jpeg2 = None
    rval = None
    frame_count = 0


    def __init__(self, source, type):
        self.type = type
        try:
            self.video = cv2.VideoCapture(source)
            self.rval = True
        except cv2.error:
            self.rval = False
            if type == Mode.HIGH_GOAL:
                comms.set_high_goal_state(State.CAMERA_ERROR)
            else:
                comms.set_gear_state(State.CAMERA_ERROR)
        self.source = source
        self.live = True if isinstance(source, int) else False
        self.__configure()

    def __del__(self):
        self.video.release()

    def update(self):
        self.rval, self.frame = self.video.read()
        self.frame_count += 1
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.

    def set_jpeg(self, frame):
        self.jpeg = cv2.imencode('.jpg', frame)[1].tostring()

    def set_jpeg2(self, frame):
        self.jpeg2 = cv2.imencode('.jpg', frame)[1].tostring()

    def __configure(self):
        if self.rval:
            self.video.set(cv.CV_CAP_PROP_FRAME_WIDTH, config.RESOLUTION_X)
            self.video.set(cv.CV_CAP_PROP_FRAME_HEIGHT, config.RESOLUTION_Y)
            log.info("Set resolution to %sx%s", config.RESOLUTION_X, config.RESOLUTION_Y)
            # find out if the camera is actually working
            if self.video.isOpened():
                self.rval, frame = self.video.read()

                # run some configuration if everything is good
                if self.rval:
                    log.info("Read from capture successfully")
                    # run config using v4l2 driver if the platform is linux and the feed is live
                    if platform.system() == "Linux" and self.live:
                        try:
                            log.info("Running Linux config using v4l2ctl")
                            v4l2ctl.restore_defaults(self.source)
                            if self.type == Mode.HIGH_GOAL:
                                for prop in config.CAMERA_SETTINGS_HIGH_GOAL:
                                    v4l2ctl.set(self.source, prop, config.CAMERA_SETTINGS_HIGH_GOAL[prop])
                            else:
                                for prop in config.CAMERA_SETTINGS_GEARS:
                                    v4l2ctl.set(self.source, prop, config.CAMERA_SETTINGS_GEARS[prop])
                        except AttributeError as e:
                            log.error('Setting camera properties failed!')
                            if type == Mode.HIGH_GOAL:
                                comms.set_high_goal_state(State.CAMERA_ERROR)
                            else:
                                comms.set_gear_state(State.CAMERA_ERROR)
                            print(e)
                else:
                    self.rval = False
                    log.critical("Problem reading from capture")
                    if type == Mode.HIGH_GOAL:
                        comms.set_high_goal_state(State.CAMERA_ERROR)
                    else:
                        comms.set_gear_state(State.CAMERA_ERROR)

            else:
                self.rval = False
                log.critical("Problem opening capture")

                if type == Mode.HIGH_GOAL:
                    comms.set_high_goal_state(State.CAMERA_ERROR)
                else:
                    comms.set_gear_state(State.CAMERA_ERROR)
