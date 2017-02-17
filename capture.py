import cv2
from cv2 import cv
import config
import logging
import communications as comms
from communications import States
import platform
import v4l2ctl

logging.basicConfig(stream=config.LOG_STREAM, level=config.LOG_LEVEL)
log = logging.getLogger(__name__)

class Capture(object):
    frame = None
    jpeg = None
    rval = None


    def __init__(self, source):
        try:
            self.video = cv2.VideoCapture(source)
            self.rval = True
        except cv2.error:
            self.rval = False
            comms.set_state(States.CAMERA_ERROR)
        self.source = source
        self.live = True if isinstance(source, int) else False
        self.__configure()

    def __del__(self):
        self.video.release()

    def update(self):
        self.rval, self.frame = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.

    def set_jpeg(self, frame):
        self.jpeg = cv2.imencode('.jpg', frame)[1].tobytes()

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
                        log.info("Running Linux config using v4l2ctl")
                        v4l2ctl.restore_defaults(self.source)
                        for prop in config.CAMERA_V4L_SETTINGS:
                            v4l2ctl.set(self.source, prop, config.CAMERA_V4L_SETTINGS[prop])
                else:
                    self.rval = False
                    log.critical("Problem reading from capture")
                    comms.set_state(States.CAMERA_ERROR)

            else:
                self.rval = False
                log.critical("Problem opening capture")

                comms.set_state(States.CAMERA_ERROR)
