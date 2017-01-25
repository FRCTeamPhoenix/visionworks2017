import cv2
from cv2 import cv
import config

class Capture(object):
    frame = None
    jpeg = None
    rval = None

    def __init__(self):
        self.video = cv2.VideoCapture(config.VIDEO_SOURCE)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def update(self):
        self.rval, self.frame = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        #self.jpeg = cv2.imencode('.jpg', self.frame)[1].tobytes()

    def set_jpeg(self, frame):
        self.jpeg = cv2.imencode('.jpg', frame)[1].tostring()
