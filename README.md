# FRC VisionWorks2017

Machine vision suite to be run on Team 2342's Jetson-TK1 coprocessor for 2017 FIRST Steamworks competitions. The name can be interpreted as either as a play on the competition name (steamworks, visionworks, haha) or as a prayer.

## Environment Setup

Our project uses Python 2 and OpenCV 2.4 as per the Jetson-TK1's supported software. We also use v4l2-ctl to control camera settings. On Ubuntu 14.04:

    sudo apt install python python-numpy python-enum34 python-opencv v4l-utils
    sudo pip install pynetworktables

Please note that the above lines are untested and I'm not sure if it will actually install everything you need, so take it with a grain of salt.
