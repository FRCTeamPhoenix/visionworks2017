import subprocess

__v4l2ctl__ = "v4l2-ctl"
PROP_BRIGHTNESS = "brightness"
PROP_CONTRAST = "contrast"
PROP_SATURATION = "saturation"
PROP_HUE = "hue"
PROP_WHITE_BALANCE_TEMP_AUTO = "white_balance_temperature_auto"
PROP_GAMMA = "gamma"
PROP_POWER_LINE_FREQUENCY = "power_line_frequency"
PROP_WHITE_BALANCE_TEMP = "white_balance_temperature"
PROP_SHARPNESS = "sharpness"
PROP_BACKLIGHT_COMPENSATION = "backlight_compensation"
PROP_EXPOSURE_AUTO = "exposure_auto"
PROP_EXPOSURE_ABS = "exposure_absolute"
PROP_EXPOSURE_AUTO_PRIORITY = "exposure_auto_priority"
PROP_FOCUS_AUTO = "focus_auto"
PROP_FOCUS_ABSOLUTE = "focus_absolute"

def __run__(command):
    p = subprocess.Popen(command, stdout=subprocess.PIPE)
    p.wait()
    return p.communicate()

def get(cami, prop):
    props = __get_props__(cami)
    if not prop in props:
        raise AttributeError("Property not supported for camera!")
    command = [__v4l2ctl__, "--device", "/dev/video" + str(cami), "--get-ctrl", prop]
    out, err = __run__(command)
    return out.decode().strip().split(" ")[1]

def set(cami, prop, val):
    props = __get_props__(cami)
    if not prop in props:
        raise AttributeError("Property not supported for camera!")
    if val > props[prop]["max"] or val < props[prop]["min"]:
        raise ValueError("Value not in range for property! (" + str(props[prop]["min"]) + str(props[prop]["max"]) + ")")
    command = [__v4l2ctl__, "--device", "/dev/video" + str(cami), "--set-ctrl", prop + "=" + str(val)]
    #print(command)
    __run__(command)

def restore_defaults(cami):
    props = __get_props__(cami)

    # exceptions
    if PROP_EXPOSURE_AUTO in props and PROP_EXPOSURE_ABS in props:
        set(cami, PROP_EXPOSURE_AUTO, 1)
        set(cami, PROP_EXPOSURE_ABS, props[PROP_EXPOSURE_ABS]["default"])
    if PROP_WHITE_BALANCE_TEMP_AUTO in props and PROP_WHITE_BALANCE_TEMP in props:
        set(cami, PROP_WHITE_BALANCE_TEMP_AUTO, 0)
        set(cami, PROP_WHITE_BALANCE_TEMP, props[PROP_WHITE_BALANCE_TEMP]["default"])

    command = [__v4l2ctl__, "--device", "/dev/video" + str(cami)]
    for p in props:
        if p == PROP_EXPOSURE_ABS or p == PROP_WHITE_BALANCE_TEMP:
            continue
        command.append("--set-ctrl")
        command.append(p + "=" + str(props[p]["default"]))
    __run__(command)

def __has_v4l2__():
    try:
        subprocess.check_output(__v4l2ctl__,
                                stderr=subprocess.STDOUT,
                                shell=True)
        return True;
    except subprocess.CalledProcessError as ctlexec:
        return False;


def __get_props__(cami):
    props = dict()
    command = [__v4l2ctl__, "--device", "/dev/video" + str(cami), "-l"]
    out, err = __run__(command)
    lines = out.decode().split("\n")
    lines = [l.strip() for l in lines]
    lines = [l.split(":") for l in lines]

    for l in lines:

        if len(l) != 2:
            continue

        name = l[0].split(" ")[0]
        type = l[0].split("(")[1].split(")")[0]

        props[name] = {"type": type, "min": None, "max": None, "default": None}
        l[1] = l[1].strip().split(" ")
        for p in l[1]:
            attr = p.split("=")[0]
            if attr == "step" or attr == "value" or attr == "flags":
                continue
            props[name][attr] = int(p.split("=")[1])
        if props[name]["type"] == "bool":
            props[name]["min"] = 0
            props[name]["max"] = 1

    return props