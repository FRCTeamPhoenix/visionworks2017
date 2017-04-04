from flask import Flask, render_template, Response
import time

app = Flask(__name__)
cam = None

@app.route('/')
def index():
    return render_template('./index.html')


def gen():
    while True:
        frame = cam.jpeg
        time.sleep(0.05)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen2():
    while True:
        frame = cam.jpeg2
        time.sleep(0.05)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed2')
def video_feed2():
    return Response(gen2(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def init(cam_server, port):
    global cam
    cam = cam_server
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
