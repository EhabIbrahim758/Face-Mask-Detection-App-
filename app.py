import cv2
from flask import Flask, request, render_template, redirect, url_for, send_file, Response

import os
import io
import base64
from yolov5 import detect

app = Flask(__name__)

def post_process(img, box):
    width, height, channel = img.shape
    x, y, h, w = box
    x1 = int((x - h / 2) * height)
    y1 = int((y - w / 2) * width)
    x2 = int((x + h / 2) * height)
    y2 = int((y + w / 2) * width)

    return x1, y1, x2, y2

def put_text(img, text, color, org, img_predict = True) :
    font = cv2.FONT_HERSHEY_SIMPLEX
    if img_predict :
        fontscale = 2
    else :
        fontscale = 1
    cv2.putText(img, text, org, font,
                        fontscale, color, thickness = 2)
    return img



def image_predict(img_path) :
    img = cv2.imread(img_path)
    img_size = img.shape[:2]
    boxes = detect.run(weights='yolov5/best.pt', source=img_path)
    text = ['with_mask', 'without_mask', 'incorrect_mask']

    if len(boxes) > 0 :
        for box in boxes:
            x1, y1, x2, y2 = post_process(img, box[1:])
            if box[0] == 0:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                img = put_text(img, text[0], (0, 0, 255), (x1-5, y1-1))
            elif box[0] == 1:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                img = put_text(img, text[1], (0, 0, 255), (x1-5, y1 - 1))
            elif box[0] == 2 :
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
                img = put_text(img, text[2], (0, 0, 255), (x1-5, y1 - 1))

    return img

def video_predict(video_path = None) :

    if video_path :
        video = cv2.VideoCapture(video_path)
    else :
        video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FPS, 60)
    text = ['with_mask', 'without_mask', 'incorrect_mask']
    while True:
        ## read the camera frame
        success, frame = video.read()
        if not success:
            break
        else:
            cv2.imwrite('yolov5/img.png', frame)
            boxes = detect.run(weights='yolov5/best.pt', source='yolov5/img.png')
            if len(boxes) > 0 :
                for box in boxes:
                    x1, y1, x2, y2 = post_process(frame, box[1:])

                    if box[0] == 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        frame = put_text(frame, text[0], (0, 0, 255), (x1 - 5, y1 - 1), img_predict = False)
                    elif box[0] == 1:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        frame = put_text(frame, text[1], (0, 0, 255), (x1 - 5, y1 - 1), img_predict = False)
                    elif box[0] == 2 :
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        frame = put_text(frame, text[2], (0, 0, 255), (x1 - 5, y1 - 1), img_predict = False)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/')
def index():
    return render_template("index.html")


@app.route('/image', methods = ['POST', 'GET'])
def image() :
    if request.method == 'POST' :
        return redirect(url_for('index.html'))
    return render_template("image.html")


@app.route('/video', methods = ['POST', 'GET'])
def video() :
    if request.method == 'POST' :
        return redirect(url_for('index.html'))
    return render_template("video.html")



@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST' :
        img_name = request.files['filename'].filename

    img_path = os.path.abspath(img_name)
    prediction = image_predict(img_path)
    cv2.imwrite('static/pred.png', prediction)

    return render_template('image result.html', img_data='pred.png')


@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        vid_name = request.form['filename']

    vid_path = os.path.realpath(vid_name)

    return Response(video_predict(vid_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/open_camera', methods=['GET', 'POST'])
def open_camera():
    return Response(video_predict(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__' :
    app.run(debug=True)


