import cv2
from flask import Flask, request, render_template, redirect, url_for, send_file
import Magic
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

def image_predict(img_path) :
    img = cv2.imread(img_path)
    boxes = detect.run(weights='yolov5/best.pt', source=img_path)


    for box in boxes:
        x1, y1, x2, y2 = post_process(img, box[1:])
        if box[0] == 0:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        elif box[0] == 1:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    ret, buffer = cv2.imencode('.jpg', img)
    img = buffer.tobytes()
    return img


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/image', methods = ['POST', 'GET'])
def image() :
    if request.method == 'POST' :
        return redirect(url_for('index.html'))
    return render_template("image.html")


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        img_name = request.form['filename']

    img_path = os.path.realpath(img_name)
    prediction = image_predict(img_path)

    return render_template('image result.html', img_data = prediction)






if __name__ == '__main__' :
    app.run(debug=True)


