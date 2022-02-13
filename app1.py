from flask import Flask,render_template,Response, request, url_for, redirect
import cv2
from yolov5 import detect

app=Flask(__name__)
camera=cv2.VideoCapture(0)

def post_process(img, box) :
    width,height,channel = img.shape
    x,y,h,w = box
    x1 = int((x-h/2)*height)
    y1 = int((y-w/2)*width)
    x2 = int((x + h/2)*height)
    y2 = int((y + w/2)*width)
    
    return x1, y1, x2, y2

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if not success:
            break
        else:
            cv2.imwrite('yolov5/img.png',frame)
            boxes = detect.run(weights='yolov5/best.pt', source = 'img.png')
            for box in boxes:
                x1, y1, x2, y2 = post_process(frame, boxes[1:])
                if box[0] == 0 :
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                elif box[0] == 1:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                else :
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image', methods = ['POST', 'GET'])
def image() :
    if request.method == 'POST' :
        return redirect(url_for('index.html'))
    return render_template("image.html")


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=="__main__":
    app.run(debug=True)