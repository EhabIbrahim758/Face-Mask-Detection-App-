from yolov5 import detect as dec
import cv2
import glob
from PIL import Image
import os


def post_process(img, box):  # rescale the bbox coord after normalization
    width, height, channel = img.shape
    x, y, h, w = box
    x1 = int((x - h / 2) * height)
    y1 = int((y - w / 2) * width)
    x2 = int((x + h / 2) * height)
    y2 = int((y + w / 2) * width)

    return x1, y1, x2, y2


def convert(image):  # convert image format because yolo accepts only png
    path = image.split('.')[0]
    im1 = Image.open(image)
    im1.save(path+'.png')
    path += '.png'
    return path


def infer_images(weights, glob_):  # infer on images in a path ,default yolov5/dataset/images
    _ = dec.run(weights=weights, source=glob_)


def get_bb(weights, _globe):  # infer on images in a path and get bbx, default yolov5/dataset/images
    _ = dec.run(weights=weights, source=_globe, save_txt=True)


def infer(weights):  # infer on the images folder in yolov5/dataset/
    _ = dec.run(weights=weights)


def infer_webcam_or_vid(weights, source=0):  # Run webcam inference
    _ = dec.run(weights=weights, source=source)


def infer_single_image(weights, img_path):
    if img_path[-3:] != "png":
        img_path = convert(img_path)
    _ = dec.run(weights=weights, source=img_path)


if __name__ == '__main__':

    weihgts = '' # best.pt path

    folder = glob.glob("yolov5/images/*") # add your images folder path here
    image_path = '' # add image path
    video_path = '' #add image path
    infer_webcam_or_vid(weihgts)
    # infer_single_image(weihgts, image_path)
