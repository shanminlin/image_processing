# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 18:51:49 2020

@author: SS
"""
import cv2                                     
import numpy as np

def face_detector(image_path):
    """Detects faces in an image.
    Args:
        img_path: a string, the file path of the image
    Return:
        faces: numpy 2d array, each row is a face with values specifying a bounding box.
    """
    # load BGR image
    # for this opencv model, we need to convert color images to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return faces

def draw_bounding_box(image_path):
    faces = face_detector(image_path)
    print(faces)
    image = cv2.imread(image_path)
    
    # Add bounding box to each detected face in color image
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # display the image, along with bounding box
    cv2.imshow('Image', image)
    cv2.imwrite('output_images/face_detection.jpg', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # extract pre-trained model
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    path = 'input_images/2.jpg'
    draw_bounding_box(path)
