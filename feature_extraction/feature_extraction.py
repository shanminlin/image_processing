#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from glob import glob
from skimage import measure
import config  

def load_images():
    """Loads images and return grayscale images."""
    raw_image_paths = os.path.join(config.input_image_dir, '*.{}'.format(config.input_image_format))
    image_paths = np.array(glob(raw_image_paths))
    
    # load color image 
    all_images = []
    for path in image_paths:
        BGR_image = cv2.imread(path)
        # convert to grayscale
        gray_image = cv2.cvtColor(BGR_image, cv2.COLOR_BGR2GRAY)
        all_images.append(gray_image)
    return all_images

def detect_edge(gray_image):
    # 3x3 Sobel operator filter for edge detection
    sobel_y = np.array([[ -1, -2, -1], 
                       [ 0, 0, 0], 
                       [ 1, 2, 1]])
    
    # Below filter2D takes gray image as input
    # -1 is bit depth
    filtered_image = cv2.filter2D(gray_image, -1, sobel_y) 
    
    plt.imshow(filtered_image, cmap='gray')

def detect_contours(image):
    contours = measure.find_contours(image, 0.8)
    
    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)
    
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


if __name__ == '__main__':
    all_images = load_images()
    for image in all_images:
        detect_edge(image)
        detect_contours(image)
    
