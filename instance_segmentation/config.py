# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:01:38 2020

@author: SS
"""
# Set directory that contains input images and the image format 
input_image_dir = 'input_images'
input_image_format = 'jpg'

# Set directory to store output images
output_image_dir = 'output_images'

# Set directory that contains pretrained yolo model
mask_dir = 'mask_rcnn_pretrained'

# Set model hyperparameters
confidence_threshold = 0.5 # detect when probability is above the threshold
mask_threshold = 0.3 