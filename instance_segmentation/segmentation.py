# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 01:45:14 2020

@author: SS
"""
import numpy as np
import cv2
from glob import glob
import os
import config
import random
np.random.seed(0)

def load_images():
    """import all images in input_images folder"""
    raw_image_paths = os.path.join(config.input_image_dir, '*.{}'.format(config.input_image_format))
    image_paths = np.array(glob(raw_image_paths))
    
    all_images = []
    for path in image_paths:
        image = cv2.imread(path)
        all_images.append(image)
    return all_images

def load_labels():
    """Load labels for COCO dataset"""
    label_path = os.path.join(config.mask_dir, 'object_detection_classes_coco.txt')
    with open(label_path) as file:
        class_labels = file.read().strip().split('\n')
    return class_labels

def load_colors():
    color_path = os.path.join(config.mask_dir, 'colors.txt')
    colors = []
    with open(color_path) as file:
        colors = []
        for line in file:
            values = line.strip().split(',')
            # values are string, convert to int
            values = [int(value) for value in values]
            colors.append(values)
    colors = np.array(colors)
    return colors

def load_pretrained_model():
    """Load yolo pretrained model"""
    config_path = os.path.join(config.mask_dir, 'mask_rcnn_inception_v2_coco_2018_01_28.pbtxt')
    weight_path = os.path.join(config.mask_dir, 'frozen_inference_graph.pb')

    # load mask rcnn pretrained model
    net = cv2.dnn.readNetFromTensorflow(weight_path, config_path, )
    return net

def detect(image, net):
    """Detects objects in input image
    Args:
        image: numpy array, input image
        net: pretrained model
    Returns:
        boxes: 4 dimensional numpy array
        mask: 4 dimensional numpy array
    """
    
    # Max scaling and resize input images
    processed_image = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
    net.setInput(processed_image)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    return boxes, masks

def segment(image, image_id, boxes, mask):
    H = image.shape[0]
    W = image.shape[1]
    for i in range(boxes.shape[2]):
        class_id = int(boxes[0, 0, i, 1]) # index 1
        confidence = boxes[0, 0, i, 2] # index 2
        
        if confidence > config.confidence_threshold:
            original_image = image.copy()
            # index 3, 4, 5, 6
            start_x = int(boxes[0, 0, i, 3:7][0] * W)
            start_y = int(boxes[0, 0, i, 3:7][1] * H)
            end_x = int(boxes[0, 0, i, 3:7][2] * W)
            end_y = int(boxes[0, 0, i, 3:7][3] * H)
           
            box_width = int(end_x - start_x) # must be int
            box_height = int(end_y - start_y) # must be int
            
            mask = masks[i, class_id]
            # resize mask based on original image
            mask = cv2.resize(mask, (box_width, box_height), interpolation=cv2.INTER_NEAREST)
            mask = (mask > config.mask_threshold)
            
            roi = original_image[start_y:end_y, start_x: end_x]
            
            roi = roi[mask]
            color = random.choice(colors)
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")
            original_image[start_y:end_y, start_x: end_x][mask] = blended
            
            color = [int(c) for c in color]
            cv2.rectangle(original_image, (start_x, start_y), (end_x, end_y), color, 2)
            text = "{}: {:.2f}".format(class_labels[class_id], confidence)
            cv2.putText(original_image, text, (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.imshow('Image', original_image)
            cv2.imwrite(os.path.join(config.output_image_dir, 'detect_image_{}_{}.jpg'.format(image_id, i)), original_image)
            cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    all_images = load_images()
    print('Number of input_images: ', len(all_images))
    class_labels = load_labels()
    print('\n')
    print('Number of labels: ', len(class_labels))
    colors = load_colors()
    print("Loading mask-r-CNN...")
    net = load_pretrained_model()
    
    for image_id, image in enumerate(all_images):
        boxes, masks = detect(image, net)
        segment(image, image_id, boxes, masks)
    
    
    