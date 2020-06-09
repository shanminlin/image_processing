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
    label_path = os.path.join(config.yolo_dir, 'coco.names')
    with open(label_path) as file:
        class_labels = file.read().strip().split('\n')
    return class_labels

def assign_colors(class_labels):
    """Each label is assigned a unique integer between 0 and 255 inclusive"""
    colors = np.random.choice(256, size=(len(class_labels), 3))
    return colors

def load_pretrained_model():
    """Load yolo pretrained model"""
    config_path = os.path.join(config.yolo_dir, 'yolov3.cfg')
    weight_path = os.path.join(config.yolo_dir, 'yolov3.weights')

    # load yolo pretrained model
    net = cv2.dnn.readNetFromDarknet(config_path, weight_path)
    return net

def detect(image, net):
    """Detects objects in input image"""
    
    # Get all layer names
    layers = net.getLayerNames()
    # net.getUnconnectedOutLayers() returns output layer indices starting from 1
    # and in 2d array format
    # so to get the name of that layer, we have to do some processing
    output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    # Max scaling and resize input images
    processed_image = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(processed_image)
    outputs = net.forward(output_layers)
    
    class_ids = []
    boxes = []
    confidences = []
    width = image.shape[1]
    height = image.shape[0]
    for output in outputs:
        # loop over each of the detections
        for detection in output:
            class_scores = detection[5:]
            confidence = float(np.max(class_scores))
            
            if confidence > config.confidence_threshold:
                # Rescale the bounding box
                center_x = detection[0] * width
                center_y = detection[1] * height
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                left_x = int(center_x - w / 2)
                top_y = int(center_y - h / 2)
                
                boxes.append([left_x, top_y, w, h])
                confidences.append(confidence)
                class_id = np.argmax(class_scores)
                class_ids.append(class_id)
    
    return boxes, class_ids, confidences     

def draw_boxes(image, image_id, boxes, class_ids, confidences, colors):
    # Remove overlapping bounding boxes
    # the data type of the boxes is int; the type of the confidences is float
    selected_indices = cv2.dnn.NMSBoxes(boxes, confidences, config.confidence_threshold, config.suppression_threshold)
    
    # Draw selected boxes
    for i in selected_indices:
        i = i[0]
        class_id = class_ids[i]
        confidence = confidences[i]
        label = class_labels[class_id]
        
        # !! have to use python list, not numpy ndarray below
        # color = colors[class_id] gives error
        color = [int(c) for c in colors[class_id]]
        x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = '{}: {:.2f}'.format(label, confidence)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Image", image)
    cv2.imwrite(os.path.join(config.output_image_dir, 'image_detection_{}.jpg'.format(image_id)), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    all_images = load_images()
    print('Number of input_images: ', len(all_images))
    class_labels = load_labels()
    print('\n')
    print('Number of labels: ', len(class_labels))
    
    print("Loading YOLO...")
    net = load_pretrained_model()
    
    colors = assign_colors(class_labels)
    
    for image_id, image in enumerate(all_images):
        boxes, class_ids, confidences = detect(image, net)
        draw_boxes(image, image_id, boxes, class_ids, confidences, colors)