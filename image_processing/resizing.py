# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 18:39:38 2020

@author: SS
"""
import cv2
 
path = 'input_images/resize/1.jpg'
image = cv2.imread(path)
print('Original shape: {}'.format(image.shape))
 
scale_ratio = 3 
width = int(image.shape[1] * scale_ratio)
height = int(image.shape[0] * scale_ratio)
new_dimension = (width, height)
resized = cv2.resize(image, new_dimension, interpolation=cv2.INTER_AREA)

print('Resized shape: {}'.format(resized.shape))
 
cv2.imshow('Resized image', resized)
cv2.imwrite('output_images/resized.jpg', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()