# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 18:26:14 2020

@author: SS
"""
import numpy as np
import cv2

path = 'input_images/blur/1.png'
image = cv2.imread(path)
output_image = cv2.fastNlMeansDenoisingColored(image, None, 30, 30, 7, 15)

# Show both original and processed images
horizontal_stack = np.hstack((image, output_image))
cv2.imwrite('output_images/denoised.jpg', horizontal_stack)
cv2.imshow('Image', horizontal_stack)
cv2.waitKey(0)
cv2.destroyAllWindows()