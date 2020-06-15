# Object detection
Use Yolo3 Darknet pretrained model to perform object detection.

## Setup Instructions
1. Download the pretrained model file and place it inside the **yolo_pretrained** folder.
- yolov3.weights (https://pjreddie.com/media/files/yolov3.weights)

2. Place the images that you want to perform object detection in the **input_images** folder.
3. Play with the parameters in **config.py**
4. To conduct object detection on your images. cd to this repository in your terminal and run:
```bash
python object_detection.py
```
5. The processed images are saved in the **output_images** folder.


Acknowledgement:
1. Adrian' PyImageSearch (https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/) helps me a lot in designing this project.
2. Opencv official tutorial (https://github.com/opencv/opencv/blob/master/samples/dnn/object_detection.py)
3. coco.names is downloaded from https://github.com/pjreddie/darknet/tree/master/data
4. yolo3.cfg is downloaded from https://github.com/pjreddie/darknet/tree/master/cfg
