# Instance segmentation

## Setup Instructions
1. Download the pretrained model file and place it inside the **mask_rcnn_pretrained** folder.
- frozen_inference_graph.pb (https://github.com/datitran/object_detector_app/tree/master/object_detection/ssd_mobilenet_v1_coco_11_06_2017)

2. Place the images that you want to perform instance segmentation in the **input_images** folder.
3. Play with the parameters in **config.py**
4. To conduct instance segmentation on your images. cd to this repository in your terminal and run:
```bash
python segmentation.py
```
5. The processed images are saved in the **output_images** folder.


Acknowledgements:
1. The code was built on Adrian's blog (https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/).
2. Check out https://github.com/matterport/Mask_RCNN
3. mask_rcnn_inception_v2_coco_2018_01_28.pbtxt is downloaded from https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt
4. object_detection_classes_coco is downloaded from https://github.com/opencv/opencv/blob/master/samples/data/dnn/object_detection_classes_coco.txt
