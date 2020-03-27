# Metrics evaluation for object detection

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

A simple tool to evaluate Pascal VOC mAP and COCO AP (standard) for object detection model, with dataset annotation and model inference result.

## Guide

1. Prepare dataset annotation file and class names file.

    Data annotation file format:
    * One row for one image in annotation file;
    * Row format: `image_file_path box1 box2 ... boxN`;
    * Box format: `x_min,y_min,x_max,y_max,class_id` (no space).
    * Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```
    1. For VOC style dataset, you can use [voc_annotation.py](https://github.com/david8862/Object-Detection-Evaluation/blob/master/tools/voc_annotation.py) to convert original dataset to txt annotation file:
       ```
       # cd tools && python voc_annotation.py -h
       usage: voc_annotation.py [-h] [--dataset_path DATASET_PATH]
                                [--output_path OUTPUT_PATH]
                                [--classes_path CLASSES_PATH] [--include_difficult]
                                [--include_no_obj]

       convert PascalVOC dataset annotation to txt annotation file

       optional arguments:
         -h, --help            show this help message and exit
         --dataset_path DATASET_PATH
                               path to PascalVOC dataset, default is ../VOCdevkit
         --output_path OUTPUT_PATH
                               output path for generated annotation txt files,
                               default is ./
         --classes_path CLASSES_PATH
                               path to class definitions
         --include_difficult   to include difficult object
         --include_no_obj      to include no object image
       ```
       By default, the VOC convert script will try to go through both VOC2007/VOC2012 dataset dir under the dataset_path and generate train/val/test annotation file separately


    2. For COCO style dataset, you can use [coco_annotation.py](https://github.com/david8862/Object-Detection-Evaluation/blob/master/tools/coco_annotation.py) to convert original dataset to txt annotation file:
       ```
       # cd tools && python coco_annotation.py -h
       usage: coco_annotation.py [-h] [--dataset_path DATASET_PATH]
                                 [--output_path OUTPUT_PATH]
                                 [--classes_path CLASSES_PATH] [--include_no_obj]
                                 [--customize_coco]

       convert COCO dataset annotation to txt annotation file

       optional arguments:
         -h, --help            show this help message and exit
         --dataset_path DATASET_PATH
                               path to MSCOCO dataset, default is ../mscoco2017
         --output_path OUTPUT_PATH
                               output path for generated annotation txt files,
                               default is ./
         --classes_path CLASSES_PATH
                               path to class definitions, default is
                               ../configs/coco_classes.txt
         --include_no_obj      to include no object image
         --customize_coco      It is a user customize coco dataset. Will not follow
                               standard coco class label
       ```
       This script will try to convert COCO instances_train2017 and instances_val2017 under dataset_path. You can change the code for your dataset

   For class names file format, refer to  [coco_classes.txt](https://github.com/david8862/Object-Detection-Evaluation/blob/master/configs/coco_classes.txt)

2. Generate detection result file.

    Result file format:
    * One row for one image in result file;
    * Row format: `image_file_path box1 box2 ... boxN`;
    * Box format: `x_min,y_min,x_max,y_max,class_id,score` (no space).
    * Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0,0.76 30,50,200,120,3,0.53
    path/to/img2.jpg 120,300,250,600,2,0.89
    ...
    ```

3. Run evaluation

```
# python object_detection_eval.py -h
usage: object_detection_eval.py [-h] --annotation_file ANNOTATION_FILE
                                --result_file RESULT_FILE --classes_path
                                CLASSES_PATH [--eval_type EVAL_TYPE]
                                [--iou_threshold IOU_THRESHOLD]

evaluate Object Detection model with test dataset

optional arguments:
  -h, --help            show this help message and exit
  --annotation_file ANNOTATION_FILE
                        dataset annotation txt file
  --result_file RESULT_FILE
                        detection result txt file
  --classes_path CLASSES_PATH
                        path to class definitions
  --eval_type EVAL_TYPE
                        evaluation type (VOC/COCO), default=VOC
  --iou_threshold IOU_THRESHOLD
                        IOU threshold for PascalVOC mAP, default=0.5
```
