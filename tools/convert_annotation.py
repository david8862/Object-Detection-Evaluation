#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
script to convert our ground truth and detection result txt annotation file
to following annotation directories:

<root path>/
├── ground_truth
│   ├── 000001.txt
│   ├── 000002.txt
│   ├── 000003.txt
│   └── ...
└── detection_result
    ├── 000001.txt
    ├── 000002.txt
    ├── 000003.txt
    └── ...

This kind of annotation could be used by following popular PascalVOC mAP
evaluation tools:
https://github.com/Cartucho/mAP
https://github.com/rafaelpadilla/Object-Detection-Metrics
'''
import os, argparse
from tqdm import tqdm

def convert_annotation(annotation_file, classes_path, output_path, ground_truth):
    # load classes
    class_file = open(classes_path, 'r')
    classes = class_file.readlines()

    # create output path
    os.makedirs(output_path, exist_ok=True)

    # load annotation and write to separate files
    with open(annotation_file, 'r') as f:
        annotation_lines = f.readlines()
    pbar = tqdm(total=len(annotation_lines), desc='annotation convert')
    for annotation in annotation_lines:
        pbar.update(1)

        # parse image_id from annotation to create output txt file
        annotation = annotation.split(' ')
        image_id = os.path.basename(annotation[0].strip()).split('.')[0]
        output_file_path = os.path.join(output_path, image_id+'.txt')
        output_file = open(output_file_path, 'w')

        for bbox in annotation[1:]:
            if ground_truth:
                # Here we are dealing with ground-truth annotations
                # <class_name> <left> <top> <right> <bottom> [<difficult>]
                # todo: handle difficulty
                x_min, y_min, x_max, y_max, class_id = list(map(int, bbox.split(',')))
                out_box = '{} {} {} {} {}'.format(
                    classes[int(class_id)].strip(), x_min, y_min, x_max, y_max)
            else:
                # Here we are dealing with detection-results annotations
                # <class_name> <confidence> <left> <top> <right> <bottom>
                x_min, y_min, x_max, y_max, class_id, score = list(map(float, bbox.split(',')))
                out_box = '{} {} {} {} {} {}'.format(
                    classes[int(class_id)].strip(), score, int(x_min), int(y_min), int(x_max), int(y_max))
            output_file.write(out_box + "\n")
    pbar.close()


def main():
    parser = argparse.ArgumentParser(description='convert annotations to third-party format')
    parser.add_argument('--output_path',required=False, type=str, help='Output root path for the converted annotations, default=./output', default=os.path.join(os.path.dirname(__file__), 'output'))
    parser.add_argument('--classes_path',required=False, type=str, help='path to class definitions, default ../configs/voc_classes.txt', default='../configs/voc_classes.txt')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ground_truth_file', type=str, default=None, help="converted ground truth annotation file")
    group.add_argument('--detection_result_file', type=str, default=None, help="converted detection result file")

    args = parser.parse_args()

    # specify annotation_file and output_path
    annotation_file = args.ground_truth_file if args.ground_truth_file else args.detection_result_file
    output_path = os.path.join(args.output_path, 'ground_truth') if args.ground_truth_file else os.path.join(args.output_path, 'detection_result')
    # a trick: using args.ground_truth_file as flag to check if we're converting a ground truth annotation
    convert_annotation(annotation_file, args.classes_path, output_path, args.ground_truth_file)


if __name__ == "__main__":
    main()
