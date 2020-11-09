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
import numpy as np
from tqdm import tqdm
from collections import OrderedDict


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def convert_annotation(annotation_file, classes, output_path, ground_truth):
    # create output path
    os.makedirs(output_path, exist_ok=True)

    # count class item number in each set
    class_count = OrderedDict([(item, 0) for item in classes])

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
                # TODO: handle difficult
                x_min, y_min, x_max, y_max, class_id = list(map(int, bbox.split(',')))
                class_name = classes[int(class_id)].strip()
                out_box = '{} {} {} {} {}'.format(
                    class_name, x_min, y_min, x_max, y_max)
                class_count[class_name] = class_count[class_name] + 1
            else:
                # Here we are dealing with detection-results annotations
                # <class_name> <confidence> <left> <top> <right> <bottom>
                x_min, y_min, x_max, y_max, class_id, score = list(map(float, bbox.split(',')))
                class_name = classes[int(class_id)].strip()
                out_box = '{} {} {} {} {} {}'.format(
                    class_name, score, int(x_min), int(y_min), int(x_max), int(y_max))
                class_count[class_name] = class_count[class_name] + 1
            output_file.write(out_box + "\n")
    pbar.close()

    # print out item number statistic
    print('\nDone for %s. classes number statistic'%(annotation_file))
    print('Image number: %d'%(len(annotation_lines)))
    print('Object class number:')
    for (class_name, number) in class_count.items():
        print('%s: %d' % (class_name, number))
    print('total object number:', np.sum(list(class_count.values())))


def main():
    parser = argparse.ArgumentParser(description='convert annotations to third-party format')
    parser.add_argument('--output_path',required=False, type=str, help='Output root path for the converted annotations, default=%(default)s', default=os.path.join(os.path.dirname(__file__), 'output'))
    parser.add_argument('--classes_path',required=False, type=str, help='path to class definitions, default=%(default)s', default='../configs/voc_classes.txt')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ground_truth_file', type=str, default=None, help="converted ground truth annotation file")
    group.add_argument('--detection_result_file', type=str, default=None, help="converted detection result file")

    args = parser.parse_args()

    # load classes
    classes = get_classes(args.classes_path)

    # specify annotation_file and output_path
    annotation_file = args.ground_truth_file if args.ground_truth_file else args.detection_result_file
    output_path = os.path.join(args.output_path, 'ground_truth') if args.ground_truth_file else os.path.join(args.output_path, 'detection_result')
    # a trick: using args.ground_truth_file as flag to check if we're converting a ground truth annotation
    convert_annotation(annotation_file, classes, output_path, args.ground_truth_file)


if __name__ == "__main__":
    main()
