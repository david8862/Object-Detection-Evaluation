#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
script to convert rio ground truth and detection result txt annotation files
to format in this for evaluation
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


def convert_rio_annotation(annotation_path, classes, output_path, ground_truth, include_difficult):

    # get real path for dataset
    annotation_realpath = os.path.realpath(annotation_path)

    # load annotation files list
    annotation_files = [f for f in os.listdir(annotation_realpath)]
    annotation_files.sort()

    # count class item number in each set
    class_count = OrderedDict([(item, 0) for item in classes])

    # create output file
    output_file = open(output_path, 'w')

    pbar = tqdm(total=len(annotation_files), desc='rio annotation convert')
    for annotation_file in annotation_files:
        pbar.update(1)

        # fake an image name and write as 1st part
        image_name = annotation_file.replace('.txt', '.jpg')
        output_file.write(image_name)

        annotation_lines = open(os.path.join(annotation_realpath, annotation_file), "r")
        for annotation_line in annotation_lines:
            annotation_line = annotation_line.strip()
            if annotation_line.replace(' ', '') == '':
                continue
            line = annotation_line.split(" ")

            # There will be some difference on line format between ground truth
            # annotation files and detection result files, so we need different
            # parse logic.
            #
            # line for a GT annotation:
            # <class_name> <difficult> <groupof> <xmin> <ymin> <xmax> <ymax>
            #
            # line for a detection result:
            # <class_name> <score> <xmin> <ymin> <xmax> <ymax>
            #

            # parse bbox info from each line, it's common part for
            # GT annotation and detection result annotation
            xmin = float(line[-4])
            ymin = float(line[-3])
            xmax = float(line[-2])
            ymax = float(line[-1])
            boxes = (xmin, ymin, xmax, ymax)

            if ground_truth:
                # Here we are dealing with GT annotation
                # <class_name> <difficult> <groupof> <xmin> <ymin> <xmax> <ymax>
                class_name = ' '.join(line[:-6])  # class name
                difficult = bool(int(line[-6]))
                groupof = bool(int(line[-5]))

                # check and bypass difficult box if needed
                if difficult and (not include_difficult):
                    continue

                output_file.write(" " + ",".join([str(box) for box in boxes]) + ',' + str(classes.index(class_name)))
                class_count[class_name] = class_count[class_name] + 1
            else:
                # Here we are dealing with detection result annotation
                # <class_name> <score> <xmin> <ymin> <xmax> <ymax>
                class_name = ' '.join(line[:-5])  # class name
                score = float(line[-5])

                output_file.write(" " + ",".join([str(box) for box in boxes]) + ',' + str(classes.index(class_name))+ ',' + str(score))
                class_count[class_name] = class_count[class_name] + 1

        output_file.write('\n')
        annotation_lines.close()
    pbar.close()
    output_file.close()

    # print out item number statistic
    print('\nDone for %s. classes number statistic'%(annotation_path))
    print('Image number: %d'%(len(annotation_files)))
    print('Object class number:')
    for (class_name, number) in class_count.items():
        print('%s: %d' % (class_name, number))
    print('total object number:', np.sum(list(class_count.values())))



def main():
    parser = argparse.ArgumentParser(description='convert rio on-device annotations txt files for evaluation')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ground_truth_path', type=str, default=None, help="path for rio ground truth annotation files")
    group.add_argument('--detection_result_path', type=str, default=None, help="path for rio detection result files")

    parser.add_argument('--classes_path',required=False, type=str, help='path to class definitions, default=%(default)s', default='../configs/voc_classes.txt')
    parser.add_argument('--output_path',required=True, type=str, help='path to output single txt file')
    parser.add_argument('--include_difficult', action="store_true", help='to include difficult object', default=False)

    args = parser.parse_args()

    # load classes
    classes = get_classes(args.classes_path)

    # specify annotation_path
    annotation_path = args.ground_truth_path if args.ground_truth_path else args.detection_result_path

    # a trick: using args.ground_truth_path as flag to check if we're converting a ground truth annotation
    convert_rio_annotation(annotation_path, classes, args.output_path, args.ground_truth_path, args.include_difficult)


if __name__ == "__main__":
    main()
