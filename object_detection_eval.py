#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate mAP for Detection model on some annotation dataset
"""
import os, argparse, time
import operator
import numpy as np
import cv2, colorsys
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def touchdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_colors(class_names):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.
    return colors

def get_dataset(annotation_file, shuffle=True):
    with open(annotation_file) as f:
        lines = f.readlines()

    if shuffle:
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)

    return lines

def draw_label(image, text, color, coords):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    padding = 5
    rect_height = text_height + padding * 2
    rect_width = text_width + padding * 2

    (x, y) = coords

    cv2.rectangle(image, (x, y), (x + rect_width, y - rect_height), color, cv2.FILLED)
    cv2.putText(image, text, (x + padding, y - text_height + padding), font,
                fontScale=font_scale,
                color=(255, 255, 255),
                lineType=cv2.LINE_AA)
    return image


def draw_boxes(image, boxes, classes, scores, class_names, colors, show_score=True):
    if classes is None or len(classes) == 0:
        return image

    for box, cls, score in zip(boxes, classes, scores):
        xmin, ymin, xmax, ymax = box

        class_name = class_names[cls]
        if show_score:
            label = '{} {:.2f}'.format(class_name, score)
        else:
            label = '{}'.format(class_name)
        #print(label, (xmin, ymin), (xmax, ymax))

        # if no color info, use black(0,0,0)
        if colors == None:
            color = (0,0,0)
        else:
            color = colors[cls]
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1, cv2.LINE_AA)
        image = draw_label(image, label, color, (xmin, ymin))

    return image


def annotation_parse(annotation_lines, class_names):
    '''
    parse annotation lines to get image dict and ground truth class dict

    image dict would be like:
    annotation_records = {
        '/path/to/000001.jpg': {'100,120,200,235':'dog', '85,63,156,128':'car', ...},
        ...
    }

    ground truth class dict would be like:
    classes_records = {
        'car': [
                ['000001.jpg','100,120,200,235'],
                ['000002.jpg','85,63,156,128'],
                ...
               ],
        ...
    }
    '''
    annotation_records = {}
    classes_records = {class_name: [] for class_name in class_names}

    for line in annotation_lines:
        box_records = {}
        image_name = line.split(' ')[0]
        boxes = line.split(' ')[1:]
        for box in boxes:
            #strip box coordinate and class
            class_name = class_names[int(box.split(',')[-1])]
            coordinate = ','.join(box.split(',')[:-1])
            box_records[coordinate] = class_name
            #append or add ground truth class item
            record = [os.path.basename(image_name), coordinate]
            if class_name in classes_records:
                classes_records[class_name].append(record)
            else:
                classes_records[class_name] = list([record])
        annotation_records[image_name] = box_records

    return annotation_records, classes_records


def result_parse(result_lines, class_names):
    '''
    parse detection result lines to get prediction class dict
    predict class dict would contain image_name, coordinary and score, and
    sorted by score:
    pred_classes_records = {
        'car': [
                ['000001.jpg','94,115,203,232',0.98],
                ['000002.jpg','82,64,154,128',0.93],
                ...
               ],
        ...
    }
    '''
    pred_classes_records = {class_name: [] for class_name in class_names}

    for line in result_lines:
        image_name = line.split(' ')[0]
        boxes = line.split(' ')[1:]
        for box in boxes:
            #strip box coordinate, class and score
            coordinate = ','.join(box.split(',')[:-2])
            class_name = class_names[int(box.split(',')[-2])]
            score = float(box.split(',')[-1])
            record = [os.path.basename(image_name), coordinate, score]

            if class_name in pred_classes_records:
                pred_classes_records[class_name].append(record)
            else:
                pred_classes_records[class_name] = list([record])

    # sort pred_classes_records for each class according to score
    for pred_class_list in pred_classes_records.values():
        pred_class_list.sort(key=lambda ele: ele[2], reverse=True)

    return pred_classes_records


def box_iou(pred_box, gt_box):
    '''
    Calculate iou for predict box and ground truth box
    Param
         pred_box: predict box coordinate
                   (xmin,ymin,xmax,ymax) format
         gt_box: ground truth box coordinate
                 (xmin,ymin,xmax,ymax) format
    Return
         iou value
    '''
    # get intersection box
    inter_box = [max(pred_box[0], gt_box[0]), max(pred_box[1], gt_box[1]), min(pred_box[2], gt_box[2]), min(pred_box[3], gt_box[3])]
    # compute overlap (IoU) = area of intersection / area of union
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    inter_area = (inter_box[2] - inter_box[0]) * (inter_box[3] - inter_box[1])
    union_area = pred_area + gt_area - inter_area
    return 0 if union_area == 0 else float(inter_area) / float(union_area)


def match_gt_box(pred_record, gt_records, iou_threshold=0.5):
    '''
    Search gt_records list and try to find a matching box for the predict box

    Param
         pred_record: with format ['image_file', 'xmin,ymin,xmax,ymax', score]
         gt_records: record list with format
                     [
                      ['image_file', 'xmin,ymin,xmax,ymax', 'usage'],
                      ['image_file', 'xmin,ymin,xmax,ymax', 'usage'],
                      ...
                     ]
         iou_threshold:

         pred_record and gt_records should be from same annotation image file

    Return
         matching gt_record index. -1 when there's no matching gt
    '''
    max_iou = 0.0
    max_index = -1
    #get predict box coordinate
    pred_box = [float(x) for x in pred_record[1].split(',')]

    for i, gt_record in enumerate(gt_records):
        #get ground truth box coordinate
        gt_box = [float(x) for x in gt_record[1].split(',')]
        iou = box_iou(pred_box, gt_box)

        # if the ground truth has been assigned to other
        # prediction, we couldn't reuse it
        if iou > max_iou and gt_record[2] == 'unused' and pred_record[0] == gt_record[0]:
            max_iou = iou
            max_index = i

    # drop the prediction if couldn't match iou threshold
    if max_iou < iou_threshold:
        max_index = -1

    return max_index


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
        mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
      (goes from the end to the beginning)
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #   range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #   range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
    """
    # matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
      (numerical integration)
    """
    # matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre

'''
def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap, mrec, mpre
'''


def get_rec_prec(true_positive, false_positive, gt_records):
    '''
    Calculate precision/recall based on true_positive, false_positive
    result.
    '''
    cumsum = 0
    for idx, val in enumerate(false_positive):
        false_positive[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(true_positive):
        true_positive[idx] += cumsum
        cumsum += val

    rec = true_positive[:]
    for idx, val in enumerate(true_positive):
        rec[idx] = (float(true_positive[idx]) / len(gt_records)) if len(gt_records) != 0 else 0

    prec = true_positive[:]
    for idx, val in enumerate(true_positive):
        prec[idx] = float(true_positive[idx]) / (false_positive[idx] + true_positive[idx])

    return rec, prec


def draw_rec_prec(rec, prec, mrec, mprec, class_name, ap):
    """
     Draw plot
    """
    plt.plot(rec, prec, '-o')
    # add a new penultimate point to the list (mrec[-2], 0.0)
    # since the last line segment (and respective area) do not affect the AP value
    area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
    area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
    plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
    # set window title
    fig = plt.gcf() # gcf - get current figure
    fig.canvas.set_window_title('AP ' + class_name)
    # set plot title
    plt.title('class: ' + class_name + ' AP = {}%'.format(ap*100))
    #plt.suptitle('This is a somewhat long figure title', fontsize=16)
    # set axis titles
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # optional - set axes
    axes = plt.gca() # gca - get current axes
    axes.set_xlim([0.0,1.0])
    axes.set_ylim([0.0,1.05]) # .05 to give some extra space
    # Alternative option -> wait for button to be pressed
    #while not plt.waitforbuttonpress(): pass # wait for key display
    # Alternative option -> normal display
    #plt.show()
    # save the plot
    rec_prec_plot_path = os.path.join('result','classes')
    touchdir(rec_prec_plot_path)
    fig.savefig(os.path.join(rec_prec_plot_path, class_name + ".jpg"))
    plt.cla() # clear axes for next plot


def adjust_axes(r, t, fig, axes):
    """
     Plot - adjust axes
    """
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    """
     Draw plot using Matplotlib
    """
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar != "":
        """
         Special case to draw in (green=true predictions) & (red=false predictions)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Predictions')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Predictions', left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            #   first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    else:
      plt.barh(range(n_classes), sorted_values, color=plot_color)
      """
       Write number on side of bar
      """
      fig = plt.gcf() # gcf - get current figure
      axes = plt.gca()
      r = fig.canvas.get_renderer()
      for i, val in enumerate(sorted_values):
          str_val = " " + str(val) # add a space before
          if val < 1.0:
              str_val = " {0:.2f}".format(val)
          t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
          # re-set axes to show number inside the figure
          if i == (len(sorted_values)-1): # largest bar
              adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15    # in percentage of the figure height
    bottom_margin = 0.05 # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    if to_show:
        plt.show()
    # close the plot
    plt.close()


def calc_AP(gt_records, pred_records, class_name, iou_threshold, show_result):
    '''
    Calculate AP value for one class records

    Param
         gt_records: ground truth records list for one class, with format:
                     [
                      ['image_file', 'xmin,ymin,xmax,ymax'],
                      ['image_file', 'xmin,ymin,xmax,ymax'],
                      ...
                     ]
         pred_record: predict records for one class, with format:
                     [
                      ['image_file', 'xmin,ymin,xmax,ymax', score],
                      ['image_file', 'xmin,ymin,xmax,ymax', score],
                      ...
                     ]
    Return
         AP value for the class
    '''
    # append usage flag in gt_records for matching gt search
    gt_records = [gt_record + ['unused'] for gt_record in gt_records]

    # init true_positive and false_positive list
    nd = len(pred_records)  # number of predict data
    true_positive = [0] * nd
    false_positive = [0] * nd
    true_positive_count = 0
    # assign predictions to ground truth objects
    for idx, pred_record in enumerate(pred_records):
        # filter out gt record from same image
        image_gt_records = [ gt_record for gt_record in gt_records if gt_record[0] == pred_record[0]]

        i = match_gt_box(pred_record, image_gt_records, iou_threshold=iou_threshold)
        if i != -1:
            # find a valid gt obj to assign, set
            # true_positive list and mark image_gt_records.
            #
            # trick: gt_records will also be marked
            # as 'used', since image_gt_records is a
            # reference list
            image_gt_records[i][2] = 'used'
            true_positive[idx] = 1
            true_positive_count += 1
        else:
            false_positive[idx] = 1

    # compute precision/recall
    rec, prec = get_rec_prec(true_positive, false_positive, gt_records)
    ap, mrec, mprec = voc_ap(rec, prec)
    if show_result:
        draw_rec_prec(rec, prec, mrec, mprec, class_name, ap)

    return ap, true_positive_count


def plot_Pascal_AP_result(count_images, count_true_positives, num_classes,
                          gt_counter_per_class, pred_counter_per_class,
                          precision_dict, recall_dict, mPrec, mRec,
                          APs, mAP, iou_threshold):
    '''
     Plot the total number of occurences of each class in the ground-truth
    '''
    window_title = "Ground-Truth Info"
    plot_title = "Ground-Truth\n" + "(" + str(count_images) + " files and " + str(num_classes) + " classes)"
    x_label = "Number of objects per class"
    output_path = os.path.join('result','Ground-Truth_Info.jpg')
    draw_plot_func(gt_counter_per_class, num_classes, window_title, plot_title, x_label, output_path, to_show=False, plot_color='forestgreen', true_p_bar='')

    '''
     Plot the total number of occurences of each class in the "predicted" folder
    '''
    window_title = "Predicted Objects Info"
    # Plot title
    plot_title = "Predicted Objects\n" + "(" + str(count_images) + " files and "
    count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(pred_counter_per_class.values()))
    plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
    # end Plot title
    x_label = "Number of objects per class"
    output_path = os.path.join('result','Predicted_Objects_Info.jpg')
    draw_plot_func(pred_counter_per_class, len(pred_counter_per_class), window_title, plot_title, x_label, output_path, to_show=False, plot_color='forestgreen', true_p_bar=count_true_positives)

    '''
     Draw mAP plot (Show AP's of all classes in decreasing order)
    '''
    window_title = "mAP"
    plot_title = "mAP@IoU={0}: {1:.2f}%".format(iou_threshold, mAP)
    x_label = "Average Precision"
    output_path = os.path.join('result','mAP.jpg')
    draw_plot_func(APs, num_classes, window_title, plot_title, x_label, output_path, to_show=False, plot_color='royalblue', true_p_bar='')

    '''
     Draw Precision plot (Show Precision of all classes in decreasing order)
    '''
    window_title = "Precision"
    plot_title = "mPrec@IoU={0}: {1:.2f}%".format(iou_threshold, mPrec)
    x_label = "Precision rate"
    output_path = os.path.join('result','Precision.jpg')
    draw_plot_func(precision_dict, len(precision_dict), window_title, plot_title, x_label, output_path, to_show=False, plot_color='royalblue', true_p_bar='')

    '''
     Draw Recall plot (Show Recall of all classes in decreasing order)
    '''
    window_title = "Recall"
    plot_title = "mRec@IoU={0}: {1:.2f}%".format(iou_threshold, mRec)
    x_label = "Recall rate"
    output_path = os.path.join('result','Recall.jpg')
    draw_plot_func(recall_dict, len(recall_dict), window_title, plot_title, x_label, output_path, to_show=False, plot_color='royalblue', true_p_bar='')


def get_mean_metric(metric_records, gt_classes_records):
    '''
    Calculate mean metric, but only count classes which have ground truth object

    Param
        metric_records: metric dict like:
            metric_records = {
                'aeroplane': 0.79,
                'bicycle': 0.79,
                    ...
                'tvmonitor': 0.71,
            }
        gt_classes_records: ground truth class dict like:
            gt_classes_records = {
                'car': [
                    ['000001.jpg','100,120,200,235'],
                    ['000002.jpg','85,63,156,128'],
                    ...
                    ],
                ...
            }
    Return
         mean_metric: float value of mean metric
    '''
    mean_metric = 0.0
    count = 0
    for (class_name, metric) in metric_records.items():
        if (class_name in gt_classes_records) and (len(gt_classes_records[class_name]) != 0):
            mean_metric += metric
            count += 1
    mean_metric = (mean_metric/count)*100 if count != 0 else 0.0
    return mean_metric


def compute_mAP_PascalVOC(annotation_records, gt_classes_records, pred_classes_records, class_names, iou_threshold, show_result=True):
    '''
    Compute PascalVOC style mAP
    '''
    APs = {}
    count_true_positives = {class_name: 0 for class_name in list(gt_classes_records.keys())}
    #get AP value for each of the ground truth classes
    for _, class_name in enumerate(class_names):
        #if there's no gt obj for a class, record 0
        if class_name not in gt_classes_records:
            APs[class_name] = 0.
            continue
        gt_records = gt_classes_records[class_name]
        #if we didn't detect any obj for a class, record 0
        if class_name not in pred_classes_records:
            APs[class_name] = 0.
            continue
        pred_records = pred_classes_records[class_name]
        ap, true_positive_count = calc_AP(gt_records, pred_records, class_name, iou_threshold, show_result)
        APs[class_name] = ap
        count_true_positives[class_name] = true_positive_count

    #get mAP percentage value
    #mAP = np.mean(list(APs.values()))*100
    mAP = get_mean_metric(APs, gt_classes_records)


    #get GroundTruth count per class
    gt_counter_per_class = {}
    for (class_name, info_list) in gt_classes_records.items():
        gt_counter_per_class[class_name] = len(info_list)

    #get Precision count per class
    pred_counter_per_class = {class_name: 0 for class_name in list(gt_classes_records.keys())}
    for (class_name, info_list) in pred_classes_records.items():
        pred_counter_per_class[class_name] = len(info_list)


    #get the precision & recall
    precision_dict = {}
    recall_dict = {}
    for (class_name, gt_count) in gt_counter_per_class.items():
        if (class_name not in pred_counter_per_class) or (class_name not in count_true_positives) or pred_counter_per_class[class_name] == 0:
            precision_dict[class_name] = 0.
        else:
            precision_dict[class_name] = float(count_true_positives[class_name]) / pred_counter_per_class[class_name]

        if class_name not in count_true_positives or gt_count == 0:
            recall_dict[class_name] = 0.
        else:
            recall_dict[class_name] = float(count_true_positives[class_name]) / gt_count

    #get mPrec, mRec
    #mPrec = np.mean(list(precision_dict.values()))*100
    #mRec = np.mean(list(recall_dict.values()))*100
    mPrec = get_mean_metric(precision_dict, gt_classes_records)
    mRec = get_mean_metric(recall_dict, gt_classes_records)


    if show_result:
        plot_Pascal_AP_result(len(annotation_records), count_true_positives, len(gt_classes_records),
                                  gt_counter_per_class, pred_counter_per_class,
                                  precision_dict, recall_dict, mPrec, mRec,
                                  APs, mAP, iou_threshold)
        #show result
        print('\nPascal VOC AP evaluation')
        for (class_name, AP) in APs.items():
            print('%s: AP %.4f, precision %.4f, recall %.4f' % (class_name, AP, precision_dict[class_name], recall_dict[class_name]))
        print('mAP@IoU=%.2f result: %f' % (iou_threshold, mAP))
        print('mPrec@IoU=%.2f result: %f' % (iou_threshold, mPrec))
        print('mRec@IoU=%.2f result: %f' % (iou_threshold, mRec))

    #return mAP percentage value
    return mAP


def compute_AP_COCO(annotation_records, gt_classes_records, pred_classes_records, class_names, show_result=True):
    '''
    Compute MSCOCO AP list on AP 0.5:0.05:0.95
    '''
    iou_threshold_list = np.arange(0.50,0.95,0.05)
    APs = {}
    for iou_threshold in iou_threshold_list:
        iou_threshold = round(iou_threshold, 2)
        mAP = compute_mAP_PascalVOC(annotation_records, gt_classes_records, pred_classes_records, class_names, iou_threshold, show_result=False)
        APs[iou_threshold] = round(mAP, 6)

    #get overall AP percentage value
    AP = np.mean(list(APs.values()))

    if show_result:
        '''
         Draw MS COCO AP plot
        '''
        touchdir('result')
        window_title = "MSCOCO AP on different IOU"
        plot_title = "COCO AP = {0:.2f}%".format(AP)
        x_label = "Average Precision"
        output_path = os.path.join('result','COCO_AP.jpg')
        draw_plot_func(APs, len(APs), window_title, plot_title, x_label, output_path, to_show=False, plot_color='royalblue', true_p_bar='')

        print('\nMS COCO AP evaluation')
        for (iou_threshold, AP_value) in APs.items():
            print('IOU %.2f: AP %f' % (iou_threshold, AP_value))
        print('total AP: %f' % (AP))

    #return AP percentage value
    return AP


def compute_AP_COCO_Scale(annotation_records, scale_gt_classes_records, pred_classes_records, class_names):
    '''
    Compute MSCOCO AP on different scale object: small, medium, large
    '''
    scale_APs = {}
    for scale_key in ['small','medium','large']:
        gt_classes_records = scale_gt_classes_records[scale_key]
        scale_AP = compute_AP_COCO(annotation_records, gt_classes_records, pred_classes_records, class_names, show_result=False)
        scale_APs[scale_key] = round(scale_AP, 4)

    #get overall AP percentage value
    scale_mAP = np.mean(list(scale_APs.values()))

    '''
     Draw Scale AP plot
    '''
    touchdir('result')
    window_title = "MSCOCO AP on different scale"
    plot_title = "scale mAP = {0:.2f}%".format(scale_mAP)
    x_label = "Average Precision"
    output_path = os.path.join('result','COCO_scale_AP.jpg')
    draw_plot_func(scale_APs, len(scale_APs), window_title, plot_title, x_label, output_path, to_show=False, plot_color='royalblue', true_p_bar='')

    '''
     Draw Scale Object Sum plot
    '''
    for scale_key in ['small','medium','large']:
        gt_classes_records = scale_gt_classes_records[scale_key]
        gt_classes_sum = {}

        for _, class_name in enumerate(class_names):
            # summarize the gt object number for every class on different scale
            gt_classes_sum[class_name] = np.sum(len(gt_classes_records[class_name])) if class_name in gt_classes_records else 0

        total_sum = np.sum(list(gt_classes_sum.values()))

        window_title = "{} object number".format(scale_key)
        plot_title = "total {} object number = {}".format(scale_key, total_sum)
        x_label = "Object Number"
        output_path = os.path.join('result','{}_object_number.jpg'.format(scale_key))
        draw_plot_func(gt_classes_sum, len(gt_classes_sum), window_title, plot_title, x_label, output_path, to_show=False, plot_color='royalblue', true_p_bar='')

    print('\nMS COCO AP evaluation on different scale')
    for (scale, AP_value) in scale_APs.items():
        print('%s scale: AP %f' % (scale, AP_value))
    print('total AP: %f' % (scale_mAP))


def add_gt_record(gt_records, gt_record, class_name):
    # append or add ground truth class item
    if class_name in gt_records:
        gt_records[class_name].append(gt_record)
    else:
        gt_records[class_name] = list([gt_record])

    return gt_records


def get_scale_gt_dict(gt_classes_records, class_names):
    '''
    Get ground truth class dict on different object scales, according to MS COCO metrics definition:
        small objects: area < 32^2
        medium objects: 32^2 < area < 96^2
        large objects: area > 96^2

    input gt_classes_records would be like:
    gt_classes_records = {
        'car': [
                ['000001.jpg','100,120,200,235'],
                ['000002.jpg','85,63,156,128'],
                ...
               ],
        ...
    }
    return a record dict with following format, for AP/AR eval on different scale:
        scale_gt_classes_records = {
            'small': {
                'car': [
                        ['000001.jpg','100,120,200,235'],
                        ['000002.jpg','85,63,156,128'],
                        ...
                       ],
                ...
            },

            'medium': {
                'car': [
                        ['000003.jpg','100,120,200,235'],
                        ['000004.jpg','85,63,156,128'],
                        ...
                       ],
                ...
            },

            'large': {
                'car': [
                        ['000005.jpg','100,120,200,235'],
                        ['000006.jpg','85,63,156,128'],
                        ...
                       ],
                ...
            }
        }
    '''
    scale_gt_classes_records = {}
    small_gt_records = {}
    medium_gt_records = {}
    large_gt_records = {}

    for _, class_name in enumerate(class_names):
        gt_records = gt_classes_records[class_name]

        for (image_file, box) in gt_records:
            # get box area based on coordinate
            box_coord = [int(p) for p in box.split(',')]
            box_area = (box_coord[2] - box_coord[0]) * (box_coord[3] - box_coord[1])

            # add to corresponding gt records dict according to area size
            if box_area <= 32*32:
                small_gt_records = add_gt_record(small_gt_records, [image_file, box], class_name)
            elif box_area > 32*32 and box_area <= 96*96:
                medium_gt_records = add_gt_record(medium_gt_records, [image_file, box], class_name)
            elif box_area > 96*96:
                large_gt_records = add_gt_record(large_gt_records, [image_file, box], class_name)

    # form up scale_gt_classes_records
    scale_gt_classes_records['small'] = small_gt_records
    scale_gt_classes_records['medium'] = medium_gt_records
    scale_gt_classes_records['large'] = large_gt_records

    return scale_gt_classes_records


def eval_AP(annotation_lines, result_lines, class_names, eval_type, iou_threshold):
    '''
    Compute AP for detection model on annotation dataset
    '''
    annotation_records, gt_classes_records = annotation_parse(annotation_lines, class_names)
    pred_classes_records = result_parse(result_lines, class_names)
    AP = 0.0

    if eval_type == 'VOC':
        AP = compute_mAP_PascalVOC(annotation_records, gt_classes_records, pred_classes_records, class_names, iou_threshold)
    elif eval_type == 'COCO':
        AP = compute_AP_COCO(annotation_records, gt_classes_records, pred_classes_records, class_names)
        # get AP for different scale: small, medium, large
        scale_gt_classes_records = get_scale_gt_dict(gt_classes_records, class_names)
        compute_AP_COCO_Scale(annotation_records, scale_gt_classes_records, pred_classes_records, class_names)
    else:
        raise ValueError('Unsupported evaluation type')

    return AP


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='evaluate Object Detection model with test dataset')
    '''
    Command line options
    '''
    parser.add_argument(
        '--annotation_file', type=str, required=True,
        help='dataset annotation txt file')

    parser.add_argument(
        '--result_file', type=str, required=True,
        help='detection result txt file')

    parser.add_argument(
        '--classes_path', type=str, required=True,
        help='path to class definitions')

    parser.add_argument(
        '--eval_type', type=str,
        help='evaluation type (VOC/COCO), default=VOC', default='VOC')

    parser.add_argument(
        '--iou_threshold', type=float,
        help='IOU threshold for PascalVOC mAP, default=0.5', default=0.5)

    args = parser.parse_args()
    # param parse
    class_names = get_classes(args.classes_path)

    annotation_lines = get_dataset(args.annotation_file, shuffle=False)
    result_lines = get_dataset(args.result_file, shuffle=False)

    start = time.time()
    eval_AP(annotation_lines, result_lines, class_names, args.eval_type, args.iou_threshold)
    end = time.time()
    print("Evaluation time cost: {:.6f}s".format(end - start))


if __name__ == '__main__':
    main()
