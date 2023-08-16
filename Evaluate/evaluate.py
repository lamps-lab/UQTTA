import argparse
import xml.etree.ElementTree as ET
import re
import os

# import cv2
cell_count = 0

import os
import shutil
import copy
import json
import pandas as pd

def compare_dir(dir_1, dir_2):
    for file_2 in os.listdir(dir_2 + "_bak"):
        shutil.move(os.path.join(dir_2 + "_bak", file_2), dir_2)

    dir_1_list = os.listdir(dir_1)
    dir_2_list = os.listdir(dir_2)
    for file_1 in dir_1_list:
        if file_1 not in dir_2_list:
            shutil.move(os.path.join(dir_1, file_1), dir_1 + "_bak")
    for file_2 in dir_2_list:
        if file_2 not in dir_1_list:
            shutil.move(os.path.join(dir_2, file_2), dir_2 + "_bak")


def read_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f_read:
        coco_json = json.load(f_read)
    return coco_json


def split_label_data(origin_coco, ground_truth_split):
    coco_json = read_json(origin_coco)
    new_image_id = 0
    new_coco_images = []
    new_coco_annotations = []
    new_cell_id = 0
    for file in coco_json['images']:
        file_name = file['file_name']
        file_id = file['id']
        cells_ann = []
        tables_ann = []
        for ann in coco_json['annotations']:
            ann_copy = copy.deepcopy(ann)
            image_id = ann_copy['image_id']
            bbox_original = ann_copy['bbox']
            x_max = bbox_original[0] + bbox_original[2]
            bbox_original[2] = x_max
            y_max = bbox_original[1] + bbox_original[3]
            bbox_original[3] = y_max
            ann_copy['bbox'] = bbox_original
            if image_id == file_id:
                category_id = ann_copy['category_id']
                if category_id == 1:
                    cells_ann.append(ann_copy)
                else:
                    tables_ann.append(ann_copy)
        with open(os.path.join(ground_truth_split, file_name.replace("jpg", "json")), 'w', encoding='utf8') as f_write:
            json.dump(cells_ann, f_write)





"""This function gets the root of the XML file"""

#
# def getRoots(index):
#     parser = get_args()
#     args = parser.parse_args()
#     gtDirectory = args.gtDirectory
#     predDirectory = args.predDirectory
#
#     gtFiles = os.listdir(gtDirectory)
#     predFiles = os.listdir(predDirectory)
#
#     # get an xml file from the ground truth file
#     predFilePath = predFiles[index]
#     # print(predFilePath)
#     # check if ground truth file is found in prediction files
#
#     if predFilePath in gtFiles:
#         try:
#             gtTree = ET.parse(os.path.join(gtDirectory, predFilePath))
#             predTree = ET.parse(os.path.join(predDirectory, predFilePath))
#
#             # get the roots of the trees
#             gtRoot = gtTree.getroot()
#             predRoot = predTree.getroot()
#
#             return gtRoot, predRoot
#         except Exception as error:
#             print(error)
#             print(f"Prediction file: {predFilePath}")
#

"""This function extract the bounding boxes of the ground truth and prediction files"""

#
# def getBoundingBox(index):
#     try:
#
#         gtRoot, predRoot = getRoots(index)
#
#         gtBox, prBox = [], []  # TODO: remove
#         # getting coordinates of bounding boxes
#         for gt in gtRoot[0]:
#
#             # This loop through the root children and extract the contents of the cell tag
#             if gt.tag == 'cell':
#
#                 for pr in predRoot[check_bbox0]:
#                     if pr.tag == 'cell':
#                         cellAttrB = pr.attrib  # a dictionary with keys like start_row, end_row, etc
#                         start_rowB = cellAttrB['start-row']
#                         end_rowB = cellAttrB['end-row']
#                         start_colB = cellAttrB['start-col']
#                         end_colB = cellAttrB['end-col']
#
#                         cellAttrA = gt.attrib  # a dictionary with keys like start_row, end_row, etc
#                         start_rowA = cellAttrA['start-row']
#                         end_rowA = cellAttrA['end-row']
#                         start_colA = cellAttrA['start-col']
#                         end_colA = cellAttrA['end-col']
#
#                         # This check if the start_row, start_col, end_row, and end_col of both ground truth and predicted are aligned
#                         if (start_rowB == start_rowA) and (end_rowB == end_rowA) and (start_colB == start_colA) and (
#                                 end_colB == end_colA):
#                             cellValuesA, cellValuesB = gt[0].attrib['points'], pr[0].attrib['points']
#                             valA, valB = re.findall('\d*\d+', cellValuesA), re.findall('\d*\d+', cellValuesB)
#
#                             xminA, xminB = int(valA[0]), int(valB[0])
#                             yminA, yminB = int(valA[1]), int(valB[1])
#                             xmaxA, xmaxB = int(valA[0]) + int(valA[4]), int(valB[0]) + int(valB[4])
#                             ymaxA, ymaxB = int(valA[1]) + int(valA[3]), int(valB[1]) + int(valB[3])
#                             # xmaxA, xmaxB = int(valA[4]), int(valB[4])
#                             # ymaxA, ymaxB  = int(valA[3]), int(valB[3])
#                             bbox_gt = [xminA, yminA, xmaxA, ymaxA]
#                             bbox_pr = [xminB, yminB, xmaxB, ymaxB]
#                             gtBox.append(bbox_gt)
#                             prBox.append(bbox_pr)
#
#         return gtBox, prBox
#     except Exception as error:
#         print(error)


def getBoundingBox_v2(predFiles, index, args):
    file = predFiles[index]
    one_file_pre_res = read_json(os.path.join(args.predDirectory, file))
    one_file_pre_res = [item['bbox'] for item in one_file_pre_res]
    one_file_gt_res = read_json(os.path.join(args.gtDirectory, file))
    one_file_gt_res = [item['bbox'] for item in one_file_gt_res]
    return one_file_gt_res, one_file_pre_res


def check_bbox(index, threshold, predFiles=[], args=None):
    # fp = 0; fn = 0; tp = 0
    global cell_count
    precision, recall = 0, 0
    gtBox, prBox = getBoundingBox_v2(predFiles, index, args)
    same = 0
    precision = len(prBox)
    recall = len(gtBox)
    for gt in gtBox:
        for pr in prBox:
            iou, precision_iou, recall_iou = calc_iou(gt, pr)
            if iou > threshold:
                same += 1
                break
            # if iou > threshold:
            #     tp += 1
            # if precision_iou > threshold and recall_iou < threshold:
            #     precision.append(precision_iou)
            # elif recall_iou > threshold and precision_iou < threshold:
            #     recall.append(recall_iou)
            # elif precision_iou > threshold and recall_iou > threshold:
            #     precision.append(precision_iou)
            #     recall.append(recall_iou)
        # cell_count += 1
        # else:
        #     fp += 1

    return precision, recall, same  # tp, fp


"""This function takes bounding boxes of ground truth and predictions and calculates the IOU"""


def calc_iou(ground_truth, detection):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(ground_truth[0], detection[0])
    yA = max(ground_truth[1], detection[1])
    xB = min(ground_truth[2], detection[2])
    yB = min(ground_truth[3], detection[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    gtArea = (ground_truth[2] - ground_truth[0] + 1) * (ground_truth[3] - ground_truth[1] + 1)
    detectionArea = (detection[2] - detection[0] + 1) * (detection[3] - detection[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    # Calculate iou , precision and recall and append precision and recall to global lists
    iou = interArea / float(gtArea + detectionArea - interArea)
    precision_iou = interArea / float(detectionArea)
    recall_iou = interArea / float(gtArea)
    # precision.append(precision_iou)
    # recall.append(recall_iou)

    # return the intersection over union value
    return iou, precision_iou, recall_iou

def get_args():
    pre_default = '/data/cs_lzhan011/project/Graph-based-TSR/codes/test/test/str'
    # pre_default = '/data/cs_lzhan011/project/CascadeTabNet/Data/infer_res'
    ground_truth = '/data/cs_lzhan011/project/Graph-based-TSR/codes/test/test/test_ground_truth'
    ground_truth = '/data/cs_lzhan011/project/Graph-based-TSR/data/ctdar19_B2_m/test/SCAN/xml'

    # CascadeTabNet
    pre_default = '/data/cs_lzhan011/project/mmdetection/data/Cell/instances_cell_annotations2014_new.json'

    ground_truth = '/data/cs_lzhan011/project/mmdetection/data/Cell/instances_cell_annotations2014.json'
    ground_truth_split = '/data/cs_lzhan011/project/mmdetection/data/Cell/coco_split_table_input'
    # ground_truth_split =  '/data/cs_lzhan011/project/mmdetection/data/Cell/coco_split_document'
    # split_label_data(ground_truth, ground_truth_split)

    # pre_default = ground_truth_split
    # pre_default = '/data/cs_lzhan011/project/mmdetection/data/Cell/predict_res'

    # ground_truth_coco = '/data/cs_lzhan011/project/mmdetection/data/Cell/instances_cell_annotations2014_new.json'
    # ground_truth_split = '/data/cs_lzhan011/project/mmdetection/data/Cell/coco_split_table_input'
    # pre_default = '/data/cs_lzhan011/project/mmdetection/data/Cell/predict_res_input_table'
    #
    # ground_truth = '/data/cs_lzhan011/project/mmdetection/data/Cell/instances_cell_annotations2014_new_filter.json'
    # ground_truth_split = '/data/cs_lzhan011/project/mmdetection/data/Cell/coco_split_document'
    # # split_label_data(ground_truth, ground_truth_split)
    # pre_default = '/data/cs_lzhan011/project/mmdetection/data/Cell/predict_res_input_document_output_only_cell'

    pre_default = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/output_image_only_cell_test_predict'
    ground_truth_split = '/data/cs_lzhan011/project/mmdetection/data/Cell/coco_split_table_input'

    pre_default = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/output_image_document_input_3cate_test_predict'
    ground_truth_split = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/coco_split_document'

    pre_default = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/output_image_only_cell_test_white_3t_p20_predict'
    ground_truth_split = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/output_image_only_cell_test_white_3t_p20_gt'

    # pre_default = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/output_image_only_cell_test_white_2t_p20_predict'
    # ground_truth_split = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/output_image_only_cell_test_white_2t_p20_gt'

    # pre_default = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/output_image_only_cell_test_white_all_p20_predict'
    # ground_truth_split = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/output_image_only_cell_test_white_all_p20_gt'
    # ground_truth_split = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/coco_split_table_input'

    ground_truth = ground_truth_split
    compare_dir(ground_truth, pre_default)
    parser = argparse.ArgumentParser()
    parser.add_argument("--predDirectory", help="path to the predicted file", default=pre_default)
    parser.add_argument("--gtDirectory", help="path to the groundtruth path", default=ground_truth)

    return parser

if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()

    predDirectory = args.predDirectory
    predFiles = os.listdir(predDirectory)

    total_precision = 0.0
    total_recall = 0.0
    total_same = 0.0
    evaluate_res = []
    for thresh in range(5, 10):
        threshold = thresh / 10
        # tp = 0; fp = 0;

        for index in range(len(predFiles)):
            # print("index:",index)
            precision, recall, same = check_bbox(index, threshold, predFiles=predFiles, args=args)
            # print(precision, recall , same)
            total_precision += precision
            total_recall += recall
            total_same += same
            # print(total_precision, total_recall, total_same)
            # print()
        avg_precision = total_same / total_precision
        avg_recall = total_same / total_recall
        avg_f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)

        evaluate_res.append(
            {"Precision": avg_precision, "avg_recall": avg_recall, "F1": avg_f1, "threshold": threshold})
        print(
            f"Precision: {round(avg_precision, 3)}; Recall: {round(avg_recall, 3)}; F1: {round(avg_f1, 3)}; threshold: {threshold}")
        print("-" * 50)
        #     tp_i, fp_i= check_bbox(index, threshold)
        #     tp += tp_i
        #     fp += fp_i
        #     # fn += fn_i
        # precision = tp / (tp + fp)

        # print(f"precision: {round(precision, 3)}; threshold: {threshold}")
    evaluate_res = pd.DataFrame(evaluate_res)
    evaluate_res.to_excel(os.path.join(
        '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/output_image_only_cell_test_img_evaluate',
        'original.xlsx'))


