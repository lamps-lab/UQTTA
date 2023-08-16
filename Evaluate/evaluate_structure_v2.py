import argparse
import xml.etree.ElementTree as ET
import re
import os

import pandas as pd

# import cv2
cell_count = 0

import os
import shutil
import copy
import json
import cv2


def write_json(json_path, json_object):
    with open(json_path, 'w', encoding='utf-8') as f_write:
        json.dump(json_object, f_write)


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


def get_args(ground_truth, pre_default):
    # pre_default = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/output_image_only_cell_test_predict'
    # ground_truth_split = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/output_image_only_cell_test_white_all_p20_gt'
    # ground_truth_split = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/coco_split_table_input'

    # ground_truth = ground_truth_split
    compare_dir(ground_truth, pre_default)
    parser = argparse.ArgumentParser()
    parser.add_argument("--predDirectory", help="path to the predicted file", default=pre_default)
    parser.add_argument("--gtDirectory", help="path to the groundtruth path", default=ground_truth)

    return parser


def getBoundingBox_v2(predFiles, index, args):
    file = predFiles[index]
    one_file_pre_res = read_json(os.path.join(args.predDirectory, file))
    one_file_pre_res = [item['bbox'] for item in one_file_pre_res]
    one_file_gt_res = read_json(os.path.join(args.gtDirectory, file))
    one_file_gt_res = [item['bbox'] for item in one_file_gt_res]

    # print("detection:", one_file_pre_res)
    if len(one_file_pre_res) > 0 and len(one_file_pre_res[0]) == 5:
        one_file_pre_res = [item for item in one_file_pre_res if item[-1] >= 0]

    return one_file_gt_res, one_file_pre_res


def get_gtBox_error(gt, gtBox_correct):
    gt_copy = copy.deepcopy(gt)
    for i in range(len(gt_copy) - 1, -1, -1):
        cell = gt_copy[i]
        for y in range(len(gtBox_correct)):
            if cell == gtBox_correct[y]:
                gt_copy.pop(i)
                break
    return gt_copy


def check_bbox(index, threshold, predFiles=[], args=None):
    # fp = 0; fn = 0; tp = 0
    global cell_count
    precision, recall = 0, 0
    gtBox, prBox = getBoundingBox_v2(predFiles, index, args)

    precision = len(prBox)
    recall = len(gtBox)

    prBox_error = []
    prBox_correct = []
    gtBox_correct = []
    matched_Box = []
    same = 0
    # print("gtBox:", gtBox)
    for pr in prBox:
        # print("****************************")
        one_cell_same = 0
        all_iou = []
        for gt in gtBox:
            iou, precision_iou, recall_iou = calc_iou(gt, pr)

            if iou > threshold:
                same += 1
                one_cell_same = one_cell_same + 1
                pr.append(iou)
                prBox_correct.append(pr)
                gtBox_correct.append(gt)
                matched_Box.append({'pr': pr, 'gt': gt})
                # print("break iou > threshold",iou, '***', threshold )
                break
            else:
                # print("*********iou:", iou)
                all_iou.append(iou)

        if one_cell_same == 0:
            # print("pr_error:", pr)
            if len(all_iou) == 0:
                max_iou = 0
            else:
                max_iou = max(all_iou)
                if threshold == 0.5:
                    print("max_iou:", max_iou, "confidence:", pr[4])
                    with open(
                            '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/predict_error.csv',
                            'a', encoding='utf8') as f_write:
                        f_write.write(str(max_iou) + "," + str(pr[4]) + '\n')
            pr.append(max_iou)
            prBox_error.append(pr)
    gtBox_error = get_gtBox_error(gtBox, gtBox_correct)
    return precision, recall, same, prBox_error, prBox_correct, gtBox_error, gtBox_correct, matched_Box


"""This function takes bounding boxes of ground truth and predictions and calculates the IOU"""


def calc_iou(ground_truth, detection):
    # exit()
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


def get_gt_row(structure_score_df, gt):
    gt_structure = {}
    for index, row in structure_score_df.iterrows():
        cell = row['cell']
        if cell == str(gt):
            gt_structure = row
            break
    return gt_structure


def analysis_matched_box(matched_Box, file_name, is_match=True):
    file_name = file_name.split('.')[0]
    file_name = file_name + ".xlsx"
    structure_score_file = os.path.join(
        '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/structure_score_collect',
        file_name)
    structure_score_df = pd.read_excel(structure_score_file)

    matched_Box_analysied = []
    if is_match:
        for i in range(len(matched_Box)):
            item = matched_Box[i]
            pr = item['pr']
            gt = item['gt']
            gt_structure = get_gt_row(structure_score_df, gt)
            x_left = pr[0]
            y_left = pr[1]
            x_right = pr[2]
            y_right = pr[3]
            if len(pr) == 5 or len(pr) == 6:
                confidence = pr[4]
            elif len(pr) == 4:
                confidence = 0.20

            confidence_section = 0
            if confidence <= 0.2:
                confidence_section = 0.2
            elif confidence > 0.2 and confidence <= 0.4:
                confidence_section = 0.4
            elif confidence > 0.4 and confidence <= 0.6:
                confidence_section = 0.6
            elif confidence > 0.6 and confidence <= 0.8:
                confidence_section = 0.8
            elif confidence > 0.8:
                confidence_section = 1
            else:
                confidence_section = 0

            pr_entry = {"x_left": x_left,
                        "y_left": y_left,
                        "x_right": x_right,
                        "y_right": y_right,
                        "confidence": confidence,
                        "confidence_section": confidence_section,
                        "is_correct": True,
                        "file_name": file_name}
            new_item = {"pr": pr[:4],
                        'gt': gt,
                        "confidence": confidence,
                        "confidence_section": confidence_section,
                        "is_correct": True,
                        "file_name": file_name
                        }
            for structure_score_name in list(structure_score_df.columns):
                if "Unnamed" not in structure_score_name:
                    new_item[structure_score_name] = gt_structure[structure_score_name]
            matched_Box_analysied.append(new_item)
    else:
        for item in matched_Box:
            gt_structure = get_gt_row(structure_score_df, item)
            new_item = {"pr": 'NAN',
                        'gt': item,
                        "confidence": 'NAN',
                        "confidence_section": 'NAN',
                        "is_correct": 'MISS',
                        "file_name": file_name
                        }

            for structure_score_name in list(structure_score_df.columns):
                if "Unnamed" not in structure_score_name:
                    new_item[structure_score_name] = gt_structure[structure_score_name]

            matched_Box_analysied.append(new_item)

    return matched_Box_analysied


def analysis_predict_error(prBox_error, file_name, is_correct=None):
    prBox_error_entry = []
    for cell in prBox_error:
        x_left = cell[0]
        y_left = cell[1]
        x_right = cell[2]
        y_right = cell[3]
        if len(cell) == 5 or len(cell) == 6:
            confidence = cell[4]
        elif len(cell) == 4:
            confidence = 0.20

        confidence_section = 0
        if confidence <= 0.2:
            confidence_section = 0.2
        elif confidence > 0.2 and confidence <= 0.4:
            confidence_section = 0.4
        elif confidence > 0.4 and confidence <= 0.6:
            confidence_section = 0.6
        elif confidence > 0.6 and confidence <= 0.8:
            confidence_section = 0.8
        elif confidence > 0.8:
            confidence_section = 1
        else:
            confidence_section = 0

        cell_entry = {"x_left": x_left,
                      "y_left": y_left,
                      "x_right": x_right,
                      "y_right": y_right,
                      "confidence": confidence,
                      "confidence_section": confidence_section,
                      "is_correct": is_correct,
                      'file_name': file_name}
        prBox_error_entry.append(cell_entry)
    return prBox_error_entry


def draw_predict(bbox_list, image_path, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # print("output_dir:", output_dir)
    # exit()
    image = cv2.imread(image_path)
    image_name = re.split('/', image_path)[-1]

    for bbox in bbox_list:
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        if len(bbox) == 6:
            import random
            y_text = int(random.uniform(ymin, ymax))
            cv2.putText(image, str(bbox[5])[:4], (xmax, y_text), 2, 1, (255, 255, 0))
    cv2.imwrite(os.path.join(output_dir, image_name), image)


def draw_gt(c_root, ground_truth):
    output_image_only_cell_test = os.path.join(c_root, 'output_image_only_cell_test')
    for file in os.listdir(ground_truth):
        file_path = os.path.join(ground_truth, file)
        gt = read_json(file_path)
        bbox = [item['bbox'] for item in gt]
        draw_predict(bbox, os.path.join(output_image_only_cell_test, file.replace('json', 'jpg')),
                     output_image_only_cell_test + "_gt_draw")


if __name__ == '__main__':

    pre_default = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/output_image_only_cell_test_predict'
    ground_truth_split = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/output_image_only_cell_test_white_all_p20_gt'
    ground_truth = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/coco_split_table_input'
    c_root = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images'

    aug_flag_list = ['', '_img_vertical_lines', '_img_nolines', '_img_horizontal_lines', '_img_both_lines']
    aug_flag_list = ['_img_nolines']
    # aug_flag_list = ['_img_horizontal_lines']
    for aug_flag in aug_flag_list:
        white_2t_p20 = os.path.join(c_root, 'output_image_only_cell_test' + aug_flag + '_white_2t_p20' + '_predict')
        white_3t_p20 = os.path.join(c_root, 'output_image_only_cell_test' + aug_flag + '_white_3t_p20' + '_predict')
        white_all_p20 = os.path.join(c_root, 'output_image_only_cell_test' + aug_flag + '_white_all_p20' + '_predict')
        no_white = os.path.join(c_root, 'output_image_only_cell_test' + aug_flag + '_predict')
        all_retrain = os.path.join(c_root, 'output_image_only_cell_test' + aug_flag + '_predict_all_retrain')
        # tta = os.path.join(c_root, 'output_image_only_cell_test' + '_img_tta')
        # tta = os.path.join(c_root, 'output_image_only_cell_test_img_tta_white_3t_p20_predict')
        tta = os.path.join(c_root, 'output_image_only_cell_test_img_tta_predict_all_retrain')
        tta = os.path.join(c_root, 'output_image_only_cell_test_img_tta_predict_all_retrain_no_trick')
        # output_image_only_cell_test_img_tta_predict_category_softmax
        # tta = os.path.join(c_root, 'output_image_only_cell_test_img_tta_predict_category_softmax')
        # tta = os.path.join(c_root, 'output_image_only_cell_test' + '_img_tta_predict')
        # tta = os.path.join(c_root, 'output_image_only_cell_test_img_tta_white_all_p20_predict')
        # output_image_only_cell_test_img_tta_white_all_p20_predict
        # aug_predict_list = [white_2t_p20, white_3t_p20, white_all_p20, no_white]
        aug_predict_list = [all_retrain]
        aug_predict_list = [tta]
        prBox_entry_list = []
        matched_Box_analysied_all_file = []
        for item_white in aug_predict_list:
            item_white_only = re.split('/', item_white)[-1]
            print("item_white_only:", item_white_only)
            item_white_only = item_white_only + '_no_trick'
            print("\n\n")
            pre_default = item_white
            parser = get_args(ground_truth, pre_default)
            args = parser.parse_args()

            predDirectory = args.predDirectory
            predFiles = os.listdir(predDirectory)

            total_precision = 0.0
            total_recall = 0.0
            total_same = 0.0
            evaluate_res = []
            for thresh in range(5, 10):
                threshold = thresh / 10

                for index in range(len(predFiles)):
                    file_name = predFiles[index]

                    precision, recall, same, prBox_error, prBox_correct, gtBox_error, gtBox_correct, matched_Box = check_bbox(
                        index, threshold, predFiles=predFiles, args=args)

                    draw_output_path = pre_default + "_predict_error_draw"
                    if not os.path.exists(draw_output_path):
                        os.mkdir(draw_output_path)

                    if threshold == 0.5:
                        print("11111****")
                        draw_predict(prBox_error, os.path.join(c_root + '/output_image_only_cell_test',
                                                               predFiles[index].replace('json', 'jpg')),
                                     draw_output_path)

                    pre_error_path = pre_default + "_predict_error"
                    if not os.path.exists(pre_error_path):
                        os.mkdir(pre_error_path)
                    prBox_error_entry = analysis_predict_error(prBox_error, file_name, is_correct="False")
                    prBox_correct_entry = analysis_predict_error(prBox_correct, file_name, is_correct="True")
                    matched_Box_analysied = analysis_matched_box(matched_Box, file_name, is_match=True)
                    matched_Box_analysied_error = analysis_matched_box(gtBox_error, file_name, is_match=False)
                    matched_Box_analysied_all_file = matched_Box_analysied_all_file + matched_Box_analysied + matched_Box_analysied_error
                    prBox_entry_list = prBox_entry_list + prBox_error_entry + prBox_correct_entry
                    write_json(os.path.join(pre_error_path, predFiles[index]), prBox_error)
                    total_precision += precision
                    total_recall += recall
                    total_same += same

                avg_precision = total_same / total_precision
                avg_recall = total_same / total_recall
                avg_f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)

                evaluate_res.append(
                    {"Precision": avg_precision, "avg_recall": avg_recall, "F1": avg_f1, "threshold": threshold})
                print(
                    f"Precision: {round(avg_precision, 3)}; Recall: {round(avg_recall, 3)}; F1: {round(avg_f1, 3)}; threshold: {threshold}")
                print("-" * 50)

                prBox_error_entry_df = pd.DataFrame(prBox_entry_list)
                prBox_error_entry_df.to_excel(os.path.join(
                    '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/output_image_only_cell_test_img_evaluate',
                    item_white_only + '_predict_entry_softmax' + str(thresh) + '_2.xlsx'))

                matched_Box_analysied_all_file_df = pd.DataFrame(matched_Box_analysied_all_file)
                matched_Box_analysied_all_file_df.to_excel(os.path.join(
                    '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/output_image_only_cell_test_img_evaluate',
                    item_white_only + '_structure_score_analysied_' + str(thresh) + '_2.xlsx'))

            evaluate_res = pd.DataFrame(evaluate_res)
            evaluate_res.to_excel(os.path.join(
                '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/output_image_only_cell_test_img_evaluate',
                item_white_only + '.xlsx'))
