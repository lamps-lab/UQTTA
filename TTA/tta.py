# python3
import numpy as np
import os
import json

def get_ovr_cell_index(ovr, thresh):
    ovr_cell = {}
    ovr_cell_list = []
    for cell_index in range(len(ovr)):
        if cell_index in ovr_cell_list:
            continue
        cell_index_row = ovr[cell_index]
        over_cell_index = np.where(cell_index_row >= thresh)[0].tolist()
        over_cell_index = [index for index in over_cell_index if index > cell_index]
        ovr_cell[cell_index] = over_cell_index
        ovr_cell_list.append(cell_index)
        ovr_cell_list = ovr_cell_list + over_cell_index
    return ovr_cell


def nms_tta(x1,y1, x2, y2, order, areas, thresh ):
    # xx1 = np.maximum(x1[1:], x1[order[1:]])
    # yy1 = np.maximum(y1[1:], y1[order[1:]])
    # xx2 = np.minimum(x2[1:], x2[order[1:]])
    # yy2 = np.minimum(y2[1:], y2[order[1:]])

    keep = []
    ovr_cell = {}
    while order.size > 0:
        i = order[0]  # 最大得分box的坐标索引
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])  # 最高得分的boax与其他box的公共部分(交集)

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)  # 求高和宽，并使数值合法化
        inter = w * h  # 其他所有box的面积
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  # IOU:交并比

        inds = np.where(ovr <= thresh)[0]  # ovr小表示两个box交集少，可能是另一个物体的框，故需要保留
        inds_ovr = np.where(ovr >= thresh)[0]
        ovr_cell[i] = order[inds_ovr + 1]
        order = order[inds + 1]  # iou小于阈值的框

    return ovr_cell


def tta(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    x11 = [x1]
    y11 = [y1]
    x22 = [x2]
    y22 = [y2]

    xx1 = np.maximum(x11, np.transpose(x11))
    yy1 = np.maximum(y11, np.transpose(y11))
    xx2 = np.minimum(x22, np.transpose(x22))
    yy2 = np.minimum(y22, np.transpose(y22))

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)  # Find the height and width, and legalize the values
    inter = w * h  # The area all other boxes

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # All box area
    # print("all box aress: ", areas)
    order = scores.argsort()[::-1]  # Arrange in descending order to get the coordinate index of scores

    ovr = inter / (areas + areas - inter)  # IOU: Intersection over Union

    ovr_cell = get_ovr_cell_index(ovr, thresh)

    keep = []
    ovr_cell = nms_tta(x1, y1, x2, y2, order, areas, thresh)
    return keep, ovr_cell


def read_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f_read:
        json_read = json.load(f_read)

    return json_read


def write_json(json_path, json_object):
    with open(json_path, 'w', encoding='utf-8') as f_write:
        json.dump(json_object, f_write)


def add_aug_type_flag(aug_res, aug_type):
    aug_type_map = {'both': 1,
                    'vertical': 2,
                    'horizontal': 3,
                    'nolines': 4,
                    'original': 0}
    assert aug_type in aug_type_map, 'aug_type must be in the aug_type_map, please check'
    return [item['bbox'] + [1, aug_type_map[aug_type]] for item in aug_res]

def calc_iou(ground_truth, detection):
    # determine the (x, y)-coordinates of the intersection rectangle
    # if ground_truth[0] == detection[0]:
    #     if detection[0] == 543:
    #         pass


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

def remove_left_iod_threshold(original_json, other_res_json, threshold = 1.0):
    for i in range(len(other_res_json)-1,-1,-1):
        for y in range(len(original_json)):
            iou, precision_iou, recall_iou = calc_iou(original_json[y]['bbox'], other_res_json[i]['bbox'])
            if precision_iou >= threshold and recall_iou < 0.5:
                print("precision_iou:", precision_iou)
                print("recall_iou:", recall_iou)
                print("other_res_json[i]:", other_res_json[i])
                print("original_json[y]:", original_json[y])
                other_res_json.pop(i)
                break

    return other_res_json



def receive_cascade_output(c_root):

    # Here is the tta five kind of input direction.
    # predict_path = '_predict'
    # predict_path = '_predict_softmax'
    # predict_path = '_white_3t_p20_predict'
    # predict_path  = '_predict_category_softmax'
    predict_path = '_predict_all_retrain'
    predict_img_both_lines = os.path.join(c_root, 'output_image_only_cell_test') + '_img_both_lines' + predict_path
    predict_img_vertical_lines = os.path.join(c_root,
                                              'output_image_only_cell_test') + '_img_vertical_lines' + predict_path
    predict_img_horizontal_lines = os.path.join(c_root,
                                                'output_image_only_cell_test') + '_img_horizontal_lines' +predict_path
    predict_img_nolines = os.path.join(c_root, 'output_image_only_cell_test') + '_img_nolines' + predict_path
    predict_original = os.path.join(c_root, 'output_image_only_cell_test') + predict_path

    all_collect_json = []
    for item_name in os.listdir(predict_img_both_lines):
        print("item_name:", item_name)
        # if 'cTDaR_t10121_0' in item_name:
        #     pass
        # else:
        #     continue
        path_img_both_lines = os.path.join(predict_img_both_lines, item_name)
        path_img_vertical_lines = os.path.join(predict_img_vertical_lines, item_name)
        path_img_horizontal_lines = os.path.join(predict_img_horizontal_lines, item_name)
        path_img_nolines = os.path.join(predict_img_nolines, item_name)
        path_predict_original = os.path.join(predict_original, item_name)
        img_both_lines_json = read_json(path_img_both_lines)
        img_vertical_lines_json = read_json(path_img_vertical_lines)
        img_horizontal_lines_json = read_json(path_img_horizontal_lines)
        img_nolines_json = read_json(path_img_nolines)
        original_json = read_json(path_predict_original)
        # img_both_lines_json = remove_left_iod_threshold(original_json, img_both_lines_json, threshold=1.0)
        # img_vertical_lines_json = remove_left_iod_threshold(original_json, img_vertical_lines_json, threshold=1.0)
        # img_horizontal_lines_json = remove_left_iod_threshold(original_json, img_horizontal_lines_json, threshold=1.0)


        collect_json = add_aug_type_flag(img_both_lines_json, 'both') + add_aug_type_flag(img_vertical_lines_json,
                                                                                          'vertical') \
                       + add_aug_type_flag(img_horizontal_lines_json, 'horizontal') + add_aug_type_flag(
            img_nolines_json, 'nolines') + add_aug_type_flag(
            original_json, 'original')
        all_collect_json.append([item for item in collect_json])

        if collect_json == []:
            output_dir = os.path.join(c_root, 'output_image_only_cell_test_img_tta') + predict_path
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            print("item_name:::::::", item_name)
            print("output_dir:", output_dir)
            # output_dir, Here is the tta output direction.
            write_json(os.path.join(output_dir, item_name), [])
        else:
            dets = np.array(collect_json)
            rtn_box, ovr_cell = tta(dets, 0.5)
            print("ovr_cell:", ovr_cell)
            combine_res = find_overlap_area(ovr_cell, collect_json)
            output_dir = os.path.join(c_root, 'output_image_only_cell_test_img_tta')+ predict_path+"_no_trick"
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            print("item_name:::::::", item_name)
            print("output_dir:", output_dir)
            # output_dir, Here is the tta output direction.
            write_json(os.path.join(output_dir, item_name), combine_res)

    return all_collect_json


def find_overlap_area(ovr_cell, collect_json):
    combine_res = []
    for k, v in ovr_cell.items():
        k_cell = collect_json[k]
        overlap_cell = []
        overlap_cell.append(k_cell)
        for vi in v:
            vi_cell = collect_json[vi]
            overlap_cell.append(vi_cell)
        overlap_cell = np.array(overlap_cell)
        x1 = min(overlap_cell[:, 0])
        y1 = min(overlap_cell[:, 1])
        x2 = max(overlap_cell[:, 2])
        y2 = max(overlap_cell[:, 3])
        binary_score = sum(overlap_cell[:, 4])
        binary_len = len(overlap_cell[:, 4])
        total_score = binary_len * 100
        img_tmp = np.zeros((x2 + 1, y2 + 1))

        for cell in overlap_cell:
            img_tmp[cell[0]:cell[2], cell[1]:cell[3]] = img_tmp[cell[0]:cell[2], cell[1]:cell[3]] + 1

        max_cnt = img_tmp.max()
        combine_res.append({'bbox': [int(x1), int(y1), int(x2), int(y2), min(max_cnt / 5, 1)]})

        # combine_res.append({'bbox': [int(x1), int(y1), int(x2), int(y2), min(binary_score / 500, 1)]})
        # combine_res.append({'bbox': [int(x1), int(y1), int(x2), int(y2), min(binary_score / total_score, 1)]})
        # print("combine_res:", combine_res)
    return combine_res


if __name__ == '__main__':
    c_root = '/data/cs_lzhan011/uq/mmdetection/data/Cell_split_train_test/Cell_images'
    # c_root = '/home/lei/fsdownload/tta_20221130'
    all_collect_json = receive_cascade_output(c_root)

# import cv2
# import numpy as np
# import random


# collect_json = all_collect_json[5]
# img = np.zeros((1000, 1000))
# dets = np.array([[83, 54, 165, 163, 1], [67, 48, 118, 132, 1], [91, 38, 192, 171, 1], [59, 120, 137, 368, 1]], np.float)
# dets = np.array([[20, 20, 100, 100, 1], [21, 21, 110, 110, 1], [30, 30, 130, 130, 1], [160, 160, 260, 260, 1]],
#                 np.float)
# dets = np.array(collect_json)
# img_cp = img.copy()
# for box in dets.tolist():  # 显示待测试框及置信度
#     x1, y1, x2, y2, score = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[-1]
#     y_text = int(random.uniform(y1, y2))
#     cv2.rectangle(img_cp, (x1, y1), (x2, y2), (255, 255, 255), 2)
#     cv2.putText(img_cp, str(score), (x2 - 30, y_text), 2, 1, (255, 255, 0))
# cv2.imshow("ori_img", img_cp)
#
# rtn_box, ovr_cell = nms(dets, 0.5)  # 0.3为faster-rcnn中配置文件的默认值
# print("rtn_box:", rtn_box)
# print("ovr_cell:", ovr_cell)
# # find_overlap_area(ovr_cell, collect_json)
# cls_dets = dets[rtn_box, :]
# print("nms box:", cls_dets)

# img_cp = img.copy()
# for box in cls_dets.tolist():
#     x1, y1, x2, y2, score = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[-1]
#     y_text = int(random.uniform(y1, y2))
#     cv2.rectangle(img_cp, (x1, y1), (x2, y2), (255, 255, 255), 2)
#     cv2.putText(img_cp, str(score), (x2 - 30, y_text), 2, 1, (255, 255, 0))
# cv2.imshow("black1_nms", img_cp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # test
# if __name__ == "__main__":
#     dets = np.array([[30, 20, 230, 200, 1],
#                      [50, 50, 260, 220, 1],
#                      [210, 30, 420, 5, 1],
#                      [430, 280, 460, 360, 1]])
#     thresh = 0.35
#     keep_dets = py_nms(dets, thresh)
#     print(keep_dets)
#     print(dets[keep_dets])
