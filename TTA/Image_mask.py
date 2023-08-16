
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import json
def clamp(pv):
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    else:
        return pv

def read_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f_read:
        coco_json = json.load(f_read)
    return coco_json


def write_json(json_path, json_object):
    with open(json_path, 'w', encoding='utf-8') as f_write:
        json.dump(json_object, f_write)

def gaussian_noise(image):  # 加高斯噪声
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)
            b = image[row, col, 0]  # blue
            g = image[row, col, 1]  # green
            r = image[row, col, 2]  # red
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    dst = cv2.GaussianBlur(image, (15, 15), 0)  # 高斯模糊
    return dst, image

# def judge_white(image_noise, row,col):
#     if image_noise[row, col, 0] == 255:


def gaussian_noise_2(image, one_image_bbox, gt_c_root, image_name, aug_flag='no'):  # 加高斯噪声
    h, w, c = image.shape
    image_origin = copy.deepcopy(image)
    image_noise = copy.deepcopy(image)
    image_decrease_all = copy.deepcopy(image)
    image_decrease_3 = copy.deepcopy(image)
    image_decrease_2 = copy.deepcopy(image)
    # random.shuffle(one_image_bbox)
    one_image_bbox_decress_2 = copy.deepcopy(one_image_bbox)
    one_image_bbox_decress_3 = copy.deepcopy(one_image_bbox)
    one_image_bbox_decress_all = copy.deepcopy(one_image_bbox)
    for i in range(len(one_image_bbox)-1,-1,-1):
        if i % 10 in [0,1, 2,3,4,5,6,7,8,9]:
            one_bbox = one_image_bbox[i]['bbox']
            image_decrease_2_total_pixel = []
            image_decrease_3_total_pixel = []
        # for one_bbox in one_image_bbox:
            print("one_bbox:", one_bbox)
            print("image.shape:",image.shape)
            for row in range(one_bbox[1], one_bbox[3] ):
                for col in range(one_bbox[0],  one_bbox[2]):
                    s = np.random.normal(0, 20, 3)
                    b = image_origin[row, col, 0]  # blue
                    g = image_origin[row, col, 1]  # green
                    r = image_origin[row, col, 2]  # red
                    image_noise[row, col, 0] = clamp(b + s[0])
                    image_noise[row, col, 1] = clamp(g + s[1])
                    image_noise[row, col, 2] = clamp(r + s[2])

                    image_decrease_2[row, col, 0] = clamp(b * 2)
                    image_decrease_2[row, col, 1] = clamp(g * 2)
                    image_decrease_2[row, col, 2] = clamp(r * 2)
                    image_decrease_2_total_pixel.append(image_decrease_2[row, col, 0])
                    image_decrease_2_total_pixel.append(image_decrease_2[row, col, 1])
                    image_decrease_2_total_pixel.append(image_decrease_2[row, col, 2])


                    image_decrease_3[row, col, 0] = clamp(b * 3)
                    image_decrease_3[row, col, 1] = clamp(g * 3)
                    image_decrease_3[row, col, 2] = clamp(r * 3)
                    image_decrease_3_total_pixel.append(image_decrease_3[row, col, 0])
                    image_decrease_3_total_pixel.append(image_decrease_3[row, col, 1])
                    image_decrease_3_total_pixel.append(image_decrease_3[row, col, 2])

                    image_decrease_all[row, col, 0] = 255
                    image_decrease_all[row, col, 1] = 255
                    image_decrease_all[row, col, 2] = 255

            if max(image_decrease_2_total_pixel) == min(image_decrease_2_total_pixel):
                one_image_bbox_decress_2.pop(i)
                print("d2:", i)
            if max(image_decrease_3_total_pixel) == min(image_decrease_3_total_pixel):
                one_image_bbox_decress_3.pop(i)
                print("d3:", i)
            one_image_bbox_decress_all.pop(i)

    white_2t_p20_gt = os.path.join(gt_c_root, 'output_image_only_cell_test'+aug_flag+'_white_2t_p20_gt/')
    white_3t_p20_gt = os.path.join(gt_c_root, 'output_image_only_cell_test' + aug_flag + '_white_3t_p20_gt/')
    white_all_p20_gt = os.path.join(gt_c_root, 'output_image_only_cell_test' + aug_flag + '_white_all_p20_gt/')
    if not os.path.exists(white_2t_p20_gt):
        os.mkdir(white_2t_p20_gt)
        os.mkdir(white_3t_p20_gt)
        os.mkdir(white_all_p20_gt)
    write_json(os.path.join(white_2t_p20_gt, image_name[:-4]+".json"), one_image_bbox_decress_2)
    write_json(os.path.join(white_3t_p20_gt, image_name[:-4]+".json"), one_image_bbox_decress_3)
    write_json(os.path.join(white_all_p20_gt, image_name[:-4]+".json"), one_image_bbox_decress_all)

    # dst = cv2.GaussianBlur(image, (15, 15), 0)  # 高斯模糊

    return image_noise, image_decrease_all, image_decrease_2, image_decrease_3

def find_image_ann(coco_ann, image_name):
    one_image_ann = []
    one_image_bbox = []
    for file in coco_ann['images']:
        file_name = file['file_name']
        if file_name == image_name:
            file_id = file['id']
            cells_ann = []
            tables_ann = []
            for ann in coco_ann['annotations']:
                ann_copy = copy.deepcopy(ann)
                image_id = ann_copy['image_id']
                bbox_original = ann_copy['bbox']
                if image_id == file_id:
                    one_image_ann.append(ann_copy)
                    one_image_bbox.append(bbox_original)
    random.shuffle(one_image_bbox)
    one_image_bbox_20 = []
    for i in range(len(one_image_bbox)):
        if i % 10 in [1, 7]:
            one_image_bbox_20.append(one_image_bbox[i])


    return one_image_ann, one_image_bbox, one_image_bbox_20


if __name__ == "__main__":
    table_image_path = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/output_image_only_cell_test_img_vertical_lines'
    aug_flag_list  = ['_img_vertical_lines','_img_nolines','_img_horizontal_lines', '_img_both_lines']
    aug_flag_list = ['_img_nolines']
    for aug_flag in aug_flag_list:
        output_image_white_root = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images'
        table_image_path = os.path.join(output_image_white_root, 'output_image_only_cell_test'+aug_flag)
        ann_path = '/data/cs_lzhan011/project/mmdetection/data/Cell/instances_cell_annotations2014_new.json'
        gt_c_root = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images'
        coco_ann = read_json(ann_path)
        for image_name in os.listdir(table_image_path):
            image_path = os.path.join(table_image_path,image_name)
            src = cv2.imread(image_path)
            one_image_ann, one_image_bbox, one_image_bbox_20 = find_image_ann(coco_ann, image_name)
            # plt.subplot(3, 2, 1)
            # plt.imshow(src)
            # plt.axis('off')
            # plt.title('Offical')
            print("image_name:", image_name)
            print("aug_flag:", aug_flag)
            lab_res = read_json(os.path.join('/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/coco_split_table_input', image_name[:-4]+".json"))

            image_noise, image_decrease_all, image_decrease_2, image_decrease_3 = gaussian_noise_2(src, lab_res, gt_c_root, image_name, aug_flag = aug_flag)

            white_2t_p20 = os.path.join(output_image_white_root, 'output_image_only_cell_test'+aug_flag+'_white_2t_p20')
            white_3t_p20 = os.path.join(output_image_white_root, 'output_image_only_cell_test' + aug_flag + '_white_3t_p20')
            white_all_p20 = os.path.join(output_image_white_root, 'output_image_only_cell_test' + aug_flag + '_white_all_p20')
            if not os.path.exists(white_2t_p20):
                os.mkdir(white_2t_p20)
                os.mkdir(white_3t_p20)
                os.mkdir(white_all_p20)

            cv2.imwrite(os.path.join(white_2t_p20, image_name), image_decrease_2)
            cv2.imwrite(os.path.join(white_3t_p20, image_name), image_decrease_3)
            cv2.imwrite(os.path.join(white_all_p20, image_name), image_decrease_all)



