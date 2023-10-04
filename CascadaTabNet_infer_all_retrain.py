from CascadeTabNet.Table_Structure_Recognition.border import border

from mmdetection.mmdet.apis import inference_detector, show_result_pyplot, init_detector
import cv2
from CascadeTabNet.Table_Structure_Recognition.Functions.blessFunc import borderless
import lxml.etree as etree
import glob
import os
import json
# from Augmentation.augmentation import aug_process
# import gdown
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def replace_proposal_score(result):
    bbox_results = result[0]
    proposal_score_list = result[2]
    # proposal_score_imgs = []
    # for num_classes_i in range(len(bbox_results)):
    #     proposal_score_img_classes = []
    #     for entry_i in range(len(bbox_results[num_classes_i])):
    #         entry = bbox_results[num_classes_i][entry_i]
    #         bbox_results[num_classes_i][entry_i][-1] = proposal_score_list[num_classes_i][entry_i].tolist()
    result = (bbox_results, result[1])
    return result


def write_json(json_path, json_object):
    with open(json_path, 'w', encoding='utf-8') as f_write:
        json.dump(json_object, f_write)


def cascade_infer(config_file, checkpoint_file, image_path, predict_res):
    model = init_detector(config_file, checkpoint_file)
    # import ttach as tta
    # model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
    # List of images in the image_path
    img_dir = os.listdir(image_path)

    for ii in  range(len(img_dir)):

        path = img_dir[ii]
        # path_next = img_dir[ii+1]

        image_fullPath = os.path.join(image_path, path)
        # image_fullPath_next = os.path.join(image_path,path_next)
        # image_fullPath = (image_fullPath, image_fullPath_next)
        result = inference_detector(model, image_fullPath)
        # print("result:", result)
        result = replace_proposal_score(result)
        # print("\n\n\n")
        # print("result2222:", result)
        # exit()
        if not os.path.exists(image_path + "_infer_draw/"):
            os.mkdir(image_path + "_infer_draw/")
        out_file = image_path + "_infer_draw/" + path
        show_result_pyplot(model=model, img=image_fullPath, result=result, out_file=out_file)

        res_cell = []

        for r in result[0][0]:
            if r[4] > 0:
                r[4] = r[4] * 100
                res_cell.append({"bbox": r[:4].astype(int).tolist()})
        print("os.path.join(predict_res, path[:-3] + 'json'):", os.path.join(predict_res, path[:-3] + 'json'))
        # Here will produce 5 output directions to save the 5 kinds of prediction result.
        write_json(os.path.join(predict_res, path[:-3] + 'json'), res_cell)



if __name__ == '__main__':

    # before conduct this code, please conduct the code of mmdetection install
    # here is the introduction of installing mmdetection: https://github.com/open-mmlab/mmdetection
    # must run: pip install mmcv-full==1.7.0


    c_root = r"/data/cs_lzhan011/uq/mmdetection/data/Cell_split_train_test/Cell_images"
    origin_img_path = os.path.join(c_root, 'output_image_only_cell_test')
    # img_nolines_dir, img_vertical_lines_dir, img_horizontal_lines_dir, img_both_lines_dir = aug_process(origin_img_path)

    aug_flag = '_img_vertical_lines'
    aug_flag = '_img_nolines'
    aug_flag = '_img_horizontal_lines'
    aug_flag = '_img_both_lines'
    # aug_flag_list = ['','_img_vertical_lines', '_img_nolines', '_img_horizontal_lines', '_img_both_lines']
    # aug_flag_list = ['_img_nolines']
    aug_flag_list = ['','_img_vertical_lines', '_img_nolines', '_img_horizontal_lines', '_img_both_lines']
    # aug_flag_list contain 5 kinds of input datasets.
    for aug_flag in aug_flag_list:
        # white_2t_p20 = os.path.join(c_root, 'output_image_only_cell_test' + aug_flag + '_white_2t_p20')
        # white_3t_p20 = os.path.join(c_root, 'output_image_only_cell_test' + aug_flag + '_white_3t_p20')
        # white_all_p20 = os.path.join(c_root, 'output_image_only_cell_test' + aug_flag + '_white_all_p20')
        no_white = os.path.join(c_root, 'output_image_only_cell_test' + aug_flag)
        print("no_white:", no_white)
        # dir_list = [origin_img_path, img_nolines_dir, img_vertical_lines_dir, img_horizontal_lines_dir,
        #                      img_both_lines_dir]

        # dir_list = [white_2t_p20,white_3t_p20, white_all_p20 , no_white]
        dir_list = [no_white]

        # The loop will load five kinds of retrained model.
        for item_aug_dir in dir_list:
            image_path = item_aug_dir
            # predict_res = image_path + "_predict_category_softmax"
            # predict_res = item_aug_dir + '_predict'
            predict_res = item_aug_dir +'_predict_all_retrain'
            if not os.path.exists(predict_res):
                os.mkdir(predict_res)
                os.mkdir(predict_res + "_bak")
            if aug_flag == '':
                config_file = '/data/cs_lzhan011/uq/mmdetection/work_dirs/config_CascadeTabNet_2_2_train/config_CascadeTabNet_2_2_train.py'
                checkpoint_file = '/data/cs_lzhan011/uq/mmdetection/work_dirs/config_CascadeTabNet_2_2_train/epoch_30.pth'
                # image_path = '/data/cs_lzhan011/uq/mmdetection/data/Cell_split_train_test/Cell_images/output_image_only_cell_test_img_nolines'
                # predict_res = '/data/cs_lzhan011/uq/mmdetection/data/Cell_split_train_test/Cell_images/output_image_only_cell_test_img_nolines_predict'
            elif aug_flag == '_img_vertical_lines':
                config_file = '/data/cs_lzhan011/uq/mmdetection/work_dirs/config_CascadeTabNet_2_2_train_vert_lines/config_CascadeTabNet_2_2_train_vert_lines.py'
                checkpoint_file = '/data/cs_lzhan011/uq/mmdetection/work_dirs/config_CascadeTabNet_2_2_train_vert_lines/epoch_30.pth'
            elif aug_flag == '_img_nolines':
                config_file = '/data/cs_lzhan011/uq/mmdetection/work_dirs/config_CascadeTabNet_2_2_train_nolines/config_CascadeTabNet_2_2_train_nolines.py'
                checkpoint_file = '/data/cs_lzhan011/uq/mmdetection/work_dirs/config_CascadeTabNet_2_2_train_nolines/epoch_30.pth'
            elif aug_flag == '_img_horizontal_lines':
                config_file = '/data/cs_lzhan011/uq/mmdetection/work_dirs/config_CascadeTabNet_2_2_train_hori_lines/config_CascadeTabNet_2_2_train_hori_lines.py'
                checkpoint_file = '/data/cs_lzhan011/uq/mmdetection/work_dirs/config_CascadeTabNet_2_2_train_hori_lines/epoch_30.pth'
            elif aug_flag == '_img_both_lines':
                config_file = '/data/cs_lzhan011/uq/mmdetection/work_dirs/config_CascadeTabNet_2_2_train_both_lines/config_CascadeTabNet_2_2_train_both_lines.py'
                checkpoint_file = '/data/cs_lzhan011/uq/mmdetection/work_dirs/config_CascadeTabNet_2_2_train_both_lines/epoch_30.pth'
            else:
                config_file = '/data/cs_lzhan011/uq/mmdetection/work_dirs/config_CascadeTabNet_2_2_train/config_CascadeTabNet_2_2_train.py'
                checkpoint_file = '/data/cs_lzhan011/uq/mmdetection/work_dirs/config_CascadeTabNet_2_2_train/epoch_30.pth'

            cascade_infer(config_file, checkpoint_file, image_path, predict_res)
