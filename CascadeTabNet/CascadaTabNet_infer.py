from CascadeTabNet.Table_Structure_Recognition.border import border

from mmdetection.mmdet.apis import inference_detector, show_result_pyplot, init_detector
import cv2
from CascadeTabNet.Table_Structure_Recognition.Functions.blessFunc import borderless
import lxml.etree as etree
import glob
import os
import json
from Augmentation.augmentation import aug_process
import gdown
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def write_json(json_path, json_object):
    with open(json_path, 'w', encoding='utf-8') as f_write:
        json.dump(json_object, f_write)


def cascade_infer(config_file, checkpoint_file, image_path, predict_res):
    model = init_detector(config_file, checkpoint_file)

    # List of images in the image_path
    img_dir = os.listdir(image_path)

    for path in img_dir:
        image_fullPath = os.path.join(image_path, path)

        result = inference_detector(model, image_fullPath)

        if not os.path.exists(image_path + "_infer_draw/"):
            os.mkdir(image_path + "_infer_draw/")
        out_file = image_path + "_infer_draw/" + path
        show_result_pyplot(model=model, img=image_fullPath, result=result, out_file=out_file)

        res_cell = []

        for r in result[0][0]:
            if r[4] > 0:
                r[4] = r[4] * 100
                res_cell.append({"bbox": r[:4].astype(int).tolist()})

        write_json(os.path.join(predict_res, path[:-3] + 'json'), res_cell)
if __name__ == '__main__':

    # before conduct this code, please conduct the code of mmdetection install

    gdown_url = 'https://drive.google.com/file/d/1dCT-OJUEQuGmFlEs86387PhYEsSz1ZcO/view?usp=sharing'
    gdown_output_dir = 'model_dir'
    gdown.download(gdown_url, gdown_output_dir)
    c_root = r'/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images'
    origin_img_path = os.path.join(c_root, 'output_image_only_cell_test')
    img_nolines_dir, img_vertical_lines_dir, img_horizontal_lines_dir, img_both_lines_dir = aug_process(origin_img_path)
    for item_aug_dir in [origin_img_path, img_nolines_dir, img_vertical_lines_dir, img_horizontal_lines_dir,
                         img_both_lines_dir]:
        image_path = item_aug_dir
        predict_res = image_path + "_predict"
        if not os.path.exists(predict_res):
            os.mkdir(predict_res)
            os.mkdir(predict_res + "_bak")
        config_file = '/data/cs_lzhan011/project/mmdetection/work_dirs/config_CascadeTabNet_2_2_train/config_CascadeTabNet_2_2_train.py'
        checkpoint_file = '/data/cs_lzhan011/project/mmdetection/work_dirs/config_CascadeTabNet_2_2_train/epoch_30.pth'
        # image_path = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/output_image_only_cell_test_img_nolines'
        # predict_res = '/data/cs_lzhan011/project/mmdetection/data/Cell_split_train_test/Cell_images/output_image_only_cell_test_img_nolines_predict'
        cascade_infer(config_file, checkpoint_file, image_path, predict_res)
