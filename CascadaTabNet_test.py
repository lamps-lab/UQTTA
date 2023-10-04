from CascadeTabNet.Table_Structure_Recognition.border import border

from mmdetection.mmdet.apis import inference_detector, show_result_pyplot, init_detector
import cv2
from CascadeTabNet.Table_Structure_Recognition.Functions.blessFunc import borderless
import lxml.etree as etree
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def replace_proposal_score(result):
    bbox_results = result[0]
    proposal_score_list = result[2]
    proposal_score_imgs = []
    for num_classes_i in range(len(bbox_results)):
        proposal_score_img_classes = []
        for entry_i in range(len(bbox_results[num_classes_i])):
            entry = bbox_results[num_classes_i][entry_i]
            bbox_results[num_classes_i][entry_i][-1] = proposal_score_list[num_classes_i][entry_i].tolist()
    result = (bbox_results, result[1])
    return result

############ To Do ############
image_path = '/content/drive/MyDrive/Modern_TRACKB2_CascadeNet/'
xmlPath = '/content/drive/MyDrive/Modern_TRACKB2_CascadeNet/'

config_fname = "/content/CascadeTabNet/Config/cascade_mask_rcnn_hrnetv2p_w32_20e.py"
checkpoint_path = "//"
epoch = 'epoch_36.pth'
##############################

image_path = '/data/cs_lzhan011/project/CascadeTabNet/Data/chunk_images'
xmlPath = '/data/cs_lzhan011/project/CascadeTabNet/Data/orig_chunk'
xmlPath_write = '/data/cs_lzhan011/project/CascadeTabNet/Data/infer_res/'

# config_fname = '/content/mmdetection/configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_20e_coco.py'
# checkpoint_path = '/content/CascadeTabNet/Checkpoints/cascade_rcnn_r50_fpn_20e_coco_bbox_mAP-0.41_20200504_175131-e9872a90.pth'

# config_fname = '/content/mmdetection/mmdet/.mim/configs/hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e_coco.py'
# checkpoint_path = '/content/CascadeTabNet/Model_Pretrained/epoch_24.pth'
# checkpoint_path = '/home/lei/Downloads/epoch_36.pth'
# checkpoint_path = '/home/lei/Downloads/epoch_36_2.pth'

# config_fname = '/content/mmdetection/Config_Customer/config_mm_test.py'
# checkpoint_path = '/content/mmdetection/work_dirs/config_mm_test_class_1/epoch_20.pth'
config_file = '/content/mmdetection/Config_Customer/config_mm_test.py'
checkpoint_file = '/content/mmdetection/work_dirs/config_mm_test/epoch_3.pth'
config_file = '/data/cs_lzhan011/project/mmdetection/work_dirs/config_CascadeTabNet_1/config_CascadeTabNet_1.py'
checkpoint_file = '/data/cs_lzhan011/project/mmdetection/work_dirs/config_CascadeTabNet_1/epoch_3.pth'


envir = "local"
envir = 'hubble'
if envir == 'local':
    config_file = '/content/mmdetection/work_dirs/config_CascadeTabNet_1/config_CascadeTabNet_1.py'
    checkpoint_file = '/home/lei/fsdownload/epoch_5.pth'
    image_path = '/content/CascadeTabNet/Data/chunk_images'
    xmlPath = '/content/CascadeTabNet/Data/orig_chunk'
    xmlPath_write = '/content/CascadeTabNet/Data/infer_res/'
else:
    config_file = '/data/cs_lzhan011/project/mmdetection/work_dirs/config_CascadeTabNet_1/config_CascadeTabNet_1.py'
    checkpoint_file = '/data/cs_lzhan011/project/mmdetection/work_dirs/config_CascadeTabNet_1/epoch_30.pth'
    image_path = '/data/cs_lzhan011/project/CascadeTabNet/Data/chunk_images'
    image_path = '/data/cs_lzhan011/project/CascadeTabNet/Data/test_4_images'
    xmlPath = '/data/cs_lzhan011/project/CascadeTabNet/Data/orig_chunk'
    xmlPath_write = '/data/cs_lzhan011/project/CascadeTabNet/Data/infer_res/'


model = init_detector(config_file, checkpoint_file)

# List of images in the image_path
img_dir = os.listdir(image_path)
print("img_dir")
print(img_dir)
for path in img_dir:
    image_fullPath = os.path.join(image_path, path)
    print("image_fullPath")
    print(image_fullPath)
    xmlPath_write_name = xmlPath_write+image_fullPath.split('/')[-1][:-3]+'xml'
    if os.path.exists(xmlPath_write_name):
        size = os.path.getsize(xmlPath_write_name)
    else:
        size = 0
    print("size")
    print(size)
    # if size > 1000 or '10039' in image_fullPath or '10481' in image_fullPath or '10360' in image_fullPath or '10090' \
    #         in image_fullPath  or '10142'  in image_fullPath or '10044' in image_fullPath or '10314' in image_fullPath:
    #     continue
    # else:
    #     pass
    result = inference_detector(model, image_fullPath)
    print("result1")
    print(result)
    result = replace_proposal_score(result)
    print("result2")
    print(result)
    out_file = image_path+"_infer/" + path
    print("out_file:", out_file)
    show_result_pyplot(model=model, img=image_fullPath, result=result, out_file=out_file)
    # exit()
    res_border = []
    res_bless = []
    res_cell = []
    root = etree.Element("document")
    print("11111")
    ## for border
    for r in result[0][0]:
        if r[4]>0:
            res_border.append(r[:4].astype(int))
    ## for cells
    for r in result[0][1]:
        if r[4]>0:
            r[4] = r[4]*100
            res_cell.append(r.astype(int))
    ## for borderless
    for r in result[0][2]:
        if r[4]>0:
            res_bless.append(r[:4].astype(int))
    print("2222")
    ## if border tables detected
    if len(res_border) != 0:
        print("border root")
        ## call border script for each table in image
        for res in res_border:
            try:
                root.append(border(res,cv2.imread(image_fullPath)))
            except:
                pass
        print("border root")
        print(root)
    print("3333")
    print("res_bless")
    print(res_bless)
    # if borderless tables detected
    if len(res_bless) != 0:
        print("borderless root")
        if len(res_cell) != 0:
            for no,res in enumerate(res_bless):
                print("cvcvcv")
                root.append(borderless(res,cv2.imread(image_fullPath),res_cell))

        print("borderless root")
        print(root)
    # write results to XML file
    print("write image_fullPath: ",xmlPath_write+image_fullPath.split('/')[-1][:-3]+'xml')
    myfile = open(xmlPath_write+image_fullPath.split('/')[-1][:-3]+'xml', "w")
    myfile.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    myfile.write(etree.tostring(root, pretty_print=True,encoding="unicode"))
    myfile.close()