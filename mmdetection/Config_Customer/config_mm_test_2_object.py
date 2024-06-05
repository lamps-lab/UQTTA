_base_ = '../Config_Customer/yolov3_mobilenetv2_mstrain-416_300e_coco.py'

data = dict(
    samples_per_gpu=1,
    train=dict(dataset=dict(
        ann_file='data/General_Dataset/orig.json',
        img_prefix='data/Orig_Image/',
        classes=("table",)
        )),

    val=dict(
        ann_file='data/General_Dataset/orig.json',
        img_prefix='data/Orig_Image/',
        classes=("table",)
    ),
    test=dict(
            ann_file='data/General_Dataset/orig.json',
            img_prefix='data/Orig_Image/',
            classes=("table",)
            ),
)

model = dict(bbox_head =
            dict(
                num_classes=1,
                ))

runner = dict(type='EpochBasedRunner', max_epochs=3)
# model = dict(roi_head = dict(mask_head=dict(num_classes=1)))
load_from = 'yolov3_mobilenetv2_mstrain-416_300e_coco_20210718_010823-f68a07b3.pth'