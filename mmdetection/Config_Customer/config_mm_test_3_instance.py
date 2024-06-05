_base_ = '../Config_Customer/mask_rcnn_r50_fpn_2x_coco.py'


data = dict(
    samples_per_gpu=1,
    train=dict(ann_file='data/General_Dataset/orig.json',
        img_prefix='data/Orig_Image/',
        classes=("table",)
        ),

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


model = dict(roi_head=
            dict( bbox_head= dict(
                num_classes=1),
                mask_head= dict(
                num_classes=1)
                ))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.00001,
    step=[2, 6])

runner = dict(type='EpochBasedRunner', max_epochs=3)
checkpoint_config = dict(interval=1/2)
# model = dict(roi_head = dict(mask_head=dict(num_classes=1)))
load_from = 'mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth'
