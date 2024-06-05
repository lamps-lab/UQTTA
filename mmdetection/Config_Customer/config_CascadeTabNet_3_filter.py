_base_ = '../Config_Customer/cascade_mask_rcnn_hrnetv2p_w32_20e_coco.py'

data = dict(
    samples_per_gpu=1,
    train=dict(
        ann_file='/data/cs_lzhan011/project/mmdetection/data/Cell/instances_cell_annotations2014_new_filter.json',
        img_prefix='/data/cs_lzhan011/project/CascadeTabNet/Data/chunk_images',
        classes=('cell',)
        ),

    val=dict(
        ann_file='/data/cs_lzhan011/project/mmdetection/data/Cell/instances_cell_annotations2014_new_filter.json',
        img_prefix='/data/cs_lzhan011/project/CascadeTabNet/Data/chunk_images',
        classes=('cell',)
    ),
    test=dict(
        ann_file='/data/cs_lzhan011/project/mmdetection/data/Cell/instances_cell_annotations2014_new_filter.json',
        img_prefix='/data/cs_lzhan011/project/CascadeTabNet/Data/chunk_images',
        classes=('cell',)
            ),
)

model = dict(roi_head = dict(mask_head=dict(num_classes=1),
                             bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]))



# model = dict(roi_head = dict(mask_head=dict(num_classes=1)))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=4000,
    warmup_ratio=0.000001,
    step=[1, 2])

log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
runner = dict(type='EpochBasedRunner', max_epochs=30)
load_from = '/data/cs_lzhan011/project/mmdetection/general_data_epoch_3.pth'