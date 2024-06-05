from mmcv import Config
config = Config.fromfile('Config_Customer/config_mm_test_3_instance.py')
print(config.pretty_text)


# ValueError: Expected models: {'cascade r-cnn', 'centernet', 'panopticfpn', 'deformable convolutional networks v2', 'groie', 'mask r-cnn', 'localization distillation', 'generalized focal loss', 'faster r-cnn', 'dyhead', 'centripetalnet', 'detr', 'ghm', 'group normalization', 'foveabox', 'atss', 'sparse r-cnn', 'yolof', 'pisa', 'fcos', 'scnet', 'detectors', 'label assignment distillation', 'carafe', 'cornernet', 'retinanet', 'nas-fcos', 'solo', 'yolact', 'rethinking imagenet pre-training', 'deformable detr', 'cascade mask r-cnn', 'rethinking classification and localization for object detection', 'libra r-cnn', 'dynamic r-cnn', 'instaboost', 'tridentnet', 'yolox', 'guided anchoring', 'paa', 'gcnet', 'htc', 'ssd', 'seesaw loss', 'yolov3', 'freeanchor', 'autoassign', 'fsaf', 'pafpn', 'reppoints', 'feature pyramid grids', 'empirical attention', 'cascade rpn', 'mask scoring r-cnn', 'nas-fpn', 'weight standardization', 'tood', 'grid r-cnn', 'deformable convolutional networks', 'vfnet', 'pointrend', 'sabl', 'queryinst'}, but got {'cascade msk rcnn'}
