from mmdet.apis import init_detector, inference_detector,show_result_pyplot

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
# 初始化检测器
img='demo/demo.jpg'
img_2='demo/demo_2.jpg'
model = init_detector(config_file, checkpoint_file, device=device)
# 推理演示图像
result = inference_detector(model, img)
show_result_pyplot(model=model,img=img_2,result=result)