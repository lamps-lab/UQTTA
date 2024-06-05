import json
import os
import shutil
from random import sample

# 设置参数
ann_file_path = '/data/cs_lzhan011/uq/mmdetection/data/P1M_Table_Transformer/Structure/Annotations/test/test_coco_format.json'
img_prefix = '/data/cs_lzhan011/uq/mmdetection/data/P1M_Table_Transformer/Structure/Annotations/test/image'
output_ann_file_path = '/data/cs_lzhan011/uq/mmdetection/data/P1M_Table_Transformer/Structure/Annotations/test/test_coco_format_100.json' # 请替换为新的注释文件保存路径
output_img_prefix = '/data/cs_lzhan011/uq/mmdetection/data/P1M_Table_Transformer/Structure/Annotations/test/image_100' # 请替换为新的图片保存路径
num_samples_to_keep = 100 # 保留的样本数量

# 读取原始注释文件
with open(ann_file_path, 'r') as f:
    data = json.load(f)

# 随机选择样本
selected_images = sample(data['images'], num_samples_to_keep)

# 获取选中图片的ID
selected_image_ids = set([img['id'] for img in selected_images])

# 筛选对应的注释
selected_annotations = [ann for ann in data['annotations'] if ann['image_id'] in selected_image_ids]

# 更新注释文件数据
data['images'] = selected_images
data['annotations'] = selected_annotations

# 保存新的注释文件
with open(output_ann_file_path, 'w') as f:
    json.dump(data, f)

# 复制选中的图片到新目录
if not os.path.exists(output_img_prefix):
    os.makedirs(output_img_prefix)

for img in selected_images:
    filename = img['file_name']
    src_path = os.path.join(img_prefix, filename)
    dst_path = os.path.join(output_img_prefix, filename)
    shutil.copy(src_path, dst_path)

print(f"完成！保留了{len(selected_images)}个样本，并复制了对应的图片。")
