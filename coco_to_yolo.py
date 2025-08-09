import json
import os

def convert_coco_to_yolo(coco_file, output_dir):
    """
    将COCO格式的标注文件转换为YOLO格式的标签文件
    每个图片生成一个对应的txt文件
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取COCO标注文件
    with open(coco_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)
    
    # 创建image_id到file_name的映射
    image_info = {}
    for image in coco_data['images']:
        image_info[image['id']] = {
            'file_name': image['file_name'],
            'width': image['width'],
            'height': image['height']
        }
    
    # 创建category_id到类别索引的映射（YOLO格式要求从0开始）
    category_map = {}
    for i, category in enumerate(coco_data['categories']):
        category_map[category['id']] = i
    
    # 按image_id分组annotations
    annotations_by_image = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)
    
    # 为每张图片生成YOLO格式的标签文件
    for image_id, annotations in annotations_by_image.items():
        if image_id not in image_info:
            continue
            
        image_data = image_info[image_id]
        file_name = image_data['file_name']
        
        # 生成txt文件名（将图片扩展名替换为.txt）
        txt_filename = os.path.splitext(file_name)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)
        
        # 写入YOLO格式的标注
        with open(txt_path, 'w', encoding='utf-8') as f:
            for ann in annotations:
                # 获取边界框信息
                bbox = ann['bbox']  # COCO格式: [x, y, width, height]
                x_min, y_min, width, height = bbox
                
                # 转换为YOLO格式: [class_id, x_center, y_center, width, height]（相对值）
                class_id = category_map[ann['category_id']]
                img_width = image_data['width']
                img_height = image_data['height']
                
                # 计算中心点和归一化
                x_center = (x_min + width / 2) / img_width
                y_center = (y_min + height / 2) / img_height
                norm_width = width / img_width
                norm_height = height / img_height
                
                # 写入YOLO格式的行
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
    
    # 生成包含所有图片路径的images_train.txt文件
    images_train_path = os.path.join(os.path.dirname(coco_file), "images_train.txt")
    with open(images_train_path, 'w', encoding='utf-8') as f:
        for image in coco_data['images']:
            # 为每个图片添加一行（这里假设图片文件与JSON文件在同一目录下）
            image_path = os.path.join(image['file_name'])
            # 如果图片没有扩展名，可以添加默认的.jpg扩展名
            if not os.path.splitext(image['file_name'])[1]:
                image_path += ".jpg"
            f.write(f"{image_path}\n")
    
    print(f"转换完成！共生成 {len(annotations_by_image)} 个标签文件到目录: {output_dir}")
    print(f"生成图片列表文件: {images_train_path}")

# 使用示例
if __name__ == "__main__":
    coco_file_path = "E:/毕设开题/yolov8/instances_default.json"
    output_directory = "E:/毕设开题/yolov8/labels"
    
    convert_coco_to_yolo(coco_file_path, output_directory)