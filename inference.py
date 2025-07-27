from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2

# Load a pretrained YOLO11n model
model = YOLO("./BeltDetection.pt")

# Load pose estimation model
pose_model = YOLO("yolov8n-pose.pt")

# Define path to the image file
name = "test2"
source = name + ".jpg"

# Run inference on the source
results = model(source)  # list of Results objects

# Run pose estimation
pose_results = pose_model(source)

# 获取安全带检测结果
belt_results = results[0]

# 获取姿态估计结果
pose_result = pose_results[0]
keypoints = pose_result.keypoints.data if pose_result.keypoints is not None else None

# 读取原始图像
original_image = cv2.imread(source)

# 绘制检测结果
annotated_image = belt_results.plot()


# 计算两个边界框的交并比(IOU)
def calculate_iou(box1, box2):
    # box格式: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集面积
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算两个框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算IOU
    iou = intersection_area / (box1_area + box2_area - intersection_area)
    return iou


# 查找offground和safebelt框
offground_boxes = []
safebelt_boxes = []

if belt_results.boxes is not None:
    for box in belt_results.boxes:
        cls = int(box.cls.item())
        box_coords = box.xyxy[0].tolist()

        # 收集offground类别 (cls=1)
        if cls == 1:
            offground_boxes.append(box_coords)
        # 收集safebelt类别 (cls=3)
        elif cls == 3:
            safebelt_boxes.append(box_coords)

# 处理offground和safebelt的匹配
matched_pairs = []
for offground_box in offground_boxes:
    for safebelt_box in safebelt_boxes:
        # 计算重叠区域(IOU)
        iou = calculate_iou(offground_box, safebelt_box)
        # 如果重叠区域大于60%，则认为是一对匹配的框
        if iou > 0.4:
            matched_pairs.append((offground_box, safebelt_box))

# 绘制offground框

# 如果有关键点数据且存在匹配的框对，则进行安全带佩戴规范检测
if keypoints is not None and matched_pairs:
    for i, kp in enumerate(keypoints):
        # 每个人的关键点
        if kp.shape[0] >= 17:  # 确保关键点数量足够
            # 获取膝盖关键点 (通常索引为13和14)
            left_knee = kp[13]
            right_knee = kp[14]

            # 检查关键点的可见性
            if left_knee[2] > 0.5 and right_knee[2] > 0.5:
                # 计算膝盖的平均y坐标
                knee_y = min(left_knee[1], right_knee[1])

                # 在图像上绘制膝盖位置（用于调试）
                cv2.circle(annotated_image, (int(left_knee[0]), int(left_knee[1])), 5, (0, 255, 0), -1)
                cv2.circle(annotated_image, (int(right_knee[0]), int(right_knee[1])), 5, (0, 255, 0), -1)

                # 对每对匹配的offground和safebelt框进行处理
                for offground_box, safebelt_box in matched_pairs:
                    x1, y1, x2, y2 = safebelt_box
                    # 计算安全带框的中心点y坐标
                    box_center_y = y2

                    # 判断安全带位置是否在膝盖以下
                    if box_center_y > knee_y:
                        # 安全带佩戴不规范 - 在膝盖以下
                        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(knee_y)), (0, 0, 255), 5)
                        cv2.putText(annotated_image, "Unsafe Belt", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    else:
                        # 安全带佩戴规范 - 在膝盖以上
                        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(knee_y)), (0, 255, 0), 5)
                        cv2.putText(annotated_image, "Safe Belt", (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # 如果没有足够的关键点可见度，则默认标记所有匹配的安全带框
                for offground_box, safebelt_box in matched_pairs:
                    x1, y1, x2, y2 = safebelt_box
                    # 默认标记为需要检查
                    cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 165, 0), 2)
                    cv2.putText(annotated_image, "Belt Detected", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 165, 0), 2)

# 转换为RGB格式
im_rgb = Image.fromarray(annotated_image[..., ::-1])  # RGB-order PIL image

# Show results to screen (in supported environments)
im_rgb.show()

# 修复保存图像的代码，移除错误的参数名filename
im_rgb.save(name + "_results_with_belt_check.jpg")