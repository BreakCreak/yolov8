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
source =name+".jpg"

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


# 对每个offground区域进行处理
if offground_boxes:
    for idx, offground_box in enumerate(offground_boxes):
        x1, y1, x2, y2 = [int(coord) for coord in offground_box]
        
        # 裁剪offground区域
        cropped_image = original_image[y1:y2, x1:x2]
        
        # 对裁剪后的图像进行安全带检测
        cropped_results = model(cropped_image)
        cropped_belt_boxes = []
        
        if cropped_results[0].boxes is not None:
            for box in cropped_results[0].boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())  # 获取置信度
                # 检查是否检测到安全带(类别3)并且置信度大于阈值
                if cls == 3 and conf > 0.5:  # 添加置信度阈值判断
                    cropped_belt_boxes.append(box.xyxy[0].tolist())

        # 如果在裁剪图像中未检测到安全带
        if not cropped_belt_boxes:
            # 在原图上标记未佩戴安全带
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated_image, "No Belt Detected", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            # 如果检测到安全带，则进行关键点检测判断是否规范佩戴
            # 首先需要找到与当前offground区域匹配的完整图像中的关键点
            if keypoints is not None:
                # 获取姿态估计中的人体边界框用于匹配
                if pose_result.boxes is not None:
                    for i, (kp, person_box) in enumerate(zip(keypoints, pose_result.boxes.xyxy)):
                        # 检查该人体是否在offground区域内
                        person_box_coords = person_box.tolist()
                        iou = calculate_iou(person_box_coords, offground_box)

                        # 如果人体与offground区域有足够重叠
                        if iou > 0.3 and kp.shape[0] >= 17:  # 确保关键点数量足够
                            # 获取膝盖关键点 (通常索引为13和14)
                            left_knee = kp[13]
                            right_knee = kp[14]

                            # 检查关键点的可见性
                            if left_knee[2] > 0.5 and right_knee[2] > 0.5:
                                # 计算膝盖的最高点y坐标（较小的y值）
                                knee_y = min(left_knee[1], right_knee[1])

                                # 在图像上绘制膝盖位置（用于调试）
                                cv2.circle(annotated_image, (int(left_knee[0]), int(left_knee[1])), 5, (0, 255, 0), -1)
                                cv2.circle(annotated_image, (int(right_knee[0]), int(right_knee[1])), 5, (0, 255, 0), -1)

                                # 对裁剪区域中的每个安全带检测框进行处理
                                for belt_box in cropped_belt_boxes:
                                    # 将裁剪图像中的坐标转换为原图坐标
                                    belt_x1, belt_y1, belt_x2, belt_y2 = belt_box
                                    global_belt_x1 = x1 + int(belt_x1)
                                    global_belt_y1 = y1 + int(belt_y1)
                                    global_belt_x2 = x1 + int(belt_x2)
                                    global_belt_y2 = y1 + int(belt_y2)

                                    # 使用安全带框的最低点y坐标
                                    belt_bottom_y = global_belt_y2

                                    # 判断安全带位置是否在膝盖以下（最低点的安全带位置和最高点的膝盖位置比较）
                                    if belt_bottom_y > knee_y:
                                        # 安全带佩戴不规范 - 安全带位置低于膝盖位置
                                        cv2.rectangle(annotated_image, (global_belt_x1, global_belt_y1), (global_belt_x2, global_belt_y2), (0, 0, 255), 2)
                                        cv2.putText(annotated_image, "Unsafe Belt", (global_belt_x1, global_belt_y1-10),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                                    else:
                                        # 安全带佩戴规范 - 安全带位置高于膝盖位置
                                        cv2.rectangle(annotated_image, (global_belt_x1, global_belt_y1), (global_belt_x2, global_belt_y2), (0, 255, 0), 2)
                                        cv2.putText(annotated_image, "Safe Belt", (global_belt_x1, global_belt_y1-10),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            else:
                                # 如果没有足够的关键点可见度，则默认标记检测到的安全带框
                                for belt_box in cropped_belt_boxes:
                                    belt_x1, belt_y1, belt_x2, belt_y2 = belt_box
                                    global_belt_x1 = x1 + int(belt_x1)
                                    global_belt_y1 = y1 + int(belt_y1)
                                    global_belt_x2 = x1 + int(belt_x2)
                                    global_belt_y2 = y1 + int(belt_y2)
                                    # 默认标记为需要检查
                                    cv2.rectangle(annotated_image, (global_belt_x1, global_belt_y1), (global_belt_x2, global_belt_y2), (255, 165, 0), 2)
                                    cv2.putText(annotated_image, "Belt Detected", (global_belt_x1, global_belt_y1-10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 165, 0), 2)
                else:
                    # 如果姿态估计没有检测框，仍然处理安全带框
                    for belt_box in cropped_belt_boxes:
                        belt_x1, belt_y1, belt_x2, belt_y2 = belt_box
                        global_belt_x1 = x1 + int(belt_x1)
                        global_belt_y1 = y1 + int(belt_y1)
                        global_belt_x2 = x1 + int(belt_x2)
                        global_belt_y2 = y1 + int(belt_y2)
                        # 默认标记为需要检查
                        cv2.rectangle(annotated_image, (global_belt_x1, global_belt_y1), (global_belt_x2, global_belt_y2), (255, 165, 0), 2)
                        cv2.putText(annotated_image, "Belt Detected", (global_belt_x1, global_belt_y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 165, 0), 2)

# 转换为RGB格式
im_rgb = Image.fromarray(annotated_image[..., ::-1])  # RGB-order PIL image

# Show results to screen (in supported environments)
im_rgb.show()

# 修复保存图像的代码，移除错误的参数名filename
im_rgb.save(name+"_results_with_belt_check.jpg")