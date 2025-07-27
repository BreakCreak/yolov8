from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2

# Load a pretrained YOLO11n model
model = YOLO("./BeltDetection.pt")

# Load pose estimation model
pose_model = YOLO("yolov8n-pose.pt")

# Define path to the image file
source = "inference.jpg"

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

# 如果有关键点数据，则进行安全带佩戴规范检测
if keypoints is not None:
    for i, kp in enumerate(keypoints):
        # 每个人的关键点
        if kp.shape[0] >= 17:  # 确保关键点数量足够
            # 获取膝盖关键点 (通常索引为13和14)
            left_knee = kp[13]
            right_knee = kp[14]
            
            # 检查关键点的可见性
            if left_knee[2] > 0.5 and right_knee[2] > 0.5:
                # 计算膝盖的平均y坐标
                knee_y = (left_knee[1] + right_knee[1]) / 2
                
                # 在图像上绘制膝盖位置（用于调试）
                cv2.circle(annotated_image, (int(left_knee[0]), int(left_knee[1])), 5, (0, 255, 0), -1)
                cv2.circle(annotated_image, (int(right_knee[0]), int(right_knee[1])), 5, (0, 255, 0), -1)
                
                # 查找"offground"类别的安全带框 (类别索引为1)
                if belt_results.boxes is not None:
                    for box in belt_results.boxes:
                        # 获取框的类别
                        cls = int(box.cls.item())
                        # 如果是offground类别
                        if cls == 1:
                            # 获取框的坐标
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            # 计算框的中心点y坐标
                            box_center_y = (y1 + y2) / 2
                            
                            # 判断安全带位置是否在膝盖以下
                            if box_center_y > knee_y:
                                # 安全带佩戴不规范 - 在膝盖以下
                                cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                                cv2.putText(annotated_image, "Unsafe Belt", (int(x1), int(y1)-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            else:
                                # 安全带佩戴规范 - 在膝盖以上
                                cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                cv2.putText(annotated_image, "Safe Belt", (int(x1), int(y1)-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 转换为RGB格式
im_rgb = Image.fromarray(annotated_image[..., ::-1])  # RGB-order PIL image

# Show results to screen (in supported environments)
im_rgb.show()

# 修复保存图像的代码，移除错误的参数名filename
im_rgb.save("results_with_belt_check.jpg")