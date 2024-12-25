import torch
import cv2
from torchvision import transforms, models
from PIL import Image
import numpy as np

# 加载已保存的模型
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()  # 设置为评估模式
    print(f"Model weights loaded from {path}")
    return model

# 预测单张图片（改为支持实时视频流）
def predict_image(model, frame, threshold=0.7):
    # 将OpenCV图像转换为PIL图像
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # 预处理图像（与训练时相同的预处理）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(1)  # 添加一个批次维度

    # 将图像移动到 GPU（如果可用）
    image = image.to(device)
    
    # 使用模型进行预测
    with torch.no_grad():  # 不需要计算梯度
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)  # 获取每个类的概率
        max_prob, predicted_class = torch.max(probs, 1)  # 预测类别以及其最大概率

    # 类别映射
    class_names = ['angry', 'happy', 'relaxed', 'sad']
    predicted_class_name = class_names[predicted_class.item()]
    class_mapping = {'angry': 1, 'happy': 2, 'relaxed': 3, 'sad': 4}  # 类别对应的值
    predicted_class_value = class_mapping[predicted_class_name]
    prob = max_prob.item()

    # 如果置信度大于阈值，则输出预测结果
    if prob > threshold:
        print(f"Predicted emotion: {predicted_class_name} {predicted_class_value} with probability {prob:.4f}")
        return predicted_class_value, prob
    else:
        print("Confidence too low, no prediction")
        return None, prob

# 使用 ResNet-18 进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)  # 4 类别：angry, happy, relaxed, sad
model.to(device)

# 加载保存的模型权重
model = load_model(model, './emotion_model.pth')

# 打开摄像头进行实时视频流识别
cap = cv2.VideoCapture(0)  # 0 表示默认摄像头
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

threshold = 0.998  # 设置置信度阈值

while True:
    ret, frame = cap.read()  # 读取每一帧
    if not ret:
        print("Failed to grab frame")
        break

    # 调用模型进行预测
    predicted_class, prob = predict_image(model, frame, threshold)

    # 在图像上显示预测结果
    if predicted_class:
        cv2.putText(frame, f"Emotion: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No confident prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示图像
    cv2.imshow("Emotion Recognition", frame)

    # 按 'q' 键退出//
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # 释放摄像头
cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
