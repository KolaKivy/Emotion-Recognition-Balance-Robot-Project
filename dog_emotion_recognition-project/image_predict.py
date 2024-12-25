import torch
from torchvision import transforms, models
from PIL import Image

# 加载已保存的模型
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()  # 设置为评估模式
    print(f"Model weights loaded from {path}")
    return model

# 预测单张图片
def predict_image(model, image_path):
    # 加载图像
    image = Image.open(image_path)
    
    # 预处理图像（与训练时相同的预处理）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # 添加一个批次维度
    
    # 将图像移动到 GPU（如果可用）
    image = image.to(device)
    
    # 使用模型进行预测
    with torch.no_grad():  # 不需要计算梯度
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)  # 预测类别
    
    # 类别映射
    class_names = ['angry', 'happy', 'relaxed', 'sad']
    predicted_class_name = class_names[predicted_class.item()]
    print(f"Predicted emotion: {predicted_class_name}")

# 使用 ResNet-18 进行训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)  # 4 类别：angry, happy, relaxed, sad
model.to(device)

# 加载保存的模型权重
model = load_model(model, './emotion_model.pth')

# 预测某个路径下的图片
image_path = './test/angry/1.jpg'  # 替换为你的图片路径
predict_image(model, image_path)
