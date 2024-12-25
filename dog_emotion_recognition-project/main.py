import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# 定义数据转换（包括图像增强和标准化）
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        # 将图片转换为numpy数组
        img = np.array(img)
        # 添加高斯噪声
        noise = np.random.normal(self.mean, self.std, img.shape)
        img = img + noise
        img = np.clip(img, 0, 255)  # 确保像素值在有效范围内
        return Image.fromarray(img.astype(np.uint8))

# 训练时的transform（加入了高斯噪声）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    AddGaussianNoise(mean=0.0, std=0.1),  # 添加高斯噪声
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像标准化
])

# 测试时的transform（仅进行标准化）
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 图像标准化
])
# 加载数据集
train_dataset_dir = './dataset/train'  # 数据集路径
test_dataset_dir = './dataset/test'

train_dataset = datasets.ImageFolder(root=train_dataset_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dataset_dir, transform=test_transform)

# 划分训练集和测试集
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建 DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 加载预训练的 ResNet-18 模型
model = models.resnet18(pretrained=True)

# 修改最后一层，使其适应四个类别
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)  # 4 类别：angry, happy, relaxed, sad

# 将模型迁移到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")

# 定义测试函数
def test(model, test_loader):
    model.eval()  # 设置为评估模式
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # 不需要计算梯度
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 预测结果
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())  # 收集所有预测结果
            all_labels.extend(labels.cpu().numpy())  # 收集所有真实标签
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

# 定义训练函数
def train(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()  # 设置为训练模式
    epoch_accuracy_record = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for inputs, labels in train_loader:
            # 将数据迁移到 GPU
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 更新权重
            optimizer.step()
            
            # 累积损失和正确预测数量
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)
        
        # 打印每个epoch的训练情况
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_preds / total_preds       
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        test_accuracy = test(model, test_loader)
        model.train()  # 设置为训练模式
        if test_accuracy > epoch_accuracy_record:
            epoch_accuracy_record = test_accuracy
            save_model(model, './emotion_model_origin.pth')
            print("saved a checkpoint with accuracy: {}".format(epoch_accuracy_record))
    

# # 训练模型
# train(model, train_loader, criterion, optimizer, num_epochs=10)


# 训练模型
train(model, train_loader, criterion, optimizer, num_epochs=50)

# save_model(model, './emotion_model.pth')

# # 测试模型
# test(model, test_loader)








