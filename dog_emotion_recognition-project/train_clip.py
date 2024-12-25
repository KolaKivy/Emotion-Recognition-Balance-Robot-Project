import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import random
from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np
import clip  # 导入CLIP模型

# 定义高斯噪声添加类（保持原有数据增强）
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        img = np.array(img)
        noise = np.random.normal(self.mean, self.std, img.shape)
        img = img + noise
        img = np.clip(img, 0, 255)
        return Image.fromarray(img.astype(np.uint8))

# 自定义Triplet Dataset，支持对比学习的三元组（Anchor, Positive, Negative）
class TripletDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.labels = [sample[1] for sample in self.dataset.samples]
        
    def __getitem__(self, index):
        # 获取anchor图像和标签
        anchor, anchor_label = self.dataset[index]
        
        # 获取positive（同标签的样本）
        positive_index = random.choice([i for i, label in enumerate(self.labels) if label == anchor_label and i != index])
        positive, _ = self.dataset[positive_index]
        
        # 获取negative（不同标签的样本）
        negative_index = random.choice([i for i, label in enumerate(self.labels) if label != anchor_label])
        negative, _ = self.dataset[negative_index]
        
        return anchor, positive, negative, anchor_label

    def __len__(self):
        return len(self.dataset)

# 对比学习模型：基于ResNet18并加上嵌入层
class ResNetContrastiveWithSemanticLoss(nn.Module):
    def __init__(self, num_classes, clip_model, text_embeddings_dim=512, visual_embeddings_dim=512):
        super(ResNetContrastiveWithSemanticLoss, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        num_ftrs = self.base_model.fc.in_features
        self.feature_extractor = nn.Sequential(*list(self.base_model.children())[:-1])  # 移除全连接层
        self.embedding_layer = nn.Linear(num_ftrs, visual_embeddings_dim)  # 嵌入层，用于对比学习
        self.classifier = nn.Linear(visual_embeddings_dim, num_classes)   # 分类头
        
        # CLIP模型
        self.clip_model = clip_model
        self.text_embeddings_dim = text_embeddings_dim
        self.visual_embeddings_dim = visual_embeddings_dim
        self.text_projection = nn.Linear(512, self.text_embeddings_dim)  # 将CLIP文本特征投影到目标维度

    def forward(self, image, text=None):
        # 提取图像特征
        image_features = self.feature_extractor(image)
        image_features = image_features.view(image_features.size(0), -1)  # 展平
        visual_embeddings = self.embedding_layer(image_features)  # 嵌入
        
        if text is not None:
            # 使用CLIP模型提取文本特征
            text_features = self.clip_model.encode_text(clip.tokenize(text).to(device))
            text_embeddings = self.text_projection(text_features)  # 将文本特征投影到目标维度

            # 计算语义损失：图像特征与文本特征的余弦相似度
            semantic_loss = self.compute_semantic_loss(visual_embeddings, text_embeddings)
            return visual_embeddings, semantic_loss
        
        logits = self.classifier(visual_embeddings)  # 分类输出
        return visual_embeddings, logits
    
    def compute_semantic_loss(self, visual_embeddings, text_embeddings):
        # 计算视觉和文本特征的余弦相似度作为语义损失
        cos_sim = torch.nn.functional.cosine_similarity(visual_embeddings, text_embeddings, dim=-1)
        semantic_loss = 1 - cos_sim  # 语义损失，目标是最大化余弦相似度
        return semantic_loss.mean()

def contrastive_loss(anchor_emb, positive_emb, negative_emb, margin=1.0):
    pos_distance = torch.norm(anchor_emb - positive_emb, p=2, dim=1)  # anchor与positive之间的欧几里得距离
    neg_distance = torch.norm(anchor_emb - negative_emb, p=2, dim=1)  # anchor与negative之间的欧几里得距离
    loss = torch.clamp(pos_distance - neg_distance + margin, min=0.0)  # Triplet loss公式
    return loss.mean()


# 训练函数
def train(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    epoch_accuracy_record = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for anchor, positive, negative, labels in train_loader:
            # 将数据迁移到GPU
            anchor, positive, negative, labels = anchor.to(device), positive.to(device), negative.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            anchor_emb, semantic_loss = model(anchor)  # 图像嵌入和语义损失
            positive_emb, _ = model(positive)  # 对正样本进行计算
            negative_emb, _ = model(negative)  # 对负样本进行计算
            
            # 计算对比损失
            loss_class = criterion(anchor_emb, labels)  # 分类损失
            contrastive_loss_value = contrastive_loss(anchor_emb, positive_emb, negative_emb)  # 对比损失
            
            # 总损失：对比损失 + 分类损失 + 语义损失
            loss = loss_class + 0.05 * contrastive_loss_value + 0.1 * semantic_loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 累积损失和正确预测数量
            running_loss += loss.item()
            _, preds = torch.max(anchor_emb, 1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)
        
        # 打印每个epoch的训练情况
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_preds / total_preds
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
        
        # 测试
        test_accuracy = test(model, test_loader)
        
        # 保存最好的模型
        if test_accuracy > epoch_accuracy_record:
            epoch_accuracy_record = test_accuracy
            save_model(model, './emotion_model_contrastive_with_semantic.pth')
            print("saved a checkpoint with accuracy: {}".format(epoch_accuracy_record))

# 测试函数
def test(model, test_loader):
    model.eval()  # 设置为评估模式
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, logits = model(inputs)
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

# 保存模型函数
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model weights saved to {path}")

# 定义训练和测试的transform
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    AddGaussianNoise(mean=0.0, std=0.1),  # 添加高斯噪声
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 加载数据集
train_dataset_dir = './dataset/train'
test_dataset_dir = './dataset/test'

train_dataset = datasets.ImageFolder(root=train_dataset_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(root=test_dataset_dir, transform=test_transform)

# 使用TripletDataset
train_triplet_dataset = TripletDataset(train_dataset)

# 创建DataLoader
batch_size = 32
train_loader = DataLoader(train_triplet_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 加载模型
model = ResNetContrastive(num_classes=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义优化器和损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train(model, train_loader, criterion, optimizer, num_epochs=50)

# 测试模型
test(model, test_loader)