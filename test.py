import numpy as np
import torch
from medmnist import OrganMNIST3D
from torch.utils.data import DataLoader, Subset
import random
from sklearn.metrics import confusion_matrix, recall_score, roc_curve, auc  # 导入所需的库
import seaborn as sns
import matplotlib.pyplot as plt

from pretrain_loader import generate_model

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_model_for_prediction(model_path):
    """
    加载训练好的模型用于预测
    """
    # 创建模型实例
    model = generate_model(
        model_type='resnet',
        model_depth=50,
        input_W=224,
        input_H=224,
        input_D=224,
        resnet_shortcut='B',
        no_cuda=False,
        gpu_id=[0],
        pretrain_path='pretrain/resnet_50.pth',
        nb_class=11
    )
    
    # 加载保存的模型
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def predict(model, input_tensor):
    """
    使用模型进行预测
    """
    with torch.no_grad():
        input_tensor = input_tensor.float().to(device)  # 确保使用float类型
        output = model(input_tensor)
        pred = output.argmax(dim=1)
        return pred

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
# 加载数据集
data = OrganMNIST3D(split='test', download=True)

# 获取数据集大小并打印
total_samples = len(data)
print(f"数据集总样本数: {total_samples}")

# 选择较小的值作为采样数量
sample_size = min(1000, total_samples)
print(f"实际采样数量: {sample_size}")

# 随机抽取样本的索引
selected_indices = random.sample(range(total_samples), sample_size)

# 创建子数据集
subset = Subset(data, selected_indices)
loader = DataLoader(subset, batch_size=32, shuffle=False)

# 加载预训练模型
model = load_model_for_prediction('./checkpoints/best_model.pth')
model.eval()

# 用于存储预测结果和真实标签
predictions = []
true_labels = []

# 进行预测
with torch.no_grad():
    for inputs, labels in loader:
        # 确保输入数据为float类型
        inputs = inputs.float().to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.squeeze().numpy())

# 转换为numpy数组
predictions = np.array(predictions)
true_labels = np.array(true_labels)

# 打印预测结果
print("\n预测样本分析:")
print("预测类别（前20个）:", predictions[:20])
print("真实类别（前20个）:", true_labels[:20])

# 计算准确率
accuracy = np.mean(predictions == true_labels)
print(f"\n总体准确率: {accuracy:.4f}")

# 计算召回率
recall = recall_score(true_labels, predictions, average='weighted')  # 计算召回率
print(f"总体召回率: {recall:.4f}")

# 计算ROC和AUC
fpr, tpr, thresholds = roc_curve(true_labels, predictions, pos_label=1)  # 计算假阳性率和真阳性率
roc_auc = auc(fpr, tpr)  # 计算AUC
print(f"ROC AUC: {roc_auc:.4f}")

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')  # 绘制随机猜测的对角线
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('接收者操作特征曲线 (ROC)')
plt.legend(loc="lower right")
plt.show()

# 绘制混淆矩阵
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('混淆矩阵')
plt.xlabel('预测类别')
plt.ylabel('真实类别')
plt.show()

# 保存结果
results = {
    'predictions': predictions,
    'true_labels': true_labels,
    'accuracy': accuracy,
    'recall': recall,
    'roc_auc': roc_auc,
    'confusion_matrix': cm
}

# 可以选择将结果保存到文件
np.save('test_results.npy', results)
print("\n结果已保存到 test_results.npy")