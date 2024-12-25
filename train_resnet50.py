import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

# 导入自定义模块
from data_loader import load_data
from pretrain_loader import generate_model
from utils import Logger

# 设置环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 设置设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 初始化tensorboard和logger
summaryWriter = SummaryWriter("./logs/")
logger = Logger(log_dir="./logs/")

# 加载数据
train_loader, val_loader, test_loader = load_data()

# 加载模型
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

# 将模型移动到指定设备
model = model.to(device)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = ExponentialLR(optimizer, gamma=0.99)

# 训练参数设置
num_epochs = 800
total_step = len(train_loader)

# 训练循环
best_val_acc = 0.0  # 记录最佳验证准确率

for epoch in range(num_epochs):
    start = time.time()
    per_epoch_loss = 0
    num_correct = 0
    val_num_correct = 0
    
    # 训练阶段
    model.train()
    with torch.enable_grad():
        for x, label in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            x = x.to(device)
            label = label.to(device).long()
            label = torch.squeeze(label)
            
            logits = model(x)
            loss = criterion(logits, label)
            per_epoch_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pred = logits.argmax(dim=1)
            num_correct += torch.eq(pred, label).sum().float().item()
        
        train_acc = num_correct/len(train_loader.dataset)
        train_loss = per_epoch_loss/total_step
        print(f"Train Epoch: {epoch}\t Loss: {train_loss:.6f}\t Acc: {train_acc:.6f}")
        
        # 记录训练指标
        summaryWriter.add_scalars('loss', {"train_loss": train_loss}, epoch)
        summaryWriter.add_scalars('acc', {"train_acc": train_acc}, epoch)
        logger.log(epoch=epoch, loss=train_loss, acc=train_acc, phase='train')
    
    # 验证阶段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, label in tqdm(val_loader, desc=f"Validating Epoch {epoch}"):
            x = x.to(device)
            label = label.to(device).long()
            label = torch.squeeze(label)
            
            logits = model(x)
            loss = criterion(logits, label)
            val_loss += loss.item()
            
            pred = logits.argmax(dim=1)
            val_num_correct += torch.eq(pred, label).sum().float().item()
        
        current_val_acc = val_num_correct/len(val_loader.dataset)
        current_val_loss = val_loss/len(val_loader)
        print(f"Val Epoch: {epoch}\t Loss: {current_val_loss:.6f}\t Acc: {current_val_acc:.6f}")
        
        # 记录验证指标
        # 例如添加学习率的变化
        summaryWriter.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch)
        summaryWriter.add_scalars('loss', {"val_loss": current_val_loss}, epoch)
        summaryWriter.add_scalars('acc', {"val_acc": current_val_acc}, epoch)
        summaryWriter.add_scalars('time', {"time": time.time() - start}, epoch)
        logger.log(epoch=epoch, loss=current_val_loss, acc=current_val_acc, phase='val')
        
        # 保存最佳模型
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            save_path = os.path.join('./checkpoints', f'best_model.pth')
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_loss': current_val_loss,
            }, save_path)
            print(f'保存最佳模型，验证准确率: {best_val_acc:.6f}')
    
    scheduler.step()

# 保存最终模型
final_save_path = os.path.join('./checkpoints', f'final_model.pth')
torch.save({
    'epoch': num_epochs-1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_acc': current_val_acc,
    'val_loss': current_val_loss,
}, final_save_path)
print(f'保存最终模型到: {final_save_path}')

summaryWriter.close()