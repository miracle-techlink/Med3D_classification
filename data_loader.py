# data_loader.py
import os
import numpy as np
import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader

# 设置批次大小
batch_size = 64

# 数据预处理
class Transform3D:

    def __init__(self, mul=None):
        self.mul = mul

    def __call__(self, voxel):
   
        if self.mul == '0.5':
            voxel = voxel * 0.5
        elif self.mul == 'random':
            voxel = voxel * np.random.uniform()
        
        return voxel.astype(np.float32)
    
def load_data(batch_size= batch_size, data_flag='organmnist3d', download=True):
    # 数据集配置
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # 数据集加载
    train_dataset = DataClass(split='train', transform=Transform3D(mul='random'), download=download)
    val_dataset = DataClass(split='val', transform=Transform3D(mul='0.5'), download=download)
    test_dataset = DataClass(split='test', transform=Transform3D(mul='0.5'), download=download)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader