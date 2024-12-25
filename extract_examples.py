import os
import numpy as np
from PIL import Image
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

def extract_example_images():
    # 创建保存图片的目录
    save_dir = Path(r'D:\2024\SDGB\medmnist\project\static\images')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据集
    data_path = r'C:\Users\Administrator\.medmnist\organmnist3d.npz'  # 替换为你的数据集路径
    data = np.load(data_path)
    
    # 获取训练数据和标签
    images = data['train_images']
    labels = data['train_labels']
    
    # 获取所有唯一的类别
    unique_labels = np.unique(labels)
    
    # 确保从每个类别至少选择一张图片
    selected_indices = []
    for label in unique_labels:
        # 找到该类别的所有图片索引
        label_indices = np.where(labels == label)[0]
        # 随机选择一张
        selected_idx = np.random.choice(label_indices)
        selected_indices.append(selected_idx)
    
    # 如果还需要更多图片，随机选择
    while len(selected_indices) < 11:
        idx = np.random.randint(0, len(images))
        if idx not in selected_indices:
            selected_indices.append(idx)
    
    # 保存选中的图片
    for i, idx in enumerate(selected_indices):
        img = images[idx]
        label = labels[idx][0]
        
        # 选择中间切片作为2D表示
        middle_slice = img[img.shape[0]//2]
        
        # 归一化到0-255范围
        img_normalized = ((middle_slice - middle_slice.min()) * 255 / 
                         (middle_slice.max() - middle_slice.min())).astype(np.uint8)
        
        # 创建RGB图像
        img_rgb = np.stack([img_normalized] * 3, axis=-1)
        
        # 保存图片
        img_pil = Image.fromarray(img_rgb)
        img_pil.save(save_dir / f'example_{i}_label_{label}.png')
        
        print(f'已保存图片 {i+1}/11，类别: {label}')

if __name__ == '__main__':
    extract_example_images()