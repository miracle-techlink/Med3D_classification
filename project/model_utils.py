import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import os
from models.resnet import resnet50

def load_model():
    """
    加载和初始化模型
    返回: 初始化并加载了权重的模型
    """
    try:
        # 创建模型实例
        model = resnet50(
            sample_input_D=28,
            sample_input_H=28,
            sample_input_W=28,
            num_seg_classes=11,
            shortcut_type='B',
            no_cuda=False
        )
        
        # 加载检查点
        checkpoint_path = os.path.join('models', 'final_model.pth')
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # 处理状态字典
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # 移除'module.'前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        # 加载状态字典
        model.load_state_dict(new_state_dict, strict=False)
        print("模型加载成功！")
        
        # 验证模型加载
        if verify_model_loading(model):
            print("模型参数验证成功！")
        else:
            print("警告：模型参数可能未完全加载！")
            
    except Exception as e:
        print(f"模型加载失败：{str(e)}")
        raise e
    
    # 设置为评估模式
    model.eval()
    return model

def verify_model_loading(model):
    """
    验证模型参数是否正确加载
    参数:
        model: PyTorch模型实例
    返回:
        bool: 是否成功加载
    """
    loaded_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if param.requires_grad and torch.sum(torch.abs(param)) > 0:
            loaded_params += 1
    
    # 如果超过90%的参数已加载，认为加载成功
    return (loaded_params / total_params) > 0.9

def preprocess_image(image_bytes):
    """
    预处理输入图片
    参数:
        image_bytes: 图片的字节数据
    返回:
        tensor: 预处理后的图片张量
    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
        transforms.Resize((28, 28)),  # 调整大小为模型输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    try:
        # 打开图片
        image = Image.open(io.BytesIO(image_bytes))
        
        # 应用变换
        image = transform(image)
        
        # 添加维度
        image = image.unsqueeze(0)  # 添加batch维度
        image = image.unsqueeze(0)  # 添加深度维度
        image = image.repeat(1, 1, 28, 1, 1)  # 复制到所需深度
        
        return image
        
    except Exception as e:
        print(f"图片预处理失败：{str(e)}")
        raise e

def get_prediction(model, image_tensor):
    """
    使用模型进行预测
    参数:
        model: PyTorch模型实例
        image_tensor: 预处理后的图片张量
    返回:
        int: 预测的类别
    """
    try:
        with torch.no_grad():
            # 进行预测
            outputs = model(image_tensor)
            
            # 处理输出
            outputs = torch.nn.functional.adaptive_avg_pool3d(outputs, (1, 1, 1))
            outputs = outputs.squeeze()
            
            # 获取预测类别
            _, predicted = torch.max(outputs, 0)
            return predicted.item()
            
    except Exception as e:
        print(f"预测过程发生错误：{str(e)}")
        raise e

def get_class_name(class_id):
    """
    获取类别名称
    参数:
        class_id: 类别ID
    返回:
        str: 类别名称
    """
    class_names = {
        0: "正常",
        1: "肺炎",
        2: "COVID-19",
        3: "肺结核",
        4: "肺癌",
        5: "气胸",
        6: "胸腔积液",
        7: "肺气肿",
        8: "支气管扩张",
        9: "间质性肺疾病",
        10: "其他肺部疾病"
    }
    return class_names.get(class_id, "未知类别")

def predict_image(model, image_bytes):
    """
    完整的图片预测流程
    参数:
        model: PyTorch模型实例
        image_bytes: 图片的字节数据
    返回:
        tuple: (预测类别ID, 预测类别名称)
    """
    try:
        # 预处理图片
        image_tensor = preprocess_image(image_bytes)
        
        # 获取预测结果
        predicted_class = get_prediction(model, image_tensor)
        
        # 获取类别名称
        class_name = get_class_name(predicted_class)
        
        return predicted_class, class_name
        
    except Exception as e:
        print(f"预测过程发生错误：{str(e)}")
        raise e