import torch
import torch.nn as nn
from models import resnet

def generate_model(model_type='resnet',   # 模型类型
                   model_depth=50,        # 模型深度
                   input_W=224,           # 输入宽度
                   input_H=224,           # 输入高度
                   input_D=224,           # 输入深度
                   resnet_shortcut='B',   # ResNet 短路连接类型
                   no_cuda=False,         # 是否不使用 GPU
                   gpu_id=[0],            # GPU 设备ID
                   pretrain_path = 'pretrain/resnet_50.pth',   # 预训练模型路径
                   nb_class=11):           # 输出类别数
    
    # 验证模型类型
    assert model_type in ['resnet']
    
    # 验证网络深度
    if model_type == 'resnet':
        assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    
    # 根据深度选择对应的模型结构
    model_mapping = {
        10: (resnet.resnet10, 256),
        18: (resnet.resnet18, 512),
        34: (resnet.resnet34, 512),
        50: (resnet.resnet50, 2048),
        101: (resnet.resnet101, 2048),
        152: (resnet.resnet152, 2048),
        200: (resnet.resnet200, 2048)
    }
    
    model_fn, fc_input = model_mapping[model_depth]
    
    # 创建模型实例
    model = model_fn(
        sample_input_W=input_W,
        sample_input_H=input_H,
        sample_input_D=input_D,
        shortcut_type=resnet_shortcut,
        no_cuda=no_cuda,
        num_seg_classes=1
    )
    
    # 修改最后的分类层
    model.conv_seg = nn.Sequential(
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten(),
        nn.Linear(in_features=fc_input, out_features=nb_class, bias=True)
    )
    
    # GPU设置
    if not no_cuda:
        if len(gpu_id) > 1:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=gpu_id)
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id[0])
            # model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)
    
    # 加载预训练权重
    net_dict = model.state_dict()
    print('loading pretrained model {}'.format(pretrain_path))
    pretrain = torch.load(pretrain_path)
    pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
    net_dict.update(pretrain_dict)
    model.load_state_dict(net_dict)
    
    print("-------- pre-train model load successfully --------")
    return model