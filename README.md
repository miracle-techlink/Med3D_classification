## 简介
这个仓库是用来储存 medmnist 3D医学影像多分类的代码
其中checkpoints是用来储存最好的模型结果的，将最好的模型放在project中做实时分类。
模型的架构是Resnet50

## 硬件配置
Nvidia 3060Ti
Cuda 12.5 Cudnn 8.4.1

## requirements
numpy==1.21.2
torch==1.13.0
torchvision==0.14.0
tqdm==4.62.3
Pillow==8.4.0
matplotlib==3.4.3
flask==2.0.2
tensorboard==2.7.0
medmnist==2.2.1
protobuf==3.20.3

## 运行
启动环境：(配置自己的gpu环境）
conda create -n pytorch_gpu python=3.10
下载gpu驱动与对应pytorch的包
conda activate pytorch_gpu
pip install -r requirements

**训练模型:**
python train_resnet50.py

**测试模型：**
python test.py

**前端展示：(位于project文件夹下,使用flask框架进行展示)**
将训练好的best_model.pth（checkpoints）文件夹下放在 /project/models 文件夹下
cd project
python app.py

## 数据集
数据集为[Medmnist]

## 模型
模型为[ResNet50]

## 参考
[ResNet50](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)

## License
MIT
