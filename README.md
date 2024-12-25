## 运行
启动环境：
conda activate pytorch_gpu

训练模型：
python train_resnet50.py

测试模型：
python test.py

前端展示：(位于project文件夹下,使用flask框架进行展示)
cd project
python app.py

## 数据集
数据集为[Medmnist]

## 模型
模型为[ResNet50]

## 参考
[Medmnist](https://github.com/Beckschen/MedMNIST)

[ResNet50](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)

## License
MIT