import os
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir="./logs/"):
        """
        初始化Logger
        Args:
            log_dir: 日志保存目录
        """
        # 创建日志目录
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化tensorboard的SummaryWriter
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # 创建日志文件名，使用时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')
        
        # 记录初始化信息
        print(f"日志将保存到: {self.log_file}")
        
    def log_training(self, epoch, loss, accuracy):
        """
        记录训练指标
        Args:
            epoch: 当前轮次
            loss: 训练损失
            accuracy: 训练准确率
        """
        self.writer.add_scalar('Loss/train', loss, epoch)
        self.writer.add_scalar('Accuracy/train', accuracy, epoch)
        
        log_dict = {
            'epoch': epoch,
            'phase': 'train',
            'loss': float(loss),
            'accuracy': float(accuracy),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self._write_log(log_dict)

    def log_validation(self, epoch, loss, accuracy):
        """
        记录验证指标
        Args:
            epoch: 当前轮次
            loss: 验证损失
            accuracy: 验证准确率
        """
        self.writer.add_scalar('Loss/val', loss, epoch)
        self.writer.add_scalar('Accuracy/val', accuracy, epoch)
        
        log_dict = {
            'epoch': epoch,
            'phase': 'validation',
            'loss': float(loss),
            'accuracy': float(accuracy),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self._write_log(log_dict)

    def log_time(self, epoch, time_taken):
        """
        记录每轮训练时间
        Args:
            epoch: 当前轮次
            time_taken: 训练耗时
        """
        self.writer.add_scalar('Time/epoch', time_taken, epoch)
        
        log_dict = {
            'epoch': epoch,
            'phase': 'time',
            'time_taken': float(time_taken),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self._write_log(log_dict)

    def log(self, epoch, loss, acc, phase='train'):
        """
        统一的日志记录接口
        Args:
            epoch: 当前轮次
            loss: 损失值
            acc: 准确率
            phase: 阶段（train/val）
        """
        if phase == 'train':
            self.log_training(epoch, loss, acc)
        else:
            self.log_validation(epoch, loss, acc)

    def _write_log(self, log_dict):
        """
        将日志写入文件
        Args:
            log_dict: 日志字典
        """
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_dict) + '\n')

    def close(self):
        """
        关闭Logger
        """
        self.writer.close()
        print(f"Logger已关闭，日志保存在: {self.log_file}")

    def get_log_file(self):
        """
        获取日志文件路径
        Returns:
            str: 日志文件路径
        """
        return self.log_file

def read_logs(log_file):
    """
    读取日志文件
    Args:
        log_file: 日志文件路径
    Returns:
        list: 日志记录列表
    """
    logs = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            logs.append(json.loads(line.strip()))
    return logs

def get_latest_metrics(log_file):
    """
    获取最新的指标
    Args:
        log_file: 日志文件路径
    Returns:
        dict: 最新的训练指标
    """
    logs = read_logs(log_file)
    if logs:
        return logs[-1]
    return None