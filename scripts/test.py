import os
import sys
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse

# 把项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从 datasets.transforms 里导入正确的预处理函数（和你源码一致）
from datasets.transforms import get_eval_transform

# 导入模型（如果这里报错，说明模型类名或路径不对，需要根据实际项目改）
from models.custom_cnn import CustomCNN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='confits/default.yaml')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 加载测试集
    test_transform = get_eval_transform()
    test_ds = datasets.CIFAR10(
        root=cfg['data']['root'], 
        train=False, 
        download=False, 
        transform=test_transform
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=cfg['data']['batch_size'], 
        shuffle=False, 
        num_workers=0
    )
    print(f"✅ 测试集加载成功，图片数量: {len(test_ds)}")

    # 2. 加载模型
    model = CustomCNN(num_classes=10)
    model = model.to(device)

    # 模型权重路径（根据你实际的时间戳文件夹修改）
    checkpoint_dir = r"experiments\custom_cnn_baseline_20261013_133359\checkpoints"

    if not os.path.exists(checkpoint_dir):
        print(f"找不到模型目录: {checkpoint_dir}")
        return

    # 寻找 .pth 文件
    pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if pth_files:
        # 优先找 best.pth，找不到就取第一个
        best_model_path = os.path.join(checkpoint_dir, 'best.pth')
        if not os.path.exists(best_model_path):
            best_model_path = os.path.join(checkpoint_dir, pth_files[0])
        print(f"加载模型文件: {best_model_path}")
    else:
        print(f"在 {checkpoint_dir} 中没有找到 .pth 文件")
        return

    # ========== 关键修改开始 ==========
    # 先加载整个 checkpoint 字典
    checkpoint = torch.load(best_model_path, map_location=device)
    
    # 从字典里取出真正的模型参数
    # 注意：如果这里报 KeyError，说明保存时的键名不是 'model_state_dict'
    # 可以打印 checkpoint.keys() 看看实际有哪些键名
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # 如果实在找不到，尝试直接加载（如果保存时只保存了参数）
        model.load_state_dict(checkpoint)
    
    model.eval()
    print("模型加载完毕，开始计算准确率...")
    # ========== 关键修改结束 ==========

    # 3. 开始测试
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"\n✅ 测试集准确率: {acc:.2f}%")

if __name__ == '__main__':
    main()