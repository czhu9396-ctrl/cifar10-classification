import re
import os
import matplotlib.pyplot as plt

# ===== 改成你的实验日志目录 =====
log_dir = r"experiments\custom_cnn_baseline_20261013_133359\logs"

# 找到 .log 文件
log_file = None
for f in os.listdir(log_dir):
    if f.endswith('.log'):
        log_file = os.path.join(log_dir, f)
        break

if log_file is None:
    print("没有找到 .log 文件，请检查 log_dir 路径")
    exit()

# 解析日志
epochs, train_acc, val_acc = [], [], []
pattern = re.compile(r"Epoch (\d+)/\d+.*train_acc=([\d.]+)%.*val_acc=([\d.]+)%")

with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        m = pattern.search(line)
        if m:
            epochs.append(int(m.group(1)))
            train_acc.append(float(m.group(2)))
            val_acc.append(float(m.group(3)))

if not epochs:
    print("日志里没解析到数据，请把日志内容截图发我")
    exit()

# 画图
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_acc, label='Train Accuracy', linewidth=2)
plt.plot(epochs, val_acc, label='Val Accuracy', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('CustomCNN Training Curve on CIFAR-10')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存到 results/
os.makedirs('results', exist_ok=True)
plt.savefig('results/training_curve.png', dpi=150)
print("图片已保存到 results/training_curve.png")