import re
import os
import matplotlib.pyplot as plt

# ===== 改成你的实验日志目录 =====
log_dir = r"D:\Classification_test\experiments\custom_cnn_baseline_20261013_133359\logs"

# 找到 .log 文件
log_file = None
for f in os.listdir(log_dir):
    if f.endswith('.log'):
        log_file = os.path.join(log_dir, f)
        break

if log_file is None:
    print("没有找到 .log 文件，请确认 log_dir 路径")
    exit()

# 解析日志：提取 epoch / train_loss / val_loss
pattern = re.compile(r"Epoch (\d+)/\d+.*train_loss=([\d.]+).*val_loss=([\d.]+)")

epochs, train_loss, val_loss = [], [], []
with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        m = pattern.search(line)
        if m:
            epochs.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            val_loss.append(float(m.group(3)))

if not epochs:
    print("日志里没有解析到 Loss 数据，请检查日志格式")
    exit()

# 画图
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='Train Loss', linewidth=2)
plt.plot(epochs, val_loss, label='Val Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CustomCNN Loss Curve on CIFAR-10')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存到 results/
os.makedirs('results', exist_ok=True)
plt.savefig('results/loss_curve.png', dpi=150)
print("✅ Loss 曲线已保存到 results/loss_curve.png")