# CIFAR10 图像分类项目

基于 **PyTorch** 的 CIFAR10 十分类科研入门项目。项目结构模块化、可复现、可扩展，支持**自写轻量 CNN** 与**公开预训练模型 ResNet18** 两种方案，通过 YAML 配置一键切换。

## 一、目录结构

```
Classification_test/
├── confits/               # 配置文件（YAML）
│   ├── default.yaml       # 自写 CNN 基线
│   └── resnet18.yaml      # ResNet18 对比
├── data/                  # CIFAR10 数据（自动下载，不入库）
├── datasets/              # 数据层：增强 + DataLoader
│   ├── transforms.py      # 数据增强策略
│   └── cifar10.py         # 数据集加载与划分
├── models/                # 模型层
│   ├── custom_cnn.py      # 自写特征提取网络 + 分类头
│   ├── resnet.py          # ResNet18 封装
│   └── __init__.py        # get_model() 工厂
├── losses/                # 损失函数层（工厂）
├── evaluation/            # 评估层
│   ├── metrics.py         # 准确率 / 混淆矩阵 / 分类报告
│   └── evaluate.py        # 测试集评估脚本
├── tools/                 # 工具层
│   ├── config.py          # 配置加载 / 设备选择
│   ├── seed.py            # 随机种子
│   ├── logger.py          # 日志 + TensorBoard
│   ├── checkpoint.py      # 模型保存 / 加载
│   └── optim.py           # 优化器 / 调度器
├── experiments/           # 实验产出（自动生成，不入库）
│   └── <exp_name>_<时间戳>/
│       ├── logs/          # TensorBoard 事件 + 日志文件
│       ├── checkpoints/   # best.pth / last.pth
│       └── config.yaml    # 本次实验配置快照
├── scripts/               # 入口脚本
│   ├── train.py           # 训练主脚本
│   ├── run.bat            # Windows 一键启动
│   └── run.sh             # Linux/macOS 一键启动
├── docs/                  # 文档
├── requirements.txt       # 依赖清单
└── .gitignore
```

## 二、环境安装

```bash
# 1. 创建虚拟环境（推荐）
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装 PyTorch（若 pip 默认源速度慢，请到 pytorch.org 选择对应 CUDA 版本）
# 例如 CPU 版：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 三、快速开始

### 1. 训练

```bash
# 自写 CNN 基线（默认）
python scripts/train.py --config confits/default.yaml

# 公开模型 ResNet18 对比
python scripts/train.py --config confits/resnet18.yaml

# Windows 一键启动（可选传配置文件）
scripts\run.bat
scripts\run.bat confits\resnet18.yaml
```

首次运行会自动下载 CIFAR10 数据集到 `data/` 目录。

### 2. 评估

训练完成后，在 `experiments/<实验目录>/checkpoints/` 下会生成 `best.pth`，运行：

```bash
python evaluation/evaluate.py \
    --config confits/default.yaml \
    --checkpoint experiments/<实验目录>/checkpoints/best.pth
```

评估会输出 **Top-1 准确率**、**分类报告**，并保存**混淆矩阵图**与报告文件到 `experiments/<实验目录>/eval/`。

### 3. 查看训练曲线（TensorBoard）

```bash
tensorboard --logdir experiments
```

浏览器打开 `http://localhost:6006` 查看 loss / acc / 学习率曲线。

## 四、配置说明

所有超参数集中在 `confits/*.yaml`，主要字段：

| 字段 | 说明 |
|------|------|
| `experiment.name` | 实验名称（输出目录前缀） |
| `experiment.seed` | 随机种子（保证可复现） |
| `experiment.device` | `auto` / `cuda` / `cpu` |
| `data.batch_size` | 批大小 |
| `data.val_ratio` | 验证集划分比例 |
| `model.name` | `custom_cnn` / `resnet18` |
| `model.base_channels` | 自写 CNN 首层通道数 |
| `model.pretrained` | ResNet 是否加载预训练 |
| `loss.name` | `cross_entropy` / `label_smoothing` |
| `optimizer.name` | `sgd` / `adam` / `adamw` |
| `scheduler.name` | `cosine` / `step` / `none` |
| `train.epochs` | 训练轮数 |

## 五、如何扩展

- **加新模型**：在 `models/` 新建文件，在 `models/__init__.py` 的 `get_model()` 中注册分支。
- **加新损失**：在 `losses/__init__.py` 的 `get_loss()` 中注册。
- **加新数据集**：在 `datasets/` 仿照 `cifar10.py` 新增加载函数。
- **复现实验**：`experiments/<实验目录>/config.yaml` 保存了当次完整配置，可直接 `--config` 复用。

## 六、常见问题

- **Windows 多进程报错**：将 `data.num_workers` 改为 `0`。
- **下载数据集慢**：可手动下载 CIFAR10 并解压到 `data/cifar-10-batches-py`。
- **CUDA out of memory**：减小 `data.batch_size`。
