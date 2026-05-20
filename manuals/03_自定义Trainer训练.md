# AIS NCCT 分割 - nnU-Net v2 自定义Trainer部署指南

## 📁 文件说明

| 文件 | 用途 | 放置位置 |
|------|------|----------|
| `nnUNetTrainer_AIS_Weighted.py` | 自定义Trainer | `nnunetv2/training/nnUNetTrainer/variants/loss/` |
| `train_ais_windows.bat` | Windows训练脚本 | 任意工作目录 |
| `dataset_ais_template.json` | dataset.json模板 | `nnUNet_raw/DatasetXXX/` |

---

## 🚀 快速部署步骤

### Step 1: 放置自定义Trainer

**找到nnU-Net安装路径** (conda环境内):
```bash
# 在conda环境中运行Python，查找路径
python -c "import nnunetv2; print(nnunetv2.__path__[0])"
```

**复制Trainer文件**:
```bash
# 将 nnUNetTrainer_AIS_Weighted.py 复制到:
# <nnunetv2路径>/training/nnUNetTrainer/variants/loss/

# 示例 (Windows):
copy nnUNetTrainer_AIS_Weighted.py D:\anaconda3\envs\nnunet\Lib\site-packages\nnunetv2\training\nnUNetTrainer\variants\loss\

# 示例 (Linux/Mac):
cp nnUNetTrainer_AIS_Weighted.py ~/anaconda3/envs/nnunet/lib/python3.10/site-packages/nnunetv2/training/nnUNetTrainer/variants/loss/
```

### Step 2: 修改 dataset.json

确保通道名包含 **"CT"** 关键字以触发CT归一化:

```json
{
  "channel_names": {
    "0": "CT_minus_1",
    "1": "CT_center", 
    "2": "CT_plus_1"
  },
  "labels": {
    "background": 0,
    "infarct": 1
  },
  "numTraining": 100,
  "file_ending": ".nii.gz"
}
```

> ⚠️ **关键**: 如果通道名不含"CT"，nnU-Net会使用ZScoreNormalization，这对NCCT数据不利。

### Step 3: 运行训练

```bash
# 基础命令
nnUNetv2_train DATASET_ID 2d 0 -tr nnUNetTrainer_AIS_Weighted

# 带npz输出 (用于后续集成)
nnUNetv2_train DATASET_ID 2d 0 -tr nnUNetTrainer_AIS_Weighted --npz

# 从检查点恢复
nnUNetv2_train DATASET_ID 2d 0 -tr nnUNetTrainer_AIS_Weighted --c

# 验证
nnUNetv2_train DATASET_ID 2d 0 -tr nnUNetTrainer_AIS_Weighted --val
```

---

## 🔧 核心优化点详解

### 1. 加权Cross-Entropy (解决0.17%极端不平衡)

```python
ce_weights = [1.0, 30.0]  # [背景, 病灶]
```
- **默认版本**: 病灶类30倍权重
- **激进版本** (`nnUNetTrainer_AIS_Weighted_Aggressive`): 病灶类50倍权重
- 如果验证Dice仍低，切换到激进版本

### 2. 前景采样率提升

```python
self.oversample_foreground_percent = 0.5  # 默认0.33 -> 0.5
```
- 确保50%的训练patch包含前景(梗死区)
- 对极小病灶至关重要

### 3. Epsilon平滑防NaN

```python
'smooth': 1e-5  # Dice分母平滑
```
- 防止Epoch 740闪退
- 在`train_step()`中加入NaN检查，自动跳过异常batch

### 4. 检查点保存频率

```python
self.save_every = 25  # 默认50 -> 25
```
- 每25epoch保存一次，防止长时间训练丢失

---

## 🐛 常见问题排查

### Q1: Windows路径报错 (OSError/WinError)

**现象**: `OSError: [WinError 6] The handle is invalid`

**解决**:
1. 设置环境变量避免多进程问题:
   ```bash
   set OMP_NUM_THREADS=1
   set MKL_NUM_THREADS=1
   ```
2. 使用单线程数据加载 (修改Trainer中的`get_allowed_n_proc_DA()`返回0)
3. 确保路径不含中文或特殊字符

### Q2: 训练闪退/NaN (Epoch 740)

**现象**: Loss突然变成NaN，训练中断

**解决**:
- 本Trainer已内置NaN检测，会自动跳过异常batch
- 如果频繁出现，尝试:
  1. 降低学习率: `self.initial_lr = 5e-3`
  2. 增加weight_decay: `self.weight_decay = 1e-4`
  3. 检查数据预处理是否正常 (CT值范围应在-1000~1000)

### Q3: 过拟合严重 (Train Dice 0.67, Val Dice 0.377)

**解决**:
1. 使用本Trainer的加权CE (已内置)
2. 增加数据增强强度 (修改`get_training_transforms`中的概率)
3. 降低学习率
4. 考虑使用3D配置而非2D (如果显存允许)

### Q4: 推理时找不到自定义Trainer

**现象**: `Unable to locate trainer class nnUNetTrainer_AIS_Weighted`

**解决**:
推理时必须指定相同的Trainer:
```bash
nnUNetv2_predict -i INPUT -o OUTPUT -d DATASET_ID -c 2d -tr nnUNetTrainer_AIS_Weighted
```

---

## 📊 训练监控

训练过程中查看日志文件:
```
<nnUNet_results>/DatasetXXX/nnUNetTrainer_AIS_Weighted__nnUNetPlans__2d/fold_0/training_log_*.txt
```

关键指标:
- `train_loss`: 训练损失 (应为负数，如-0.7)
- `val_loss`: 验证损失
- `Pseudo dice`: 验证集Dice (目标 > 0.6)
- `ema_fg_dice`: EMA平滑的前景Dice

---

## 🔄 从检查点恢复

如果训练中途中断:
```bash
nnUNetv2_train DATASET_ID 2d 0 -tr nnUNetTrainer_AIS_Weighted --c
```

检查点文件:
- `checkpoint_latest.pth`: 最新epoch
- `checkpoint_best.pth`: 最佳EMA Dice
- `checkpoint_final.pth`: 训练结束
- `checkpoint_emergency.pth`: 异常中断时紧急保存

---

## 📝 进阶调整

### 调整CE权重
编辑`nnUNetTrainer_AIS_Weighted.py`:
```python
# 第~85行
ce_weights = [1.0, 30.0]  # 改为需要的权重
```

### 调整前景采样率
```python
# 第~50行
self.oversample_foreground_percent = 0.5  # 0.0~1.0
```

### 添加真正的Dropout (需修改网络架构)
如需在网络中加入Dropout，需要修改`build_network_architecture`方法或自定义网络类。

---

## 📚 参考

- nnU-Net v2文档: https://github.com/MIC-DKFZ/nnUNet/tree/master/documentation
- 自定义Trainer指南: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/extending_nnunet.md
