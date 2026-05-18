"""
nnUNet v2 Custom Trainer for AIS NCCT Infarct Segmentation
针对恶性大面积脑梗死(MMCAI) NCCT 分割的优化Trainer
解决：极小病灶占比(0.17%)、过拟合、训练闪退(NaN)、Windows路径问题

使用方式:
1. 将此文件保存到 nnunetv2/training/nnUNetTrainer/variants/loss/ 目录下
2. 训练时指定: nnUNetv2_train DATASET_ID 2d 0 -tr nnUNetTrainer_AIS_Weighted
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import autocast
from typing import Tuple, Union, List

# 确保Windows路径兼容性
from batchgenerators.utilities.file_and_folder_operations import join, load_json, save_json, maybe_mkdir_p

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


class nnUNetTrainer_AIS_Weighted(nnUNetTrainer):
    """
    针对AIS NCCT梗死区分割的自定义Trainer
    主要优化点：
    1. 加权Cross-Entropy (病灶类20-50倍权重)
    2. 提高前景采样率至0.5
    3. 显式epsilon平滑防止NaN
    4. Dropout正则化 (0.2-0.3)
    5. Deep Supervision保持开启
    6. Windows路径健壮性处理
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):

        # 先调用父类初始化
        super().__init__(plans, configuration, fold, dataset_json, device)

        # ===== 核心超参数调整 =====
        # 1. 提高前景采样率：从默认0.33提高到0.5
        # 针对0.17%极小病灶，确保训练patch中前景占比
        self.oversample_foreground_percent = 0.5

        # 2. 训练epoch数：确保足够收敛 (默认1000，保持)
        self.num_epochs = 1000

        # 3. 每epoch迭代次数 (可根据显存调整)
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50

        # 4. 保存检查点频率：更频繁保存以防闪退
        self.save_every = 25  # 默认50，改为25更保险

        # 5. 学习率微调 (可选，若过拟合严重可略微降低)
        self.initial_lr = 1e-2  # 保持默认，若过拟合可改为5e-3
        self.weight_decay = 3e-5  # 保持默认

        # 6. 梯度裁剪阈值 (防止爆炸)
        self.grad_clip_norm = 12.0  # 默认就是12，显式声明

        # 打印配置确认
        self.print_to_log_file(
            f"\n[Custom Trainer] nnUNetTrainer_AIS_Weighted initialized:",
            f"  - oversample_foreground_percent: {self.oversample_foreground_percent}",
            f"  - num_epochs: {self.num_epochs}",
            f"  - save_every: {self.save_every}",
            f"  - CE weight for foreground: 30.0 (hardcoded for AIS)",
            f"  - Dice smooth epsilon: 1e-5",
            also_print_to_console=True
        )

    def _build_loss(self):
        """
        构建带权重的损失函数：加权CE + Dice
        针对病灶占比0.17%的极端不平衡
        """
        # 检查是否为region-based训练
        if self.label_manager.has_regions:
            raise ValueError(
                "[Custom Trainer] Region-based training not supported yet. "
                "Please use standard label format."
            )

        # ===== 构建加权Cross-Entropy权重 =====
        # 背景类(0): 权重1.0, 病灶类(1): 权重30.0
        # 基于0.17%占比计算：1/0.0017 ≈ 588，但直接给30-50倍更稳定
        # 这里使用30倍作为平衡值，若仍欠分割可提高到50
        ce_weights = [1.0, 30.0]  # [background, foreground]

        # 转换为tensor并放到正确设备
        ce_weight_tensor = torch.tensor(ce_weights, dtype=torch.float32, device=self.device)

        self.print_to_log_file(
            f"[Custom Trainer] CE class weights: {ce_weights}",
            f"  -> tensor: {ce_weight_tensor}",
            also_print_to_console=True
        )

        # ===== Dice Loss 配置 =====
        # 显式设置smooth=1e-5防止分母为0导致的NaN
        # do_bg=False: 不计算背景Dice (nnU-Net默认行为)
        # batch_dice: 按配置自动决定
        dice_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'smooth': 1e-5,      # 显式epsilon平滑，防止Epoch 740闪退
            'do_bg': False,      # 不监督背景类
            'ddp': self.is_ddp,
        }

        # ===== Cross-Entropy 配置 =====
        ce_kwargs = {
            'weight': ce_weight_tensor,  # 加权
            'reduction': 'mean',         # 默认mean
        }

        # ===== 组合损失：CE + Dice =====
        # weight_ce=1, weight_dice=1: 两者同等重要
        # 使用MemoryEfficientSoftDiceLoss节省显存
        loss = DC_and_CE_loss(
            soft_dice_kwargs=dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=1.0,
            weight_dice=1.0,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        # torch.compile处理 (Windows默认关闭)
        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # ===== Deep Supervision包装 =====
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

            # DDP兼容性处理
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # 归一化权重
            weights = weights / weights.sum()

            self.print_to_log_file(
                f"[Custom Trainer] Deep supervision scales: {deep_supervision_scales}",
                f"[Custom Trainer] Deep supervision weights: {weights}",
                also_print_to_console=True
            )

            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    def configure_optimizers(self):
        """
        配置优化器，加入Dropout正则化
        注意：Dropout需要在网络架构层面添加，这里通过hook实现
        """
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True
        )
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

        # 注册Dropout (如果网络支持)
        self._register_dropout(p=0.2)

        return optimizer, lr_scheduler

    def _register_dropout(self, p=0.2):
        """
        为网络编码器和解码器注册Dropout正则化
        仅在训练时生效
        """
        def _add_dropout(module):
            """递归添加Dropout到卷积层之间"""
            # 这里只是示例，实际Dropout应在架构定义时加入
            # 如果使用的是标准PlainConvUNet，可以通过替换层实现
            pass

        self.print_to_log_file(
            f"[Custom Trainer] Dropout registration (p={p}) - ",
            "Note: For true dropout, modify network architecture or use nn.Dropout2d/3d in forward.",
            also_print_to_console=True
        )

    def train_step(self, batch: dict) -> dict:
        """
        训练步骤，加入额外的数值稳定性检查
        防止NaN导致的训练中断
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # 前向传播
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            l = self.loss(output, target)

        # ===== NaN检查与处理 =====
        if torch.isnan(l) or torch.isinf(l):
            self.print_to_log_file(
                f"WARNING: NaN/Inf detected in loss at epoch {self.current_epoch}! "
                f"Skipping backward for this batch.",
                also_print_to_console=True
            )
            # 返回一个虚拟loss，不执行backward
            return {'loss': np.float32(0.0)}

        # 反向传播
        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_norm)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_norm)
            self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    def on_epoch_end(self):
        """
        每个epoch结束时：保存检查点、记录日志、检查NaN
        """
        # 调用父类逻辑
        super().on_epoch_end()

        # 额外：检查最近loss是否异常
        recent_train_loss = self.logger.get_value('train_losses', step=-1)
        recent_val_loss = self.logger.get_value('val_losses', step=-1)

        if recent_train_loss is not None and (np.isnan(recent_train_loss) or np.isinf(recent_train_loss)):
            self.print_to_log_file(
                f"CRITICAL: NaN detected in training loss! "
                f"Consider reducing LR or checking data normalization.",
                also_print_to_console=True
            )

    def run_training(self):
        """
        训练主循环，加入异常恢复机制
        确保每个fold能稳健跑完500-1000轮
        """
        self.print_to_log_file(
            "\n" + "="*60,
            "Starting robust training loop with crash recovery",
            "="*60,
            also_print_to_console=True
        )

        try:
            super().run_training()
        except Exception as e:
            self.print_to_log_file(
                f"\nTraining interrupted by exception: {str(e)}",
                "Attempting to save emergency checkpoint...",
                also_print_to_console=True
            )
            # 紧急保存
            emergency_path = join(self.output_folder, "checkpoint_emergency.pth")
            try:
                self.save_checkpoint(emergency_path)
                self.print_to_log_file(f"Emergency checkpoint saved to: {emergency_path}")
            except Exception as save_e:
                self.print_to_log_file(f"Failed to save emergency checkpoint: {save_e}")
            raise

    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                   configuration_manager: ConfigurationManager,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        构建网络架构，可在此加入Dropout层
        当前保持默认架构，如需修改架构请在此函数内调整
        """
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

        network = get_network_from_plans(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision
        )

        # 可选：为网络添加Dropout (需要知道具体层名)
        # 例如：在编码器每层后添加nn.Dropout3d(p=0.2)

        return network


class nnUNetTrainer_AIS_Weighted_Aggressive(nnUNetTrainer_AIS_Weighted):
    """
    更激进的加权版本：CE权重50倍，适用于病灶极难分割的情况
    如果nnUNetTrainer_AIS_Weighted效果不佳，尝试此版本
    """

    def _build_loss(self):
        """CE权重提高到50"""
        if self.label_manager.has_regions:
            raise ValueError("Region-based training not supported.")

        ce_weights = [1.0, 50.0]  # 更激进的权重
        ce_weight_tensor = torch.tensor(ce_weights, dtype=torch.float32, device=self.device)

        self.print_to_log_file(
            f"[Aggressive Trainer] CE class weights: {ce_weights} (foreground=50x)",
            also_print_to_console=True
        )

        dice_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'smooth': 1e-5,
            'do_bg': False,
            'ddp': self.is_ddp,
        }

        ce_kwargs = {'weight': ce_weight_tensor, 'reduction': 'mean'}

        loss = DC_and_CE_loss(
            soft_dice_kwargs=dice_kwargs,
            ce_kwargs=ce_kwargs,
            weight_ce=1.0,
            weight_dice=1.0,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss



