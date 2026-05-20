# brain_ct_normalization.py
# 保存到: <nnunetv2>/preprocessing/normalization/brain_ct_normalization.py

import numpy as np
from nnunetv2.preprocessing.normalization.default_normalization_schemes import ImageNormalization

class BrainCTNormalization(ImageNormalization):
    """
    NCCT 脑组织专用 Min-Max 归一化
    硬裁剪 [0, 100] HU -> 除以 100 -> [0.0, 1.0]
    零动态统计依赖，跨扫描仪绝对一致
    """
    def __init__(self, use_mask_for_norm: bool = False, clip_range=None,
                 mean: float = None, std: float = None, percentiles=None):
        # 显式继承父类构造函数，确保反射实例化时参数100%兼容
        super().__init__(use_mask_for_norm, clip_range, mean, std, percentiles)
        # 本类专属常量（脑组织窗）
        self.clip_low = 0.0
        self.clip_high = 100.0
        self.divisor = 100.0

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        # 防御性类型转换（原始 .nii.gz 可能是 int16）
        image = image.astype(np.float32, copy=False)
        # 物理硬裁剪：只保留脑组织窗，阻断颅骨/空气/金属伪影
        image = np.clip(image, self.clip_low, self.clip_high)
        # 固定除数 Min-Max：保留绝对 HU 语义（0.2 = 20 HU）
        image = image / self.divisor
        return image