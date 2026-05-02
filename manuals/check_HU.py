import nibabel as nib
import numpy as np


def check_hu_range(file_path):
    # 加载 NIfTI 文件
    img = nib.load(file_path)
    # 获取图像数据数组
    data = img.get_fdata()

    print(f"--- 文件: {file_path.split('/')[-1]} ---")
    print(f"最小值 (Min HU): {np.min(data):.2f}")
    print(f"最大值 (Max HU): {np.max(data):.2f}")
    print(f"平均值 (Mean HU): {np.mean(data):.2f}")

    # 查看脑组织核心区间的占比（以0-100为例）
    brain_pixels = np.sum((data > 0) & (data < 100))
    total_pixels = data.size
    print(f"0-100 HU 像素占比: {(brain_pixels / total_pixels) * 100:.2f}%")


# 替换为你实际的文件路径
check_hu_range(r"C:\Users\lenovo\Desktop\nnUNet\nnUNet_raw\Dataset002_AISD25D\imagesTr\0091440_0001.nii.gz")