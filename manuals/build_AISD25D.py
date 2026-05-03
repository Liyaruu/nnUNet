import os
import shutil
import SimpleITK as sitk
import numpy as np

# 设置路径
old_dataset = r"C:\Users\lenovo\Desktop\nnUNet\nnUNet_raw\Dataset001_AISDmini"
new_dataset = r"C:\Users\lenovo\Desktop\nnUNet\nnUNet_raw\Dataset002_AISD25D"

os.makedirs(os.path.join(new_dataset, "imagesTr"), exist_ok=True)
os.makedirs(os.path.join(new_dataset, "labelsTr"), exist_ok=True)

# 1. 直接复制标签 (Label 不需要改名，也不需要多份)
print("正在复制标签...")
for f in os.listdir(os.path.join(old_dataset, "labelsTr")):
    if f.endswith(".nii.gz"):
        shutil.copy(os.path.join(old_dataset, "labelsTr", f),
                    os.path.join(new_dataset, "labelsTr", f))

# 2. 生成 2.5D 图像数据
print("正在生成 2.5D 图像数据...")
for f in os.listdir(os.path.join(old_dataset, "imagesTr")):
    if f.endswith("_0000.nii.gz"):
        case_name = f.replace("_0000.nii.gz", "")
        img_itk = sitk.ReadImage(os.path.join(old_dataset, "imagesTr", f))
        img_npy = sitk.GetArrayFromImage(img_itk)  # (D, H, W)

        # ================== 新增：脑窗裁剪 ==================
        # 1. 裁剪：将 HU 值限制在 [0, 100] 区间
        # 这会将所有颅骨 (>100) 统一变为 100，所有空气和水 (<0) 变为 0
        img_npy = np.clip(img_npy, 0, 100)

        # 2. (建议项) 线性归一化：将 [0, 100] 映射到 [0, 1]
        # 虽然 nnU-Net 会做 Z-Score，但预先统一量程能让不同病例的对比度更一致
        img_npy = (img_npy - 0) / 100.0
        # ====================================================

        # 生成 t-1, t, t+1
        ch0 = np.roll(img_npy, 1, axis=0)
        ch0[0] = img_npy[0]
        ch1 = img_npy
        ch2 = np.roll(img_npy, -1, axis=0)
        ch2[-1] = img_npy[-1]

        # 保存为 _0000, _0001, _0002
        for i, data in enumerate([ch0, ch1, ch2]):
            out_img = sitk.GetImageFromArray(data.astype(np.float32))  # 确保以 float32 保存
            out_img.CopyInformation(img_itk)
            out_name = f"{case_name}_{i:04d}.nii.gz"
            sitk.WriteImage(out_img, os.path.join(new_dataset, "imagesTr", out_name))
        print(f"已完成 (含脑窗裁剪): {case_name}")