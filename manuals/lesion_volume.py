import nibabel as nib
import os
import numpy as np
import pandas as pd


def analyze_masks(mask_dir):
    results = []

    # 获取目录下所有 nii.gz 文件
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')]

    print(f"开始分析 {len(mask_files)} 个 Mask 文件...")

    for filename in mask_files:
        file_path = os.path.join(mask_dir, filename)

        # 加载影像
        img = nib.load(file_path)
        data = img.get_fdata()
        header = img.header

        # 计算体素体积 (单位: mm^3)，通过 header 获取 spacing
        # spacing 通常是 (x_res, y_res, z_res)
        voxel_volume = np.prod(header.get_zooms())

        # 统计标签
        count_0 = np.sum(data == 0)
        count_1 = np.sum(data == 1)

        # 计算实际物理体积 (单位: cm^3 / ml)
        volume_cm3 = (count_1 * voxel_volume) / 1000.0

        results.append({
            'FileName': filename,
            'Background_Voxels': count_0,
            'Lesion_Voxels': count_1,
            'Lesion_Volume_cm3': round(volume_cm3, 2),
            'Lesion_Ratio_%': round((count_1 / (count_0 + count_1)) * 100, 4)
        })

    # 转换为 DataFrame 方便观察
    df = pd.DataFrame(results)

    # 打印全局统计信息
    print("\n--- 全局统计结果 ---")
    print(f"平均病灶体积: {df['Lesion_Volume_cm3'].mean():.2f} cm^3")
    print(f"最小病灶体积: {df['Lesion_Volume_cm3'].min():.2f} cm^3")
    print(f"最大病灶体积: {df['Lesion_Volume_cm3'].max():.2f} cm^3")
    print(f"平均病灶占比: {df['Lesion_Ratio_%'].mean():.4f}%")

    return df


# 使用示例
mask_dir = r'C:\Users\lenovo\Desktop\nnUNet\nnUNet_raw\Dataset001_AISDmini\labelsTr'
df_results = analyze_masks(mask_dir)
df_results.to_csv("lesion_stats.csv", index=False)