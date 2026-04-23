import os
import shutil
from pathlib import Path

# ===== 配置区域 =====
dataset_id = "001"  # 你的数据集ID
base_dir = Path(r"C:\Users\Administrator\Desktop\nnUNet\nnUNet_raw")  # 根据实际路径修改
dataset_name = f"Dataset{dataset_id}_AISDmini"  # 你的数据集名称

# 官方测试集ID列表（52个）
test_case_ids = [
    "0073410", "0072723", "0226290", "0537908", "0538058", "0091415",
    "0538780", "0073540", "0226188", "0226258", "0226314", "0091507",
    "0226298", "0538975", "0226257", "0226142", "0072681", "0091538",
    "0538983", "0537961", "0091646", "0072765", "0226137", "0091621",
    "0091458", "0021822", "0538319", "0226133", "0091657", "0537925",
    "0073489", "0538502", "0091476", "0226136", "0538532", "0073312",
    "0539025", "0226309", "0226307", "0091383", "0021092", "0537990",
    "0226299", "0073060", "0538505", "0073424", "0091534", "0226125",
    "0072691", "0538425", "0226199", "0226261"
]

# 路径设置
images_tr = base_dir / dataset_name / "imagesTr"
images_ts = base_dir / dataset_name / "imagesTs"
labels_tr = base_dir / dataset_name / "labelsTr"
labels_ts = base_dir / dataset_name / "labelsTs"

# 创建目标目录（如果不存在）
images_ts.mkdir(parents=True, exist_ok=True)
labels_ts.mkdir(parents=True, exist_ok=True)

print(f"开始移动 {len(test_case_ids)} 个测试集样本...")
print(f"从: {images_tr}")
print(f"到: {images_ts}")
print("-" * 60)

# 移动图像文件
moved_count = 0
failed_cases = []

for case_id in test_case_ids:
    # nnUNet 图像命名规则: case_id_0000.nii.gz (假设是单模态CT，modality_id=0000)
    img_filename = f"{case_id}_0000.nii.gz"
    src_img = images_tr / img_filename
    dst_img = images_ts / img_filename

    # 移动图像
    if src_img.exists():
        try:
            shutil.move(str(src_img), str(dst_img))
            print(f"✓ 移动图像: {img_filename}")
            moved_count += 1
        except Exception as e:
            print(f"✗ 移动图像失败 {case_id}: {e}")
            failed_cases.append(case_id)
    else:
        print(f"✗ 图像不存在: {img_filename}")
        # 尝试查找其他可能的命名（如没有_0000后缀）
        alt_files = list(images_tr.glob(f"{case_id}*.nii.gz"))
        if alt_files:
            print(f"  找到备选文件: {[f.name for f in alt_files]}")

    # 移动对应的标签文件（如果有）
    label_filename = f"{case_id}.nii.gz"
    src_label = labels_tr / label_filename
    dst_label = labels_ts / label_filename

    if src_label.exists():
        try:
            shutil.move(str(src_label), str(dst_label))
            print(f"  ✓ 移动标签: {label_filename}")
        except Exception as e:
            print(f"  ✗ 移动标签失败 {case_id}: {e}")

print("-" * 60)
print(f"完成！成功移动 {moved_count}/{len(test_case_ids)} 个测试样本")
if failed_cases:
    print(f"失败病例: {failed_cases}")

# 验证最终数量
remaining_tr = len(list(images_tr.glob("*.nii.gz")))
moved_to_ts = len(list(images_ts.glob("*.nii.gz")))
print(f"\n验证:")
print(f"  imagesTr 剩余: {remaining_tr} 个文件")
print(f"  imagesTs 现有: {moved_to_ts} 个文件")
print(f"  总计: {remaining_tr + moved_to_ts} 个文件")