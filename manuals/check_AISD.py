from pathlib import Path
import SimpleITK as sitk


def full_dataset_check():
    # 配置
    dicom_bases = [
        Path(r"C:\Users\Administrator\Desktop\AISD\dicom-0"),
        Path(r"C:\Users\Administrator\Desktop\AISD\dicom-1"),
        Path(r"C:\Users\Administrator\Desktop\AISD\dicom-2"),
        Path(r"C:\Users\Administrator\Desktop\AISD\dicom-3"),
    ]
    mask_base = Path(r"C:\Users\Administrator\Desktop\AISD\mask")
    images_dir = Path(r"C:\Users\Administrator\Desktop\nnUNet\nnUNet_raw\Dataset001_AISDmini\imagesTr")
    labels_dir = Path(r"C:\Users\Administrator\Desktop\nnUNet\nnUNet_raw\Dataset001_AISDmini\labelsTr")

    # ✅ 修正：使用 .name.split("_0000")[0] 正确提取 case_id（去掉 .nii.gz 后缀）
    all_cases = [f.name.split("_0000")[0] for f in images_dir.glob("*_0000.nii.gz")]

    # 问题分类
    missing_dicom = []  # 完全找不到DICOM
    missing_mask = []  # 完全找不到MASK
    count_mismatch = []  # DICOM和MASK数量不匹配
    size_mismatch = []  # NIfTI维度不匹配（如16 vs 26）

    print(f"开始检查 {len(all_cases)} 个病例...\n")

    for case_id in all_cases:
        # 1. 查找DICOM（遍历所有dicom-*文件夹）
        dcm_files = []
        dicom_location = None
        for dicom_base in dicom_bases:
            case_dir = dicom_base / case_id / "CT"
            if case_dir.exists():
                dcm_files = list(case_dir.glob("*.dcm"))
                if dcm_files:
                    dicom_location = dicom_base.name
                    break

        # 2. 查找MASK
        mask_dir = mask_base / case_id
        mask_files = list(mask_dir.glob("*.png")) if mask_dir.exists() else []

        # 3. 检查NIfTI维度匹配
        img_file = images_dir / f"{case_id}_0000.nii.gz"
        label_file = labels_dir / f"{case_id}.nii.gz"

        img_size = None
        label_size = None

        if img_file.exists():
            try:
                img = sitk.ReadImage(str(img_file))
                img_size = img.GetSize()  # (x, y, z)
            except Exception as e:
                print(f"⚠️  读取图像失败 {case_id}: {e}")

        if label_file.exists():
            try:
                label = sitk.ReadImage(str(label_file))
                label_size = label.GetSize()
            except Exception as e:
                print(f"⚠️  读取标签失败 {case_id}: {e}")

        # 分类问题
        status = ""
        if len(dcm_files) == 0:
            missing_dicom.append(case_id)
            status = "❌ 缺DICOM"
        elif len(mask_files) == 0:
            missing_mask.append(case_id)
            status = "❌ 缺MASK"
        elif len(dcm_files) != len(mask_files):
            count_mismatch.append(f"{case_id}(DICOM:{len(dcm_files)}, MASK:{len(mask_files)})")
            status = f"⚠️ 数量不匹配({len(dcm_files)} vs {len(mask_files)})"
        elif img_size and label_size and img_size != label_size:
            size_mismatch.append(f"{case_id}(图像:{img_size}, 标签:{label_size})")
            status = f"❌ 维度不匹配{img_size} vs {label_size}"
        else:
            status = f"✓ 正常"

        # 打印详情
        dcm_loc = f"在{dicom_location}" if dicom_location else "未找到"
        print(f"{case_id}: DICOM {len(dcm_files):2d}个 {dcm_loc:12s}, MASK {len(mask_files):2d}个 - {status}")

    # 汇总报告
    print(f"\n{'=' * 70}")
    print(f"📊 总检查: {len(all_cases)} 个病例")
    normal_count = len(all_cases) - len(missing_dicom) - len(missing_mask) - len(count_mismatch) - len(size_mismatch)
    print(f"✅ 正常: {normal_count} 个")
    print(f"\n❌ 缺失DICOM: {len(missing_dicom)} 个")
    if missing_dicom:
        print(f"   列表: {missing_dicom}")

    print(f"\n❌ 缺失MASK: {len(missing_mask)} 个")
    if missing_mask:
        print(f"   列表: {missing_mask[:10]}...")

    print(f"\n⚠️ 数量不匹配(DICOM vs MASK): {len(count_mismatch)} 个")
    for item in count_mismatch[:5]:
        print(f"   {item}")

    print(f"\n❌ 维度不匹配(NIfTI): {len(size_mismatch)} 个")
    for item in size_mismatch:
        print(f"   {item}")

    # 返回问题病例供后续处理
    return {
        'missing_dicom': missing_dicom,
        'missing_mask': missing_mask,
        'count_mismatch': count_mismatch,
        'size_mismatch': size_mismatch
    }


# 运行检查
problems = full_dataset_check()