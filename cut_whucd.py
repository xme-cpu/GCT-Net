import os
import random
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

# 解决大图像解压限制
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def create_directories(base_path):
    """创建A、B、label、list目录结构"""
    main_dirs = ['A', 'B', 'label', 'list']
    for dir_name in main_dirs:
        dir_path = os.path.join(base_path, dir_name)
        os.makedirs(dir_path, exist_ok=True)

    return {
        'A': os.path.join(base_path, 'A'),
        'B': os.path.join(base_path, 'B'),
        'label': os.path.join(base_path, 'label'),
        'list': os.path.join(base_path, 'list')
    }


def split_into_patches(img_path, patch_size=256, target_grid=(60, 127)):
    """
    按网格切割图像，返回子图像数组和对应的行/列索引
    """
    try:
        img = Image.open(img_path)
        img_array = np.array(img)
        orig_h, orig_w = img_array.shape[:2]
        print(f"原始尺寸: {orig_w}×{orig_h}")

        # 目标尺寸计算
        target_h = target_grid[0] * patch_size
        target_w = target_grid[1] * patch_size
        print(f"目标尺寸: {target_w}×{target_h} (共{target_grid[0] * target_grid[1]}个子图)")

        # 补零计算
        pad_h = target_h - orig_h
        pad_w = target_w - orig_w
        print(f"补零量: 底部{pad_h}px, 右侧{pad_w}px")

        # 补零处理
        pad_params = [(0, pad_h), (0, pad_w)]
        if len(img_array.shape) == 3:
            pad_params.append((0, 0))
        img_array = np.pad(img_array, pad_params, mode='constant', constant_values=0)
        print(f"补零后尺寸: {img_array.shape[1]}×{img_array.shape[0]}")

        # 切割并记录索引
        patches = []
        indices = []  # 存储 (i,j) 行/列索引
        for i in range(target_grid[0]):
            for j in range(target_grid[1]):
                h_start = i * patch_size
                h_end = h_start + patch_size
                w_start = j * patch_size
                w_end = w_start + patch_size
                patches.append(img_array[h_start:h_end, w_start:w_end, ...])
                indices.append((i, j))  # 记录每个子图的行和列索引

        print(f"切割完成: 共{len(patches)}个子图像")
        return patches, indices
    except Exception as e:
        print(f"切割错误: {str(e)}")
        raise


def save_patches_by_split(patches, indices, save_dir, split_name, ext='tif'):
    """
    按数据集划分（train/val/test）保存子图，命名格式：split_name_i_j.ext
    """
    saved_names = []
    for (i, j), patch in tqdm(zip(indices, patches), desc=f"保存{split_name}子图"):
        filename = f"{split_name}_{i}_{j}.{ext}"  # 与cut.py统一：划分前缀_行索引_列索引.扩展名
        save_path = os.path.join(save_dir, filename)
        Image.fromarray(patch).save(save_path, format='TIFF')
        saved_names.append(filename)
    return saved_names


def create_list_files(file_list, list_dir, set_name):
    """生成列表文件（内容为对应划分的子图名称）"""
    file_path = os.path.join(list_dir, f"{set_name}.txt")
    with open(file_path, 'w') as f:
        f.write('\n'.join(file_list))
    print(f"生成{set_name}.txt: {len(file_list)}个样本")


def process_whu_cd(raw_path, output_path, patch_size=256):
    """主处理函数：补零→切割→划分→保存"""
    # 固定划分数量
    train_num = 6096
    val_num = 762
    test_num = 762

    # 原始图像路径
    two_period_dir = os.path.join(raw_path, "1. The two-period image data")
    img_paths = {
        'A': os.path.join(two_period_dir, "before", "before.tif"),
        'B': os.path.join(two_period_dir, "after", "after.tif"),
        'label': os.path.join(two_period_dir, "change label", "change_label.tif")
    }

    # 检查文件存在性
    for key, path in img_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{key}文件缺失: {path}")

    # 创建输出目录
    dirs = create_directories(output_path)

    # 切割A/B/label并获取索引（确保三者索引一一对应）
    print("\n=== 切割第一时相图像（A）===")
    a_patches, indices = split_into_patches(img_paths['A'])
    print("\n=== 切割第二时相图像（B）===")
    b_patches, _ = split_into_patches(img_paths['B'])
    print("\n=== 切割变化标签（label）===")
    label_patches, _ = split_into_patches(img_paths['label'])

    # 验证数量一致性
    total = train_num + val_num + test_num
    assert len(a_patches) == len(b_patches) == len(label_patches) == total, \
        f"子图数量错误，应为{total}，实际A={len(a_patches)}, B={len(b_patches)}, label={len(label_patches)}"

    # 随机划分数据集
    print("\n=== 划分数据集 ===")
    random_indices = list(range(total))
    random.shuffle(random_indices)
    train_idx = random_indices[:train_num]
    val_idx = random_indices[train_num:train_num + val_num]
    test_idx = random_indices[train_num + val_num:]

    # 按划分提取子图和索引
    split_data = {
        'train': {
            'indices': [indices[i] for i in train_idx],
            'A': [a_patches[i] for i in train_idx],
            'B': [b_patches[i] for i in train_idx],
            'label': [label_patches[i] for i in train_idx]
        },
        'val': {
            'indices': [indices[i] for i in val_idx],
            'A': [a_patches[i] for i in val_idx],
            'B': [b_patches[i] for i in val_idx],
            'label': [label_patches[i] for i in val_idx]
        },
        'test': {
            'indices': [indices[i] for i in test_idx],
            'A': [a_patches[i] for i in test_idx],
            'B': [b_patches[i] for i in test_idx],
            'label': [label_patches[i] for i in test_idx]
        }
    }

    # 保存子图并收集列表内容
    print("\n=== 保存子图像 ===")
    list_contents = {}
    for split in ['train', 'val', 'test']:
        # 保存A/B/label子图（统一用划分前缀命名）
        a_names = save_patches_by_split(
            split_data[split]['A'],
            split_data[split]['indices'],
            dirs['A'],
            split
        )
        b_names = save_patches_by_split(
            split_data[split]['B'],
            split_data[split]['indices'],
            dirs['B'],
            split
        )
        label_names = save_patches_by_split(
            split_data[split]['label'],
            split_data[split]['indices'],
            dirs['label'],
            split
        )
        # 列表文件内容与A保持一致（A/B/label文件名一一对应）
        list_contents[split] = a_names

    # 生成列表文件
    print("\n=== 生成列表文件 ===")
    for split in ['train', 'val', 'test']:
        create_list_files(list_contents[split], dirs['list'], split)

    # 最终结果提示
    print(f"\n=== 处理完成 ===")
    print(f"输出目录: {output_path}")
    print(f"训练集: {len(list_contents['train'])}个, 验证集: {len(list_contents['val'])}个, 测试集: {len(list_contents['test'])}个")
    print(f"子图尺寸: {patch_size}×{patch_size}（TIF格式）")
    print(f"文件名格式: [train/val/test]_[行索引]_[列索引].tif（与cut.py统一）")


if __name__ == "__main__":
    random.seed(42)  # 固定随机种子，确保划分结果可复现
    RAW_DATA_PATH = r"E:\RemoteDatasets\WHU-CD"
    OUTPUT_DATA_PATH = r"E:\RemoteDatasets\WHU-CD_BIT_256"
    process_whu_cd(RAW_DATA_PATH, OUTPUT_DATA_PATH)