import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# 防止大图警告
Image.MAX_IMAGE_PIXELS = None


def create_dir(path):
    """创建文件夹如果不存在"""
    if not os.path.exists(path):
        os.makedirs(path)


def get_mask_path(mask_base_dir, img_name_pure):
    """
    在指定目录下寻找对应主文件名的 mask 文件，支持多种后缀 (.png, .tif 等)
    解决测试集可能是 .tif 而训练集是 .png 的问题
    """
    for ext in ['.png', '.tif', '.jpg', '.PNG', '.TIF']:
        trial_path = os.path.join(mask_base_dir, img_name_pure + ext)
        if os.path.exists(trial_path):
            return trial_path
    return None


def process_dsifn(original_root, mid_root_512, final_root_256):
    # 1. 准备输出目录结构
    # DSIFN-CD-512 的结构 (可选，为了中间检查)
    for split in ['train', 'val', 'test']:
        create_dir(os.path.join(mid_root_512, split, 't1'))
        create_dir(os.path.join(mid_root_512, split, 't2'))
        create_dir(os.path.join(mid_root_512, split, 'label'))

    # DSIFN-CD-256 的标准结构
    create_dir(os.path.join(final_root_256, 'A'))
    create_dir(os.path.join(final_root_256, 'B'))
    create_dir(os.path.join(final_root_256, 'label'))
    create_dir(os.path.join(final_root_256, 'list'))

    # 2. 开始循环处理 train, val, test
    for split in ['train', 'val', 'test']:
        print(f"\n=== 正在处理 {split} 集 ===")
        split_dir = os.path.join(original_root, split)
        t1_dir = os.path.join(split_dir, 't1')
        t2_dir = os.path.join(split_dir, 't2')

        # 自动检测 mask 文件夹名字 (可能是 mask_256 也可能是 mask)
        if os.path.exists(os.path.join(split_dir, 'mask_256')):
            mask_dir_name = 'mask_256'
        elif os.path.exists(os.path.join(split_dir, 'mask')):
            mask_dir_name = 'mask'
        else:
            print(f"严重错误：在 {split_dir} 下找不到 mask 或 mask_256 文件夹！")
            continue
        mask_dir = os.path.join(split_dir, mask_dir_name)

        # 获取所有图片文件
        files = [f for f in os.listdir(t1_dir) if f.lower().endswith(('.png', '.jpg', '.tif'))]
        split_file_list = []  # 用于存储当前 split 切割后的所有子图名称

        for fname in tqdm(files, desc=f"处理 {split}"):
            name_pure = os.path.splitext(fname)[0]

            p_t1 = os.path.join(t1_dir, fname)
            p_t2 = os.path.join(t2_dir, fname)
            p_mask = get_mask_path(mask_dir, name_pure)

            if p_mask is None:
                print(f"警告：找不到 {fname} 对应的标签文件，已跳过。")
                continue

            # --- 阶段 1: 清洗并统一到 512x512 PNG ---
            img1 = Image.open(p_t1).convert('RGB')
            img2 = Image.open(p_t2).convert('RGB')
            # 先不转 'L'，读取原始 mask 以便检查尺寸
            mask = Image.open(p_mask)

            # 强制统一到 512x512
            if img1.size != (512, 512): img1 = img1.resize((512, 512), Image.BILINEAR)
            if img2.size != (512, 512): img2 = img2.resize((512, 512), Image.BILINEAR)

            # 【关键】如果 mask 是 256 (如训练集)，用最近邻插值放大到 512，保证标签值不变
            if mask.size != (512, 512):
                mask = mask.resize((512, 512), Image.NEAREST)

            # 转为灰度图 (L模式) 准备切割
            if mask.mode != 'L':
                mask = mask.convert('L')

            # 保存清洗后的 512 图片到中间文件夹 (可选，但你要求了)
            save_name_512 = name_pure + '.png'  # 统一保存为 .png
            img1.save(os.path.join(mid_root_512, split, 't1', save_name_512))
            img2.save(os.path.join(mid_root_512, split, 't2', save_name_512))
            mask.save(os.path.join(mid_root_512, split, 'label', save_name_512))

            # --- 阶段 2: 切割成 4 张 256x256 并保存到最终文件夹 ---
            # 坐标点：(左, 上, 右, 下)
            crops = [
                (0, 0, 256, 256),  # 左上
                (256, 0, 512, 256),  # 右上
                (0, 256, 256, 512),  # 左下
                (256, 256, 512, 512)  # 右下
            ]

            for i, box in enumerate(crops):
                # 切割
                patch1 = img1.crop(box)
                patch2 = img2.crop(box)
                patch_mask = mask.crop(box)

                # 生成统一的子图文件名
                patch_name = f"{name_pure}_{i}.png"

                # 保存到最终的 A/B/label 文件夹
                patch1.save(os.path.join(final_root_256, 'A', patch_name))
                patch2.save(os.path.join(final_root_256, 'B', patch_name))
                patch_mask.save(os.path.join(final_root_256, 'label', patch_name))

                # 记录子图文件名到列表
                split_file_list.append(patch_name)

        # 保存当前 split 的列表文件 (train.txt / val.txt / test.txt)
        list_file_path = os.path.join(final_root_256, 'list', f'{split}.txt')
        with open(list_file_path, 'w') as f:
            for item in split_file_list:
                f.write(item + '\n')
        print(f"已生成 {list_file_path}，共包含 {len(split_file_list)} 张 256x256 样本。")


if __name__ == '__main__':
    # --- 配置路径 ---
    # 1. 原始 DSIFN 数据集根目录 (里面应该有 train, val, test 文件夹)
    ORIGINAL_ROOT = r"E:\RemoteDatasets\DSIFN-CD"

    # 2. 中间 512 数据集保存位置
    MID_ROOT_512 = r"E:\RemoteDatasets\DSIFN-CD-512"

    # 3. 最终 256 标准数据集保存位置 (用于训练)
    FINAL_ROOT_256 = r"E:\RemoteDatasets\DSIFN-CD-256"
    # ----------------

    print("开始处理 DSIFN-CD 数据集...")
    print(f"源数据: {ORIGINAL_ROOT}")
    print(f"中间数据 (512): {MID_ROOT_512}")
    print(f"最终数据 (256): {FINAL_ROOT_256}")

    process_dsifn(ORIGINAL_ROOT, MID_ROOT_512, FINAL_ROOT_256)

    print("\n=== 处理全部完成 ===")
    print(f"请在 main_cd_new.py 中设置 args.data_name = 'DSIFN-CD'")
    print(f"并确保在 utils.py 中将其路径指向: {FINAL_ROOT_256}")