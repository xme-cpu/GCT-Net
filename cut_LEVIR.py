import os
import shutil
from PIL import Image


def split_image_1024_to_256(img_path, output_dir, img_name):
    """
    将1024x1024的图像切割成16个256x256的子图像
    返回切割后的子图像名称列表
    """
    # 打开图像
    with Image.open(img_path) as img:
        # 确保图像是1024x1024
        if img.size != (1024, 1024):
            raise ValueError(f"图像 {img_name} 尺寸不是1024x1024，无法切割")

        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)

        sub_img_names = []

        # 切割成4x4=16个256x256的子图像
        for i in range(4):  # 行索引 (0-3)
            for j in range(4):  # 列索引 (0-3)
                # 计算切割区域
                left = j * 256
                top = i * 256
                right = left + 256
                bottom = top + 256

                # 切割子图像
                sub_img = img.crop((left, top, right, bottom))

                # 生成子图像名称 (原名称_行索引_列索引.扩展名)
                base_name, ext = os.path.splitext(img_name)
                sub_img_name = f"{base_name}_{i}_{j}{ext}"
                sub_img_path = os.path.join(output_dir, sub_img_name)

                # 保存子图像
                sub_img.save(sub_img_path)
                sub_img_names.append(sub_img_name)

        return sub_img_names


def process_dataset(original_root, target_root):
    """
    处理整个数据集：
    1. 将A、B、label文件夹中的1024x1024图像切割为256x256子图像
    2. 更新list文件夹中的train.txt、val.txt、test.txt
    """
    # 创建目标文件夹结构
    folders = ['A', 'B', 'label', 'list']
    for folder in folders:
        os.makedirs(os.path.join(target_root, folder), exist_ok=True)

    # 处理三个子集的列表文件
    splits = ['train', 'val', 'test']

    for split in splits:
        # 读取原始列表文件
        original_list_path = os.path.join(original_root, 'list', f'{split}.txt')
        with open(original_list_path, 'r') as f:
            img_names = [line.strip() for line in f if line.strip()]

        # 用于存储切割后的子图像名称
        all_sub_img_names = []

        # 处理每张图像
        for img_name in img_names:
            print(f"处理 {split} 集中的 {img_name}...")

            # 切割A文件夹中的图像
            a_img_path = os.path.join(original_root, 'A', img_name)
            a_output_dir = os.path.join(target_root, 'A')
            a_sub_names = split_image_1024_to_256(a_img_path, a_output_dir, img_name)

            # 切割B文件夹中的图像
            b_img_path = os.path.join(original_root, 'B', img_name)
            b_output_dir = os.path.join(target_root, 'B')
            split_image_1024_to_256(b_img_path, b_output_dir, img_name)

            # 切割label文件夹中的图像
            label_img_path = os.path.join(original_root, 'label', img_name)
            label_output_dir = os.path.join(target_root, 'label')
            split_image_1024_to_256(label_img_path, label_output_dir, img_name)

            # 收集子图像名称
            all_sub_img_names.extend(a_sub_names)

        # 更新列表文件
        new_list_path = os.path.join(target_root, 'list', f'{split}.txt')
        with open(new_list_path, 'w') as f:
            for sub_img_name in all_sub_img_names:
                f.write(f"{sub_img_name}\n")

        print(f"完成 {split} 集处理，原图像 {len(img_names)} 张，切割后 {len(all_sub_img_names)} 张\n")


if __name__ == '__main__':
    # 原始数据集路径（经过上一步重组的数据集）
    original_dataset_path = r"E:\RemoteDatasets\LEVIR-CD_BIT"

    # 切割后的目标路径
    target_dataset_path = r"E:\RemoteDatasets\LEVIR-CD_BIT_256"

    # 执行切割
    process_dataset(original_dataset_path, target_dataset_path)
    print("所有图像切割完成！")
