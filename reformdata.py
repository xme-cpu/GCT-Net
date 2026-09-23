import os
import shutil


def reorganize_levir_cd(original_root, target_root):
    """
    重新组织LEVIR-CD数据集以适应BIT模型的结构

    参数:
        original_root: LEVIR-CD原始数据集路径 (E:\RemoteDatasets\LEVIR-CD)
        target_root: 目标数据集路径 (可与original_root相同，也可不同)
    """
    # 创建目标文件夹结构
    folders = ['A', 'B', 'label', 'list']
    for folder in folders:
        os.makedirs(os.path.join(target_root, folder), exist_ok=True)

    # 处理三个子集：train, val, test
    splits = ['train', 'val', 'test']

    for split in splits:
        # 原始数据集中当前子集的路径
        original_split_path = os.path.join(original_root, split)

        # 获取该子集下所有图像文件名（假设A、B、label文件夹中的文件名相同）
        # 从A文件夹读取文件名作为基准
        a_folder = os.path.join(original_split_path, 'A')
        img_names = [f for f in os.listdir(a_folder)
                     if f.endswith(('.png', '.jpg', '.jpeg'))]

        # 复制图像到目标文件夹
        for img_name in img_names:
            # 复制A文件夹图像
            src_a = os.path.join(original_split_path, 'A', img_name)
            dst_a = os.path.join(target_root, 'A', img_name)
            shutil.copyfile(src_a, dst_a)

            # 复制B文件夹图像
            src_b = os.path.join(original_split_path, 'B', img_name)
            dst_b = os.path.join(target_root, 'B', img_name)
            shutil.copyfile(src_b, dst_b)

            # 复制label文件夹图像
            src_label = os.path.join(original_split_path, 'label', img_name)
            dst_label = os.path.join(target_root, 'label', img_name)
            shutil.copyfile(src_label, dst_label)

        # 生成对应的list文件
        list_file = os.path.join(target_root, 'list', f'{split}.txt')
        with open(list_file, 'w') as f:
            for img_name in img_names:
                f.write(f'{img_name}\n')

        print(f"处理完成 {split} 集，共 {len(img_names)} 张图像")


if __name__ == '__main__':
    # 原始LEVIR-CD数据集路径
    original_dataset_path = r"E:\RemoteDatasets\LEVIR-CD"

    # 目标路径（可以和原始路径相同，也可以指定新路径）
    # 建议使用新路径，避免覆盖原始数据
    target_dataset_path = r"E:\RemoteDatasets\LEVIR-CD_BIT"

    # 执行重组
    reorganize_levir_cd(original_dataset_path, target_dataset_path)
    print("数据集重组完成！")