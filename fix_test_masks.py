import os
from PIL import Image
from tqdm import tqdm


def fix_test_masks():
    # --- [配置路径] 请确认这些路径是正确的 ---
    # 1. 源头：刚才转换好的 512x512 PNG 掩膜文件夹
    SRC_MASK_512_DIR = r"E:\RemoteDatasets\DSIFN-CD\test\mask_png"

    # 2. 目标：需要修复的 256x256 label 文件夹
    DST_LABEL_256_DIR = r"E:\RemoteDatasets\DSIFN-CD-256\label"
    # ------------------------------------

    if not os.path.exists(SRC_MASK_512_DIR):
        print(f"错误：找不到源文件夹 {SRC_MASK_512_DIR}")
        return

    print(f"开始修复...")
    print(f"源 (512): {SRC_MASK_512_DIR}")
    print(f"目标 (256): {DST_LABEL_256_DIR}")

    # 获取所有 512 的 PNG 掩膜文件 (预期是 0.png 到 47.png)
    files = [f for f in os.listdir(SRC_MASK_512_DIR) if f.lower().endswith('.png')]
    print(f"找到 {len(files)} 张待处理的 512x512 掩膜。")

    # 定义 4 个子图的坐标 (左, 上, 右, 下)
    crops = [
        (0, 0, 256, 256),  # 0: 左上
        (256, 0, 512, 256),  # 1: 右上
        (0, 256, 256, 512),  # 2: 左下
        (256, 256, 512, 512)  # 3: 右下
    ]

    count = 0
    for fname in tqdm(files):
        name_pure = os.path.splitext(fname)[0]  # 例如 "0"
        src_path = os.path.join(SRC_MASK_512_DIR, fname)

        # 1. 读取 512 大图
        img_512 = Image.open(src_path)

        # 2. 循环切割成 4 张小图
        for i, box in enumerate(crops):
            # 切割
            patch = img_512.crop(box)

            # 构造目标文件名 (例如 0_0.png, 0_1.png ...)
            # 这里严格按照你的要求：{原文件名}_{索引}.png
            target_name = f"{name_pure}_{i}.png"
            target_path = os.path.join(DST_LABEL_256_DIR, target_name)

            # 3. 覆盖保存 (相当于删掉旧的黑图，写入新的好图)
            patch.save(target_path)
            count += 1

    print(f"\n修复完成！共覆盖/生成了 {count} 张 256x256 的标签图片。")
    print("现在 DSIFN-CD-256/label 里的测试集标签应该是正常的了。")


if __name__ == '__main__':
    fix_test_masks()