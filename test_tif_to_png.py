import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# 防止超大图片报错
Image.MAX_IMAGE_PIXELS = None


def convert_test_masks(root_dir):
    # 定义源文件夹 (根据你的描述，test集里的标签在 mask_256 里)
    src_dir = os.path.join(root_dir, 'test', 'mask')
    # 定义目标文件夹
    dst_dir = os.path.join(root_dir, 'test', 'mask_png')

    if not os.path.exists(src_dir):
        print(f"错误：找不到源文件夹: {src_dir}")
        return

    # 创建目标文件夹
    os.makedirs(dst_dir, exist_ok=True)
    print(f"源文件夹: {src_dir}")
    print(f"目标文件夹: {dst_dir}")

    # 获取所有 TIF 文件
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.tif', '.tiff'))]
    print(f"找到 {len(files)} 个 TIF 文件，开始转换...")

    for fname in tqdm(files):
        src_path = os.path.join(src_dir, fname)

        # 1. 读取 TIF
        img = Image.open(src_path)

        # 2. 转为 numpy 数组
        img_np = np.array(img)

        # 【核心修复】
        # TIF 标签通常是 0/1 二值图，直接转 PNG 肉眼看就是全黑的。
        # 这里我们强制把所有大于 0 的值（即变化区域）设为 255（纯白）。
        img_np[img_np > 0] = 255

        # 3. 转回 PIL 图像 (确保是单通道灰度图 'L')
        img_png = Image.fromarray(img_np.astype(np.uint8), mode='L')

        # 4. 生成新的 PNG 文件名
        name_pure = os.path.splitext(fname)[0]
        dst_path = os.path.join(dst_dir, name_pure + '.png')

        # 5. 保存
        img_png.save(dst_path)

    print("\n转换完成！请去查看 DSIFN-CD/test/mask_png 文件夹。")


if __name__ == '__main__':
    # --- 请确认这里的路径是你 DSIFN-CD 的根目录 ---
    DSIFN_ROOT = r"E:\RemoteDatasets\DSIFN-CD"
    # -------------------------------------------

    convert_test_masks(DSIFN_ROOT)