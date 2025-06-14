import pandas as pd
import os
import shutil
from pathlib import Path

# 读取训练数据集
df = pd.read_excel("NLP_processed_dataset.xlsx", sheet_name="Sheet1")

# 配置路径
src_dir = Path("data/processed_pro_1")  # 原始图片目录
dest_dir = Path("data/classified_single-pro")  # 分类输出目录

# 定义目标标签列
LABEL_COLUMNS = ['right-N', 'right-D', 'right-G', 'right-C', 'right-A', 'right-H', 'right-M', 'right-O']


def classify_images():
    # 遍历数据集每一行
    for _, row in df.iterrows():
        # 处理图像
        # if pd.notna(row["integral-Fundus-aug2"]):
        #     process_image(row["integral-Fundus-aug2"], row[LABEL_COLUMNS])

        # 处理右眼图像
        if pd.notna(row["right-Fundus"]):
            process_image(row["right-Fundus"], row[LABEL_COLUMNS])


def process_image(filename, labels):
    src_path = src_dir / filename

    # 检查源文件是否存在
    if not src_path.exists():
        print(f"警告：文件 {filename} 不存在")
        return

    # 获取激活的标签
    active_labels = [col for col, val in labels.items() if val == 1]

    # 复制到所有对应标签目录
    for label in active_labels:
        dest_path = dest_dir / label / filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # 避免重复复制
        if not dest_path.exists():
            shutil.copy(src_path, dest_path)
            print(f"已复制 {filename} -> {label}/")


if __name__ == "__main__":
    classify_images()
    print("分类完成！")