import os
import shutil

source_path = r"H:\U202115117\data"
train_dir = os.path.join(source_path, "train")
test_dir = os.path.join(source_path, "val")
train_dir_list = os.listdir(train_dir)
for dir in train_dir_list:
    category_dir_path = os.path.join(train_dir, dir)
    image_file_list = os.listdir(category_dir_path)
    num = int(0.1 * len(image_file_list))

    # 移动10%文件到对应目录
    for i in range(num):
        shutil.move(os.path.join(category_dir_path, image_file_list[i]),
                    os.path.join(test_dir, dir, image_file_list[i]))