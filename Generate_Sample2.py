import os
import numpy as np
from PIL import Image

def gen_datasets(bg_path, minions_path, img_path, label_path):
    count = 0
    with open(label_path, "w") as f:
        for filename in os.listdir(bg_path):
            bg_img = Image.open("{0}/{1}".format(bg_path, filename))
            bg_img = bg_img.convert("RGB")
            bg_img = bg_img.resize((224, 224))
            bg_img.save("{0}/{1}.png".format(img_path, count))  # 保存背景图片
            f.write("{}.png {} {} {} {} {}\n".format(count, 0, 0, 0, 0, 0))  # 在txt文件写入背景图片标签

            count += 1
            name = np.random.randint(1, 21)
            minions_img = Image.open("{}/{}.png".format(minions_path, name))

            new_w = np.random.randint(50, 100)
            new_h = np.random.randint(50, 100)
            resize_img = minions_img.resize((new_w, new_h))  # 随机缩放
            rot_img = resize_img.rotate(np.random.randint(-45, 45))  # 经过处理后得到的小黄人的图片

            paste_x1 = np.random.randint(0, 224-new_w)
            paste_y1 = np.random.randint(0, 224-new_h)

            r, g, b, a = rot_img.split()
            bg_img.paste(rot_img, (paste_x1, paste_y1), mask=a)  # 背景图片上粘贴小黄人
            paste_x2 = paste_x1 + new_w
            paste_y2 = paste_y1 + new_h

            bg_img.save("{}/{}.png".format(img_path, count))  # 保存合成图片
            f.write("{}.png {} {} {} {} {}\n".format(
                count, 1, paste_x1, paste_y1, paste_x2, paste_y2))  # 在txt文件写入合成图片标签

            count += 1

            if count == 1500:
                print(count)
                break

if __name__ == '__main__':
    # 背景图片路径
    bg_img1 = r"D:\Train_Data_bg"
    bg_img2 = r"D:\PycharmProjects\2020-09-08-minions_reg\Dataset\Bg_Image_train"
    bg_img3 = r"./Dataset/Bg_Image_test"

    minions_img = r"./Dataset/Minions_Image"  # 小黄人图片路径

    train_img = r"./Dataset/Train_Data"  # 合成图片
    validate_img = r"./Dataset/Validate_Data"
    test_img = r"./Dataset/Test_Data"

    train_label = r"./Target/train_label.txt"  # 训练图片标签
    validate_label = r"./Target/validate_label.txt"
    test_label = r"./Target/test_label.txt"

    # gen_datasets(bg_img1, minions_img, train_img, train_label)
    gen_datasets(bg_img2, minions_img, validate_img, validate_label)
    # gen_datasets(bg_img3, minions_img, test_img, test_label)



