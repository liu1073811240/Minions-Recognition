import torch
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import os
import numpy as np
import PIL.Image as pimg


class Train_Data(Dataset):

    def __init__(self, root, txt_path):  # 定义路径和数据标准化
        self.files_name = []
        self.labels_data = []

        f = open(txt_path, "r")
        for line in f:  # 遍历训练集文件
            line = line.strip()
            img_name = line.split()

            img_path = os.path.join(root, img_name[0])
            self.files_name.append(img_path)
            # print(self.files_name)

            # print(img_name)

            c = float(img_name[1])
            x1 = float(img_name[2])/224  # 将标签数据转成float，归一化
            y1 = float(img_name[3])/224
            x2 = float(img_name[4])/224
            y2 = float(img_name[5])/224

            label_data=np.array([x1, y1, x2, y2, c]).astype(np.float32)  # 将标签数据转成float32, [0. 0. 0. 0. 0.]

            self.labels_data.append(label_data)  # 将标签存在一个列表中
            # print(self.labels_data)  # [array([0., 0., 0., 0., 0.], dtype=float32)...] 依次装标签进去

        self.labels_data = np.array(self.labels_data)  # 将标签转成numpy
        # print("=======")
        # print(self.labels_data)  # [[0. 0. 0. 0. 0. ]...], 获得图片标签的numpy矩阵
        f.close()

    def __len__(self):
        return len(self.files_name)  # 获得文件长度

    def __getitem__(self, index):
        file = self.files_name[index]  # 根据索引获得每个文件名路径
        # print(file)
        img_data = self.image_preprocess(pimg.open(file))  # 根据文件名路径打开图像数据

        xs = img_data  # 数据标准化处理
        ys = self.labels_data[index]  # 根据索引获得标签数据
        # print(ys)
        return xs, ys

    def image_preprocess(self, x):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])(x)

# 数据标准化过程

txt_path = r"D:\PycharmProjects\2020-09-08-minions_reg\Target\train_label.txt"
train_path = r"D:\PycharmProjects\2020-09-08-minions_reg\Dataset\Train_Data"
train_data = Train_Data(root=train_path, txt_path=txt_path)

# data = DataLoader(dataset=train_data,batch_size=10, shuffle=True)
# 
# for img,label in data:
#     print(img.dtype)
#     print(label.dtype)
#     print(img.size())
#     print(label.size())
