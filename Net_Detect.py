import torch
from Net_Model import Net
import os
import PIL.Image as pimg
import PIL.ImageFont as Font
import PIL.ImageDraw as draw
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
from sklearn.metrics import r2_score,explained_variance_score

test_path = r"D:\PycharmProjects\2020-09-08-minions_reg\Dataset\Test_Data"
params_path = r"D:\PycharmProjects\2020-09-08-minions_reg\Params"
font_path = r"D:\PycharmProjects\2020-09-08-minions_reg\Dataset\ARIALNB.TTF"

net = Net()
net.load_state_dict(torch.load("./Params/273.pth"))
net.eval()

for file in os.listdir(test_path):

# for file in glob.glob(r"D:\PycharmProjects\2020-09-08-minions_reg\Dataset\Test_Data\*.png"):
    # img = pimg.open(file)

    img = pimg.open("{0}/{1}".format(test_path, file))  # D:\PycharmProjects\2020-09-08-minions_reg\Dataset\Test_Data  0.png
    # img_array = pimg.open(os.path.join(test_path,file))
    img_array = (np.array(img)/255-0.5)/0.5
    trans_array = np.transpose(img_array, [2, 0, 1])

    # 转成tensor数据类型，并且增加一个维度方便传进网络
    tensor_array = torch.from_numpy(trans_array)
    input_array = torch.unsqueeze(tensor_array, dim=0)#增加一个维度
    out_array = input_array.float()

    # print(cuda_array.shape)
    #
    # print(cuda_array.dtype)
    # print(cuda_array.tolist())
    out1, out2 = net(out_array)

    coord_out = out1.cpu().data.numpy()
    # print(coord_out)

    c_out = out2.cpu().data.numpy()

    out_x1 = coord_out[0][0] * 224
    out_y1 = coord_out[0][1] * 224
    out_x2 = coord_out[0][2] * 224
    out_y2 = coord_out[0][3] * 224
    out_confidence = c_out[0][0]
    # print(out_x1)
    # exit()

    print("output_coord:", out_x1, out_y1, out_x2, out_y2)
    print("output_confidences:", out_confidence)
    # print(img_array)

    # 利用PIL显示图片
    img = pimg.fromarray(np.uint8((img_array*0.5+0.5)*255))
    imgdraw = draw.ImageDraw(img)

    imgdraw.rectangle((out_x1, out_y1, out_x2, out_y2), outline="red")
    font = Font.truetype(font_path, size=10)

    imgdraw.text(xy=(out_x1, out_y1), text=str("confident：{:.2}".format(out_confidence)), fill="red", font=font)
    plt.title("测试")
    plt.imshow(img)
    plt.pause(0.1)

    # 利用opencv显示图片
    # img = cv2.imread(file)
    # cv2.rectangle(img, (int(out_x1), int(out_x2)), (int(out_x2), int(out_y2)), [0, 0, 255], 1)
    # cv2.imshow("测试图片", img)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()



