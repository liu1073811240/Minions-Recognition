import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pimg
import PIL.ImageFont as Font
import PIL.ImageDraw as draw
from Dataset_train2 import train_data
from Dataset_validate2 import validate_data
from Net_Model import Net
from tensorboardX import SummaryWriter
from sklearn.metrics import r2_score, explained_variance_score
from tensorboardX import SummaryWriter

plt.rcParams['font.sans-serif'] = ['SimHei']
params_path = r"D:\PycharmProjects\2020-09-08-minions_reg\Params\s1.pth"

font_path = r"D:\PycharmProjects\2020-09-08-minions_reg\Font\ARIALNB.TTF"
writer = SummaryWriter("./logs")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

net = Net().to(device)
net.load_state_dict(torch.load("./Params/273.pth"))

coord_loss = nn.MSELoss()#coord坐标点损失使用均方差函数来算
c_loss = nn.BCELoss()#confidence置信度损失使用二分类交叉熵函数来算
optimizer = optim.Adam(net.parameters(), lr=1e-3)

plt.ion()
a = []
b = []
d = []
e = []
net.train()
for epoch in range(1000):
    train_loader = DataLoader(dataset=train_data,batch_size=20, shuffle=True)
    validate_loader = DataLoader(dataset=validate_data,batch_size=10, shuffle=True)
    train_loss = 0
    # for i, (img, label) in enumerate(train_loader):
    #     img = img.to(device)  # torch.Size([20, 3, 224, 224])
    #     label = label.to(device)  # torch.Size([20, 5])
    #
    #     out1, out2 = net(img)
    #     # print(out1.shape)  # torch.Size([20, 4])
    #     # print(out2.shape)  # torch.Size([20, 1])
    #
    #     label1 = label[:, :4]  # 四个坐标值
    #     # print(label1.shape)  # torch.Size([20, 4])
    #     label2 = label[:, 4:]  # 一个置信度
    #     # print(label2.shape)  # torch.Size([20, 1])
    #
    #     loss1 = coord_loss(out1, label1)
    #     loss2 = c_loss(out2, label2)
    #     loss = loss1+loss2  # 坐标值和置信度的总损失
    #     train_loss += loss.item() * label.size(0)
    #     # train_loss += loss.detach().cpu().numpy() * label.size(0)
    #
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #     if i % 100 == 0:
    #
    #         print("epoch:{}, i:{}, train_loss:{:.6f}".format(epoch, i, loss.item()))
    # torch.save(net.state_dict(), "./Params/{}.pth".format(epoch))

    mean_train_loss = train_loss / len(train_data)
    eval_loss = 0
    label_list_coord = []
    output_list_coord = []
    label_list_con = []
    output_list_con = []
    for i, (img, label) in enumerate(validate_loader):

        img = img.to(device)
        label = label.to(device)
        # print(img.shape)  # torch.Size([10, 3, 224, 224])
        # print(label.shape)  # torch.Size([10, 5])

        _out1, _out2 = net(img)  # 形状分别为torch.Size([10, 4])、torch.Size([10, 1])

        _label1 = label[:, :4]  # torch.Size([10, 4])
        print(_label1)
        print(_out1)
        _label2 = label[:, 4:]  # torch.size([10, 1])

        _loss1 = coord_loss(_out1, _label1)  # 十张验证图片做坐标值损失
        _loss2 = c_loss(_out2, _label2)  # 十张验证图片做置信度损失
        _loss = _loss1 + _loss2  # 总损失

        eval_loss += _loss.item() * label.size(0)  # 放到cpu上运行
        # eval_loss += _loss.detach().cpu().numpy() * label.size(0)


        label_list_coord.append(_label1.cpu().numpy().reshape(-1))
        print(label_list_coord)

        output_list_coord.append(_out1.data.cpu().numpy().reshape(-1))
        print(output_list_coord)
        exit()

        label_list_con.append(_label2.cpu().numpy().reshape(-1))
        output_list_con.append(_out2.data.cpu().numpy().reshape(-1))

        coord_label = _label1.cpu().data.numpy()
        coord_out = _out1.cpu().data.numpy()

        c_label = _label2.cpu().data.numpy()
        c_out = _out2.cpu().data.numpy()
        # print(_out1)
        # print(coord_out)
        # print(coord_out[0][0])
        # print(c_out)

        # 将输出的坐标值乘以224转回为原来的图片大小的尺寸,拿到十张图片的第一张图片输出的坐标值和置信度
        out_x1 = coord_out[0][0] * 224
        out_y1 = coord_out[0][1] * 224
        out_x2 = coord_out[0][2] * 224
        out_y2 = coord_out[0][3] * 224
        out_confidence = c_out[0][0]

        # 拿到十张图片的第一张图片目标的坐标值和置信度
        label_x1 = coord_label[0][0] * 224
        label_y1 = coord_label[0][1] * 224
        label_x2 = coord_label[0][2] * 224
        label_y2 = coord_label[0][3] * 224
        label_confidence = c_label[0][0]


        # print("label_coord:", label_x1, label_y1, label_x2, label_y2)
        # print("output_coord:", out_x1, out_y1, out_x2, out_y2)
        # print("label_confidences:", label_confidence)
        # print("output_confidences:", out_confidence)

        if i % 10 == 0:
            # plt.clf()

            print('epoch: {}, train_loss: {:.3}, validate_loss: {:.3}'.format(epoch,_loss.item(),_loss.item()))  # 损失值显示3位

            arr = (img[0].cpu().numpy()*0.5+0.5) * 255 # 将图片数据转numpy数据，并转到0-255之间

            # print(np.shape(arr))#(3, 224, 224)

            array = np.transpose(arr, [1, 2, 0])
            # print(np.shape(array))#(224, 224, 3)
            img = pimg.fromarray(np.uint8(array))
            imgdraw = draw.ImageDraw(img)

            imgdraw.rectangle((label_x1, label_y1, label_x2, label_y2), outline="blue")
            imgdraw.rectangle((out_x1, out_y1, out_x2, out_y2), outline="red")

            font = Font.truetype(font_path, size=10)
            imgdraw.text(xy=(label_x1, label_y1), text=str(label_confidence), fill="blue", font=font)
            imgdraw.text(xy=(out_x1, out_y1), text=str("{:.2f}".format(out_confidence)), fill="red", font=font)
            plt.imshow(img)
            plt.pause(0.1)

    mean_eval_loss = eval_loss / len(validate_data)

    # 画损失曲线，验证曲线观察是否过拟合
    # plt.clf()
    # plt.figure()
    # # plt.subplot(2, 1, 1)
    # plt.title("观察是否过拟合")
    # a.append(epoch)
    # b.append(mean_train_loss)
    # plt.plot(a, b, c="r", label="train_loss")

    # plt.subplot(2, 1, 2)
    # d.append(epoch)
    # e.append(mean_eval_loss)
    # plt.plot(d, e, c="b", label="validate_loss")
    # plt.legend()
    #
    # plt.xlabel("epoch")
    # plt.ylabel("损失值")
    # plt.pause(1)
    # plt.close()

    # 评估坐标值和置信度
    r2 = r2_score(label_list_coord, output_list_coord)
    var = explained_variance_score(label_list_coord, output_list_coord)
    print("r2_score评估坐标值：", r2)
    # print("可解释性方差评估坐标值：", var)
    _r2 = r2_score(label_list_con, output_list_con)
    _var = explained_variance_score(label_list_con, output_list_con)
    print("r2_score评估置信度：", _r2)
    # print("可解释性方差评估置信度：", _var)

    writer.add_scalars("loss", {"mean_train_loss": mean_train_loss, "test_loss": mean_eval_loss}, epoch)

plt.ioff()

