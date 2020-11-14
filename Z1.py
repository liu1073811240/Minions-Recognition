import torch
from Net_Model import Net
import os
import PIL.Image as pimg
import PIL.ImageFont as Font
import PIL.ImageDraw as draw
import matplotlib.pyplot as plt
import numpy as np

test_path = r"D:\liuhui\Project_Minions -\Dataset\Test_Data"
params_path = r"D:\liuhui\Project_Minions -\Params"
font_path = r"D:\liuhui\Project_Minions -\Dataset\msyh.ttf"

net = Net()
net.load_state_dict(torch.load("{0}/{1}".format(params_path,"ckpt.pth")))
net.eval()
for file in os.listdir(test_path):
    img = pimg.open("{0}/{1}".format(test_path,file))
    # img_array = pimg.open(os.path.join(test_path,file))
    img_array = (np.array(img)/255-0.5)/0.5
    trans_array = np.transpose(img_array,[2,0,1])
    tensor_array = torch.from_numpy(trans_array)
    input_array = torch.unsqueeze(tensor_array,dim=0)
    cuda_array = input_array.cuda().float()

    print(cuda_array.shape)

    print(cuda_array.dtype)

    # out_confidence = c_out[0]
    # out_confidence[0] c_out[0][0]