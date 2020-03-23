import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np
from PIL import Image
from train import CNN

net = CNN(1,10)
net.load_state_dict(torch.load('D:/bd/model.tar'))
input_image = 'D:/4.png'

im = Image.open(input_image).resize((28, 28))     #取图片数据
im = im.convert('L') #灰度图
im_data = np.array(im)

im_data = torch.from_numpy(im_data).float()

im_data = im_data.view(1, 1, 28, 28)
out = net(im_data)
_, pred = torch.max(out, 1)

print('预测的数字是：{}。'.format(pred))
