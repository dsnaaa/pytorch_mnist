import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np
from PIL import Image
from train import CNN
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 200    # 分批训练数据、每批数据量
DOWNLOAD_MNIST = False 


test_dataset = datasets.MNIST(
    root='C:\code\pytorch\pymnist',
    train=False,        #download test data
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



def test():
    with torch.no_grad():
        net = CNN(1,10)
        net.load_state_dict(torch.load('D:/bd/model.tar'))           
        eval_acc = 0
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device)
            out = net(img)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.item()
        print('Acc: {:.2f}%'.format(eval_acc/len(test_dataset)*100))


    
if __name__ == '__main__':
    test()