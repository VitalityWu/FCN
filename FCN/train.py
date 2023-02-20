import torch
from torch import  nn
import numpy as np

#建立block层 blcok依次包含 conv  - bn  -relu
class Block(nn.Module):

    def __init__(self,in_ch,out_ch,kernel_size=3,padding=1,stride=1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,out_ch,kernel_size=kernel_size,padding=padding,stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch) #归一化处理
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self,x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out




#建立layer加入很多Block
def make_layers(in_channels,layer_list):
    layers=[]
    for out_channels in layer_list:
        layers +=[Block(in_channels,out_channels)]
        in_channels = out_channels
    return  nn.Sequential(*layers)

class Layer(nn.Module):
    def __init__(self,in_channels,layer_list) -> None:
        super(Layer,self).__init__()
        self.layer = make_layers(in_channels,layer_list)

    def forward(self,x):
        out = self.layer(x)
        return out


class VGG_fcn32s(nn.Module):
    '''
    将VGG model 改为 FCN-32s
    '''

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,padding=100)  #pading=100, 传统VGG为1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer1=Layer(64,[64])  #第一组
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2) #降采样/2
        self.layer2 = Layer(64,[128,128])  #第二组
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2) #降采样 /4
        self.layer3 = Layer(128,[256,256,256,256]) #第三组
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2) #降采样 /8
        self.layer4 = Layer(256,[512,512,512,512])  #第四组
        self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2) #降采样 /16
        self.layer5 = Layer(512,[512,512,512,512])
        self.pool5 = nn.MaxPool2d(kernel_size=2,stride=2)  #降采样  /32


        #modify to be compatible with segmentation and classification
        #self.fc6 = nn.Linear(512*7*7,4096)#全连接层

        self.fc6 = nn.Conv2d(512,4096,7)  #padding=0
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout()

        self.fc7 = nn.Conv2d(4096,4096,1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout()

        self.score = nn.Conv2d(4096,n_class,1)

        self.upscore = nn.ConvTranspose2d(n_class,n_class,64,32)  #上采样32倍


