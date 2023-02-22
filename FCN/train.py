import time

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from VGG import *
from fcn import *
#全局变量
batch_size = 18  #每次喂入的数据量
num_print = 100  #num_print=int(5000//batch_size//4)
epoch_num = 30   #总迭代次数
lr = 0.01
step_size =10   #每n次epoch更新一次学习率

#获取数据 数据增强，归一化
def transforms_RandomHorizontalFlip():

    #ToTensor() 使图片数据转换为tensor张量， 这个过程包含了归一化， 图像数据从0~255 压缩到0~1，这个函数必须在Normalize之前使用。
    #实现原理，即针对不同类型进行处理，原理即各值除以255，最后通过torch.from_numpy 将PIL Image或者 numpy.ndarray() 针对具体类型转成torch.tensor()数据类型

    #Normalize() 是归一化进程，ToTensor()的作用是将图像数据为(0,1)之间的张量，Normalize()则使用公式(x-mean)/std
    #将每个元素分布到(-1,1).  归一化后数据转为标准格式,
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])  #mean=（0.485,0.456,0.406）， std=（0.229,0.224,0.225）
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])

    train_dataset = datasets.CIFAR10(root='./data/cifar10',train=True,transform=transform_train,download=True)
    test_dataset = datasets.CIFAR10(root='./data/cifar10',train=False,transform=transform,download=True)

    return train_dataset,test_dataset

#数据增强：随即翻转
train_dataset,test_dataset = transforms_RandomHorizontalFlip()

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True)

#选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#模型实例化
vgg = VGG16().to(device)


#多分类情况之下 一般使用交叉熵
criterion = nn.CrossEntropyLoss()
'''
param(iterable) - 待优化参数的iterablle或者定义了参数的dict
lr(float):学习率

momentum(float)-动量因子
weight_decay(float):权重衰减，使用目的是防止过拟合.在损失函数中,weight decay 是放在正则项前面的一个系数,正则项一般指示模型的复杂度
所以weight decay的作用是调节模型复杂度对损失函数的影响,若weight decay很大,则复杂的模型损失函数的值也就大。

optimizer.param_group:是长度为2的list,其中的元素是两个字典enumerate
optimzer.param_group:长度为6的字典,包括['amsgrad','params','lr','weight_decay',eps']
optimzer.param_group:表示优化器状态的一个字典
'''
optimizer = optim.SGD(vgg.parameters(),lr=lr,momentum=0.8,weight_decay=0.001)



'''
scheduler 就是为了调整学习率设置的，我这里设置的gamma衰减率为0.5，step_size为10，也就是每10个epoch将学习率衰减至原来的0.5倍。

optimizer(Optimizer):要更改学习率的优化器
milestones(list):递增的list,存放要更新的lr的epoch
gamma:(float):更新lr的乘法因子
last_epoch:：最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，这个值就等于加载的模型的epoch。默认为-1表示从头开始训练，即从epoch=1
'''
schedule = optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=0.5,last_epoch=-1)

#训练
loss_list = []  #为了后续画出损失图
start =time.time()

#train
for epoch in range(epoch_num):
    ww=0
    running_loss = 0.0
    #0是对i的给值（循环次数从0开始计数还是从1开始计数的问题）：
    for i,(inputs,labels) in enumerate(train_loader,0):   #enumerate(sequence,[start=0])
        #将数据从train_loader中读出来，一次读取样本32个
        inputs,labels = inputs.to(device),labels.to(device)  #把数据从cpu加遭到gpu
        #用于梯度清零
        optimizer.zero_grad()

        outputs = vgg(inputs)

        loss = criterion(outputs,labels).to(device)

        #反向传播
        loss.backward()
        #对反向传播以后的目标函数进行优化
        optimizer.step()

        running_loss += loss.item()
        loss_list.append(loss.item())

        if(i+1) % num_print == 0:
            print('[%d epoch,%d]  loss:%0.6f' %(epoch+1,i+1,running_loss/num_print))
            running_loss = 0.0
    lr_1 = optimizer.param_groups[0]['lr']
    print("learn_rate:%0.15f"%lr_1)
    schedule.step()
end=time.time()
print("time:{}".format(end-start))

#测试
#由于测试不需要进行梯度更新 于是进行测试模式


#由于训练集不需要梯度更新,于是进入测试模式
vgg.eval()
correct=0.0
total=0
with torch.no_grad(): #训练集不需要反向传播
    print("=======================test=======================")
    for inputs,labels in test_loader:
        inputs,labels=inputs.to(device),labels.to(device)
        outputs=vgg(inputs)

        pred=outputs.argmax(dim=1)  #返回每一行中最大值元素索引
        total+=inputs.size(0)
        correct+=torch.eq(pred,labels).sum().item()

print("Accuracy of the network on the 10000 test images:%.2f %%" %(100*correct/total) )
print("===============================================")

















