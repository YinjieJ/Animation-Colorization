import numpy as np
import torch
import torch.nn as nn
import math
from collections import OrderedDict



def conv3x3(input,ouput,stride=1):
    conv =  nn.Conv2d(input,ouput,stride=stride,padding=1,kernel_size=3)
    return nn.Sequential(conv,nn.ReLU())

def conv1x1(input,ouput,stride=1):
    conv =  nn.Conv2d(input,ouput,stride=stride,kernel_size=1)
    return nn.Sequential(conv,nn.ReLU())

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()


        self.conv1_1 = conv3x3(1,64,stride=2)
        self.conv1_2 = conv3x3(64,128)
        self.bn1=nn.BatchNorm2d(128)

        self.conv2_1= conv3x3(128,128,stride=2)
        self.conv2_2= conv3x3(128,256)
        self.bn2=nn.BatchNorm2d(256)

        self.conv3_1= conv3x3(256,256,stride=2)
        self.conv3_2= conv3x3(256,512)
        self.bn3= nn.BatchNorm2d(512)

        self.feature_conv1_1 = conv3x3(512,512,stride=2)
        self.feature_conv1_2 = conv3x3(512,512)
        self.feature_bn1  =nn.BatchNorm2d(512)
        self.feature_conv2_1 = conv3x3(512, 512, stride=2)
        self.feature_conv2_2 = conv3x3(512, 512)
        self.feature_bn2 = nn.BatchNorm2d(512)

        self.feature_fc1 = nn.Linear(512 * 8 *8,1024)
        self.active_fc1 = nn.Sigmoid()
        self.feature_fc2 = nn.Linear(1024, 512)
        self.active_fc2 = nn.Sigmoid()
        self.feature_fc3 = nn.Linear(512, 256)
        self.active_fc3 = nn.Sigmoid()

        self.classification_fc1 = nn.Linear(512,128)
        self.active_classification = nn.Sigmoid()
        self.classification_fc2 = nn.Linear(128,9)

        self.conv4_1= conv3x3(512,512)
        self.conv4_2= conv3x3(512,256)
        self.bn4= nn.BatchNorm2d(256)


        self.conv5_1= conv1x1(512,256)
        self.conv5_2= conv3x3(256,256)
        self.bn5 = nn.BatchNorm2d(256)

        self.upsample = nn.ConvTranspose2d(256,256,kernel_size=4,stride=2,padding=1)
        self.conv6_1 = conv3x3(256,256)
        self.conv6_2 = nn.Conv2d(256,313,kernel_size=1)


        self.softloss = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim = -1)


    def forward(self,x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.bn1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.bn2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.bn3(x)

        feture = self.feature_conv1_1(x)
        feture = self.feature_conv1_2(feture)
        feture = self.feature_bn1(feture)
        feture = self.feature_conv2_1(feture)
        feture = self.feature_conv2_2(feture)
        feture = self.feature_bn2(feture)

        feture = feture.view(-1,512 * 8 * 8)

        feture = self.feature_fc1(feture)
        feture = self.active_fc1(feture)
        feture = self.feature_fc2(feture)
        classifacation = self.active_fc2(feture)
        feture = self.feature_fc3(classifacation)
        feture = self.active_fc3(feture)

        classifacation = self.classification_fc1(classifacation)
        classifacation = self.active_classification(classifacation)
        classifacation = self.classification_fc2(classifacation)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.bn4(x)


        feture = feture.view(-1,256,1,1)

        feture = feture.repeat(1,1,32,32)

        x = torch.cat([x,feture],dim = 1)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.bn5(x)


        x = self.upsample(x)

        x = self.conv6_1(x)
        x = self.conv6_2(x)
        return x,classifacation

    def loss(self,logits,target):
        batchsize = logits.shape[0]
        logits = logits.permute(0, 2, 3, 1)


        logits = logits.contiguous().view(-1,313)
        target = target.view(-1,313)


        loss = - target * self.softloss(logits)


        return loss.sum()/batchsize/64/64



