import numpy as np
import torch
import torch.utils.data as data
import model
from torch.autograd import Variable
import coder
import os
import cv2

class Mydataset(data.Dataset):

    def __init__(self,data_path, class_path):
        self.data = np.load(data_path)
        length = self.data.shape[0]
        begin = int(length/9*8)
        self.data = self.data[begin:]
        self.classes = np.load(class_path)[begin:]

    def getitem(self, idx):
        return self.data[idx], self.classes[idx]

    def len(self):
        return self.data.shape[0]


def test(batch,result,need_softmax = True,name = 1):

    img = coder.decode(batch,result,need_softmax=need_softmax)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    if need_softmax:
        cv2.imwrite("./eval/eval"+ str(name) + ".png",img)
    else:
        cv2.imwrite("gt.png", img)

dset = Mydataset("./test_set.npy" ,"test_class.npy")

net = model.Net()
# load_name = r'F:\projects\cvprj\model\1572\test.cpk'
load_name = r'test-2.cpk'
if os.path.isfile(load_name) :
    net.load_state_dict(torch.load(load_name))
    sum_a = 0
    net.eval()
    MAX = 150
    with torch.no_grad():
        for i in range(0,MAX):
            print(i)
            inputs, classes= dset.getitem(i)
            inputs = inputs[:,:,0:1]
            inputs = torch.Tensor([inputs])
            inputs = inputs.permute(0,3,1,2)
            result, classifacation = net(inputs)
            v, idx = torch.max(classifacation, dim=1)
            acc = torch.sum(torch.eq(idx, classes)).cpu().detach().item()
            sum_a += acc
            # sum_a /= MAX
            testinput = inputs[0:1].permute(0,2,3,1).numpy()
            testresult = result[0:1].permute(0,2,3,1).numpy()
            test(testinput,testresult, True,name = i)
            inputs, classes = dset.getitem(i)
            img = cv2.cvtColor(inputs, cv2.COLOR_LAB2BGR)
            cv2.imwrite("./eval/gt"+str(i)+".png",img)
        sum_a /= MAX
        
        print(sum_a)