import numpy as np
import torch
import torch.utils.data as data
import model
from torch.autograd import Variable
import coder
import os
import cv2

class Mydataset(data.Dataset):

    def __init__(self,data_path,label_path, class_path):
        self.data = np.load(data_path)
        self.classes = np.load(class_path)
        self.labels = np.load(label_path)
        # self.labels = np.ones([1500, 64,64,313])
        # self.labels = [encode(self.alldata[i:i+1,:,:,1:])[0] for i in range(len(self.alldata))]


    def getitem(self, idx):

        return self.data[idx],coder.encode([self.labels[idx]])[0]

    def get_batch(self,start,size):
        return self.data[start:start+size],self.labels[start:start+size],self.classes[start:start+size]
    def len(self):
        return self.data.shape[0]


def test(batch,result,need_softmax = True):

    img = coder.decode(batch,result,need_softmax=need_softmax)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    if need_softmax:
        cv2.imwrite("test.png",img)
    else:
        cv2.imwrite("gt.png", img)

dset = Mydataset("./train_set_data.npy", "train_set_label.npy" ,"train_class.npy")


net = model.Net()
net.cuda()

optimizer = torch.optim.Adam(net.parameters(),lr = 0.0001,weight_decay=1e-7)
#optimizer = torch.optim.SGD(net.parameters(),lr = 0.005,weight_decay=0,momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.999)


load_name = 'test-2.cpk'
save_name = 'test-2.cpk'
gpu = True

if os.path.isfile(load_name) :
    net.load_state_dict(torch.load(load_name))


loss_list_class = []
loss_list_color = []
for epoch in range(100000):
    if save_name:
        torch.save(net.state_dict(),save_name)
        np.save("loss_list_class", loss_list_class)
        np.save("loss_list_color", loss_list_color)
    sum_loss = 0.0

    i = 0
    batch = 40
    loss_classifacation = torch.nn.CrossEntropyLoss()
    while i*batch + batch <= dset.len():
        net.train()
        inputs,labels, classes = dset.get_batch(i*batch,batch)
        #inputs, labels = dset.get_batch(1, 1)

        inputs,labels = torch.Tensor(inputs),torch.Tensor(labels)
        classes = torch.LongTensor(classes)

        if gpu :
            inputs = inputs.cuda()
            labels = labels.cuda()
            classes = classes.cuda()

        inputs = inputs.permute(0,3,1,2)
        optimizer.zero_grad()
        result , classifacation = net(inputs)


        loss_class = loss_classifacation(classifacation,classes)
        loss_color = net.loss(result, labels)
        loss = loss_class*1e-1 + loss_color

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1)
        optimizer.step()



        if i%1 == 0:
            v, idx = torch.max(classifacation, dim=1)
            acc = torch.sum(torch.eq(idx, classes)).cpu().detach().item()
            # print(type(acc))
            acc /= batch
            return_class_loss = loss_class.detach().item()
            return_color_loss = loss_color.detach().item()
            print(epoch,i,return_class_loss,return_color_loss, str(acc*100)+'%')
            loss_list_class.append(return_class_loss)
            loss_list_color.append(return_color_loss)
        if i == 0:
            testinput = inputs[0:1].permute(0,2,3,1).cpu().detach().numpy()
            label = labels[0:1].cpu().detach().numpy()
            testresult = result[0:1].permute(0,2,3,1).cpu().detach().numpy()
            test(testinput,label,False)
            test(testinput,testresult, True)

        i = i+1


