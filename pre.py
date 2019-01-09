from coder import encode
import numpy as np


images = np.load("images.npy")
length = images.shape[0]
labels = np.load("class_labels.npy")
index = np.array([i for i in range (0,length)])
np.random.shuffle(index)
train_data = images[index[0:int(length*0.9)]]
train_labels = labels[index[0:int(length*0.9)]]
test_data = images[index[length-int(length*0.9):]]
test_labels = labels[index[length-int(length*0.9):]]

np.save("train_set", train_data)
np.save("train_class", train_labels)
np.save("test_set", test_data)
np.save("test_class",test_labels)

img = np.load("train_set.npy")
label = img[:,:,:,1:]
data = img[:,:,:,0:1]
np.save("train_set_data", data)
del img,data
labels = []
for i in range(len(label)):
    print(i)
    labels.append(encode(label[i:i+1,:,:,:])[0])
del label
print("done")
np.save("train_set_label", labels)
# data = img[:100, :,:,0:1]
# np.save("train_set_data_1", data)
