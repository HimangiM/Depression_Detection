
# coding: utf-8

# In[144]:


from skimage import io
import sys
import torch
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data.dataset import Dataset
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import time
import math
import pickle as pkl


# In[145]:


start = time.time()


# In[146]:


sequence_length = 300
input_size = 378
hidden_size = 128
num_layers = 2
num_classes = 2 # Depressed or not depressed
batch_size = 50
num_epochs = 10
learning_rate = 0.01
rec_dropout = 0.05
feature_len = 378


# In[147]:


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):    
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout = rec_dropout, batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes) 
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out


# In[148]:


class faceFeatures(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.features_frame = pd.read_csv(root_dir + csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.features_frame)
 
    def __getitem__(self):
        features = []
        label = []
        all_features = self.features_frame.iloc[:, 3:].values
        diff = sequence_length - all_features.shape[0]
        if (diff < 0):
            rows = all_features.shape[0]
            row_idx = 0
            while(row_idx + sequence_length <= rows):
                features.append(all_features[row_idx:row_idx+sequence_length,:])
                row_idx = row_idx + sequence_length
                
            new_diff = sequence_length - (all_features.shape[0] - row_idx)
            print (new_diff)
            features_zeroes = np.zeros((new_diff, all_features.shape[1]))
            second_features = np.append(all_features[row_idx:all_features.shape[0],:], features_zeroes, axis = 0)
            features.append(second_features)
            for i in features:
                print (i.shape)
                label.append(self.features_frame.iloc[1,-1])
        else:
            features_zeroes1 = np.zeros((int(diff/2), all_features.shape[1]))
            features_zeroes2 = np.zeros((int(diff/2)+1, all_features.shape[1]))
            features = np.append(all_features, features_zeroes1, axis = 0)
            if (diff % 2 == 0):
                features = np.append(features, features_zeroes1, axis = 0)
            else:
                features = np.append(features, features_zeroes2, axis = 0)
            label = self.features_frame.iloc[1,-1]
#         print (features.shape)
#         print (features, label)
        return (features, label)


# In[149]:


# data = faceFeatures("./frames_10fps/", "305_P/305_P16.csv").__getitem__()
# print (data)


# In[150]:


class concatFrames(Dataset):
    # Initialize the list of csv files
    def __init__(self, root_dir, _input, _label, csv_files = []):
        self.csv_files = csv_files
        self.root_dir = root_dir
        self._input = _input
        self._label = _label
        
    # Create tensor of frames
    def _concat_(self):
#         self._input = np.zeros((500, 378), dtype="float32")
        data = faceFeatures(self.root_dir, self.csv_files[0])
        self._input = data.__getitem__()[0]
        self._label = data.__getitem__()[1]
        
        for i in range(1, len(self.csv_files)):
#             print (csv_files[i])
            data = faceFeatures(self.root_dir, self.csv_files[i])
            for f in data.__getitem__()[0]:
                self._input = np.vstack((f, self._input))
            for l in data.__getitem__()[1]:
                self._label = np.vstack((l, self._label))
            
        self._input = self._input.reshape((len(self.csv_files), sequence_length, feature_len))
        self._label = self._label.reshape((len(self.csv_files)))
        print (self._input.shape, self._label.shape)
        return (self._input, self._label)

    # Get the tensor by index
    def __getitem__(self, idx):
        frame_name = self.csv_files[idx]
        frame_features = self._input[idx]
        frame_label = self._label[idx]
        return (frame_features, frame_label) 


# In[151]:


# file = open("label_dict.pkl", "rb")
# label_dict = pkl.load(file)
# print (label_dict)


# In[152]:


print ("Training data preprocessing....")
# csv_files = ["300_P_new/300_P1.csv", "302_P_new/302_P2.csv","300_P_new/300_P2.csv"]
csv_files_train = []
for filename in os.listdir("./frames_10fps"):
    if filename != "test": 
        for framefile in os.listdir("./frames_10fps/"+filename):
            file = filename + "/" + framefile
            csv_files_train.append(file)
            
# print (csv_files_train)
_input = np.zeros((sequence_length, feature_len), dtype="float32")
_label = np.ones((1), dtype="int32")

data = concatFrames(root_dir = "./frames_10fps/", csv_files = csv_files_train[0:10], _input = _input, _label = _label)
_input, _label = data._concat_()
_input_train = torch.Tensor(np.array(_input))
_label_train = torch.Tensor(np.array(_label))
_label_train = (_label_train.type(torch.LongTensor))

# torch.save(_input_train, "input_train.pt")
# torch.save(_label_train, "label_train.pt")

# print (data.__getitem__(0))


# In[8]:


print ("Test data preprocessing....")

csv_files_test = []
for filename in os.listdir("./dataset/USC/bad_frames/frames/frames_with_label"):
    if filename == "test": 
        for framefile in os.listdir("./dataset/USC/bad_frames/frames/frames_with_label/"+filename):
            file = filename + "/" + framefile
            csv_files_test.append(file)
            
# print (len(csv_files_test))
_input_ = np.zeros((padding_size, feature_len), dtype="float32")
_label_ = np.zeros((1), dtype="int32")
data_test = concatFrames(root_dir = "./dataset/USC/bad_frames/frames/frames_with_label/", _input = _input_, _label = _label_, csv_files = csv_files_test)
_input_test, _label_test = data_test._concat_()
_input_test = torch.Tensor(np.array(_input_test))
_label_test = torch.Tensor(np.array(_label_test))
_label_test = (_label_test.type(torch.LongTensor))

torch.save(_input_test, "input_test.pt")
torch.save(_label_test, "label_test.pt")

print (_input_test.shape)
# print (data.__getitem__(0))


# In[9]:


model = RNN(input_size, hidden_size, num_layers, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


# In[12]:


_input_train = torch.load("input_train.pt")
_label_train = torch.load("label_train.pt")

_input_test = torch.load("input_test.pt")
_label_test = torch.load("label_test.pt")


train = data_utils.TensorDataset(_input_train, _label_train)
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

test = data_utils.TensorDataset(_input_test, _label_test)
test_loader = data_utils.DataLoader(test, shuffle=True)
# total_step = len(train_loader)

epoch_start = time.time()
loss = 0

for epoch in range(num_epochs):
# i is the counter, ith batch, j is the value of batch
    for i,(feature, label) in enumerate(train_loader):
            feature = feature.reshape(-1, sequence_length, input_size)
            print (feature.shape)
            # Forward pass
            outputs = model(feature)
            print (outputs.shape)
            print (label.shape)
            label = label.reshape(batch_size)
            loss = criterion(outputs, label)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print ("Loss")
            print (loss.item())
            loss = loss + loss.item()
        
    print ("Epoch time")
    print (time.time() - epoch_start)
    epoch_start = time.time()
    
print ("Mean loss")
print (loss/num_epochs)
        

with torch.no_grad():
    correct = 0
    total = 0
    
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total = total + labels.size(0)
        correct = correct + (predicted == labels).sum().item()
    
    print ("Test accuracy")
    print (correct/total)
        


# In[ ]:


f_time = time.time()-start
print (f_time)

