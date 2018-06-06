
# coding: utf-8

# In[1]:


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


# In[2]:


sequence_length = 378
input_size = 500
hidden_size = 128
num_layers = 2 # ?
num_classes = 2
batch_size = 50
num_epochs = 10
learning_rate = 0.001


# In[3]:


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):    
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes) 
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out


# In[4]:


class faceFeatures(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.features_frame = pd.read_csv(root_dir+csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.features_frame)
 
    def __getitem__(self):
        features = self.features_frame.iloc[:, 3:-1].values
        diff = 500 - features.shape[0]
        features_zeroes = np.zeros((diff, features.shape[1]))
#         print (features_zeroes.shape)
        features = np.append(features, features_zeroes, axis = 0)
        label = self.features_frame.iloc[1,-1]
#         print (features.shape)
        return (features, label)


# In[5]:


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
            self._input = np.vstack((data.__getitem__()[0], self._input))
            self._label = np.vstack((data.__getitem__()[1], self._label))
            
        self._input = self._input.reshape((len(self.csv_files), 500, 378))
        self._label = self._label.reshape((len(self.csv_files), 1, 1))
        print (self._input.shape)
        return (self._input, self._label)

    # Get the tensor by index
    def __getitem__(self, idx):
        frame_name = self.csv_files[idx]
        frame_features = self._input[idx]
        frameself._label = self._label[idx]
        return (frame_features, frameself._label) 


# In[6]:


print ("Training data preprocessing....")
# csv_files = ["300_P_new/300_P1.csv", "302_P_new/302_P2.csv","300_P_new/300_P2.csv"]
csv_files_train = []
for filename in os.listdir("./dataset/USC/bad_frames/frames/frames_with_label"):
    if filename != "test": 
        for framefile in os.listdir("./dataset/USC/bad_frames/frames/frames_with_label/"+filename):
            file = filename + "/" + framefile
            csv_files_train.append(file)
# print (csv_files)
_input = np.zeros((500, 378), dtype="float32")
_label = np.zeros((1,1), dtype="int64")

data = concatFrames(root_dir = "./dataset/USC/bad_frames/frames/frames_with_label/", csv_files = csv_files_train, _input = _input, _label = _label)
_input, _label = data._concat_()
_input_train = torch.Tensor(np.array(_input))
_label_train = torch.Tensor(np.array(_label))
# print (data.__getitem__(0))


# In[7]:


print ("Test data preprocessing....")

csv_files_test = []
for filename in os.listdir("./dataset/USC/bad_frames/frames/frames_with_label"):
    if filename == "test": 
        for framefile in os.listdir("./dataset/USC/bad_frames/frames/frames_with_label/"+filename):
            file = filename + "/" + framefile
            csv_files_test.append(file)
            
# print (len(csv_files_test))
_input_ = np.zeros((500, 378), dtype="float32")
_label_ = np.zeros((1,1), dtype="int64")
data_test = concatFrames(root_dir = "./dataset/USC/bad_frames/frames/frames_with_label/", _input = _input_, _label = _label_, csv_files = csv_files_test)
_input_test, _label_test = data_test._concat_()
_input_test = torch.Tensor(np.array(_input_test))
_label_test = torch.Tensor(np.array(_label_test))
print (_input_test.shape)
# print (data.__getitem__(0))


# In[8]:


model = RNN(input_size, hidden_size, num_layers, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


# In[11]:


train = data_utils.TensorDataset(_input_train, _label_train)
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

test = data_utils.TensorDataset(_input_test, _label_test)
test_loader = data_utils.DataLoader(test, shuffle=True)
# total_step = len(train_loader)
# for epoch in range(num_epochs):
# i is the counter, ith batch, j is the value of batch
for i,(feature, label) in enumerate(train_loader):
        feature = feature.reshape(-1, sequence_length, input_size)
        print (feature.shape)
        # Forward pass
        outputs = model(feature)
        loss = criterion(outputs, label)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        

