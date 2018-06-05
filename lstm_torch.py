from skimage import io
import sys
import torch
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np

sequence_length = 500
input_size = 500
hidden_size = 128
num_layers = 1
num_classes = 2
batch_size = 50
num_epochs = 10
learning_rate = 0.01

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):    
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.LSTM = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes) 
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out


model = RNN(input_size, hidden_size, num_layers, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


# Training data
print ("Training data preprocessing....")
# create input numpy 2d array for tensor
_input = np.zeros([500,378], dtype='float')
# read file
frame = pd.read_csv("./frames/frames_new/300_P_new/300_P1.csv")
# get the features which start from 3rd column, skip the label column at end
features = np.array(frame.iloc[:,3:-1].values, dtype = 'float')
# assuming question file has 100 frames
diff = 500 - features.shape[0]
features_zeroes = np.zeros((diff, features.shape[1]))
print (features_zeroes.shape)
# adjust the feature matrix to get 100*378 question video
features = np.append(features, features_zeroes, axis = 0)
print (features.shape)
_input = features
#  _input = np.dstack((features, _input))
print (_input.shape)

for i in range(300, 304):
    for filename in os.listdir("./frames/frames_new/"+str(i)+"_P_new"):
        if filename.endswith(".csv") and filename != str(i)+"_P1.csv":
            print (filename)
            frame = pd.read_csv("./frames/frames_new/"+str(i)+"_P_new/"+filename)
            features = np.array(frame.iloc[:,3:-1].values, dtype = 'float')
            # assuming question file has 100 frames
            diff = 500 - features.shape[0]
    #             print (diff)
            features_zeroes = np.zeros((diff, features.shape[1]))
    #             print ("Feature matrix shape:")
    #             print (features.shape)
    #             print ("Feature zero matrix shape:")
    #             print (features_zeroes.shape)
            # adjust the feature matrix to get 100*378 question video
            features = np.append(features, features_zeroes, axis = 0)
            # print features
            _input = np.vstack((features, _input))

# change this
_input = _input.reshape((100,500,378))
print (_input.shape)
# for i in range(0, 10):
print (_input[5][0][0])
_input_features = torch.from_numpy(_input)
print (_input_features.shape)


# Reading test data
print ("Test data preprocessing....")
# create input numpy 2d array for tensor
_input_test = np.zeros([500,378], dtype='float')
# read file
frame = pd.read_csv("./frames/frames_new/test/300_P5.csv")
# get the features which start from 3rd column, skip the label column at end
features = np.array(frame.iloc[:,3:-1].values, dtype = 'float')
# assuming question file has 100 frames
diff = 500 - features.shape[0]
features_zeroes = np.zeros((diff, features.shape[1]))
print (features_zeroes.shape)
# adjust the feature matrix to get 100*378 question video
features = np.append(features, features_zeroes, axis = 0)
print (features.shape)
_input_test = features
#  _input = np.dstack((features, _input))
print (_input_test.shape)


for filename in os.listdir("./frames/frames_new/test"):
    if filename.endswith(".csv") and filename != "300_P5.csv":
        print (filename)
        frame = pd.read_csv("./frames/frames_new/test/"+filename)
        features = np.array(frame.iloc[:,3:-1].values, dtype = 'float')
#         # assuming question file has 100 frames
        diff = 500 - features.shape[0]
#             print (diff)
        features_zeroes = np.zeros((diff, features.shape[1]))
#             print ("Feature matrix shape:")
#             print (features.shape)
#             print ("Feature zero matrix shape:")
#             print (features_zeroes.shape)
        # adjust the feature matrix to get 100*378 question video
        features = np.append(features, features_zeroes, axis = 0)
        # print features
        _input_test = np.vstack((features, _input_test))

# change this
_input_test = _input_test.reshape((10,500,378))
print (_input_test.shape)
_input_test_features = torch.from_numpy(_input_test)
# print (_input_test_features.shape)


train_loader = torch.utils.data.DataLoader(dataset = _input_features, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = _input_features, batch_size = batch_size, shuffle = True)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i in enumerate(train_loader):
        print images, labels

