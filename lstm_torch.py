from skimage import io
import sys
import torch
import torch.nn as nn
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np

# class SimpleRNN(nn.Module):
#     def __init__(self, hidden_size):
#         self.hidden_size = hidden_size
#         self.inp = nn.Linear(1, hidden_size)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.LSTM = nn.LSTM(input_size, hidden_size, output_size, batch_first = True)
        self.fcc = nn.Linear(hidden_size, num_classes) 
    
    def forward(self, x):
        h0 = torch.zeros()

# create input numpy 2d array for tensor
_input = np.zeros([500,378], dtype='float32')
# read file
frame = pd.read_csv("./dataset/USC/bad_frames/frames/frames_with_label/"+str(i)+"_P_new/"+str(i)+"_P1.csv")
# get the features which start from 3rd column, skip the label column at end
features = np.array(frame.iloc[:,3:-1].values, dtype = 'float32')
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
    for filename in os.listdir("./dataset/USC/bad_frames/frames/frames_with_label/"+str(i)+"_P_new"):
        if filename.endswith(".csv") and filename != str(i)+"_P1.csv":
            print (filename)
            frame = pd.read_csv("./dataset/USC/bad_frames/frames/frames_with_label/"+str(i)+"_P_new/"+filename)
            features = np.array(frame.iloc[:,3:-1].values, dtype = 'float32')
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
            _input = np.dstack((features, _input))

            
print (_input.shape)
_input_features = torch.from_numpy(_input)
print (_input_features.shape)
