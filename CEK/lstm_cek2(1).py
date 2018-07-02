
# coding: utf-8

# In[49]:


import torch
import numpy as np
import os
import cv2
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.functional as F


# In[50]:


sequence_length = 71
input_size = 4096
hidden_size = 256
num_layers = 2
num_classes = 8
batch_size = 50
learning_rate = 0.0001
rec_dropout = 0.05


# In[51]:


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc1 = nn.Linear(hidden_size, hidden_size/2)
        # self.fc2 = nn.Linear(hidden_size/2, num_classes)
	# Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Initialize hidden and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
	# out has the output for the lstm, _ is the hidden state
        out, _ = self.lstm(x, (h0, c0))  
        # out_ = F.relu(self.fc1(out[:,-1,:]))
        # out = self.fc2(out_)
	# Apply fully connected to the last sequence of out
        out = self.fc(out[:,-1,:])
        return out


# In[52]:

# Initialize RNN
model = RNN(input_size, hidden_size, num_layers, num_classes)
# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


# In[53]:

# Load the tensors
_input_train = torch.load("feature_cek_64_aug_class_dup_norm.pt")
_label_train = torch.load("label_cek_64_aug_class_dup_norm.pt")
# _input_train = torch.load("feature_cek_64_dup_flip_norm_concat.pt")
# _label_train = torch.load("label_cek_64_dup_flip_norm_concat.pt")


# In[54]:


print (_input_train.shape, _label_train.shape, type(_label_train), type(_input_train))
# Convert the loaded files into tensor and reshape them (not needed, but to ensure the correct shape)
_input_train = torch.Tensor(_input_train)
_label_train = torch.Tensor(_label_train)
_label_train = _label_train.view(_label_train.shape[0])
# LSTM needs the label to be in Long Type, throws error that expected long tensor, but received float tensor
_label_train = _label_train.type(torch.LongTensor)


# In[55]:

# Combine the input matrix and label into a dataset train
train = data_utils.TensorDataset(_input_train, _label_train)
# print (train[0])
# print (train[10])
# Load the dataset using loaded into batches
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True) 
# print (train_loader[0][0])


# In[ ]:


# batchTuple = namedtuple("batchTuple", "feature label batch_size")
model.train()
for t in (range(250)):
    n_correct, n_total = 0, 0
    train_loss = []
    valid_loss = []
    train_acc_list = []
    valid_acc_list = []
    # i is the counter, ith batch, j is the value of batch
    # Training

    for i, (feature, label) in enumerate(train_loader):
       	# outputs has the cell state of LSTM
        outputs = model(feature)
        print (t, outputs.data)
	
	# From the output probabilites, select the max probability	
        _,predicted_t = torch.max(outputs.data,1)
        
        print (t, label, predicted_t)
        
	# count the number of correct predictions
        n_correct = n_correct + (torch.max(outputs, 1)[1].view(label.size()) == label).sum().item()
	
	# Total the number of inputs given
        n_total = n_total + label.size(0)
        n_correct = float(n_correct)
	# Get the accuracy
	train_acc = n_correct/n_total
	# Append to train_acc
        train_acc_list.append(train_acc)
        print (n_correct, n_total, float(n_correct/n_total))
        
        # Calculate loss
        loss = criterion(outputs, label)
        train_loss.append(loss.item())
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
           
    print ("Training accuracy, Training loss")  
    print (t, sum(train_acc_list)/len(train_acc_list), sum(train_loss)/len(train_loss))  
   
'''
    # Validation
     with torch.no_grad():
         correct = 0
         total = 0

         for j, (images, labels) in enumerate(validation_loader):

             images = images.view(-1, sequence_length, input_size)
             labels = labels
             # outputs is the probabiltites, predicted is the final class
             outputs = model(images)
             _, predicted = torch.max(outputs.data, 1)
# #             print ("Validation")
# #             print (predicted, labels)
             total = total + labels.size(0)
             correct += (torch.max(outputs, 1)[1].view(labels.size()) == labels).sum().item()
             d_loss = criterion(outputs, labels)
             valid_loss.append(d_loss)
             valid_acc = correct/total
             valid_acc_list.append(valid_acc)
        
'''
