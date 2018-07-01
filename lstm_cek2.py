
# coding: utf-8

# In[39]:


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



# In[40]:


sequence_length = 71
input_size = 4096
hidden_size = 256
num_layers = 2
num_classes = 8
batch_size = 50
learning_rate = 0.001
rec_dropout = 0.05


# In[41]:

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc1 = nn.Linear(hidden_size, hidden_size/2)
        # self.fc2 = nn.Linear(hidden_size/2, num_classes)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        print (x.shape)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Decode the hidden state of the last time step
        print ("here")
#         print (out[:,:,-1].shape)
        # out_ = F.relu(self.fc1(out[:,-1,:]))
        # out = self.fc2(out_)
        out = self.fc(out[:,-1,:])
        return out


# In[44]:


model = RNN(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


# In[45]:


_input_train = torch.load("feature_cek_64_aug_class_dup_norm.pt")
_label_train = torch.load("label_cek_64_aug_class_dup_norm.pt")

print (_input_train.shape, _label_train.shape, type(_label_train), type(_input_train))
_input_train = torch.Tensor(_input_train)
_label_train = torch.Tensor(_label_train)
_label_train = _label_train.view(_label_train.shape[0])
_label_train = _label_train.type(torch.LongTensor)


# In[46]:


train = data_utils.TensorDataset(_input_train, _label_train)
# print (train[0])
# print (train[10])
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=False) 
# print (train_loader[0][0])


# In[47]:


# batchTuple = namedtuple("batchTuple", "feature label batch_size")
model.train()
for t in (range(20)):
    n_correct, n_total = 0, 0
    train_loss = []
    valid_loss = []
    train_acc_list = []
    valid_acc_list = []
    # i is the counter, ith batch, j is the value of batch
    # Training
    c = 0
    for i, (feature, label) in enumerate(train_loader):
        # n_correct, n_total = 0, 0
#         optimizer.zero_grad()
        
#         feature = feature.view(-1, sequence_length, input_size)
        print (t, feature[0][0], label[0])
#         label = label.view(batch_size)
        outputs = model(feature)
        print (t, outputs.data)
        _,predicted_t = torch.max(outputs.data,1)
        
        print (t, label, predicted_t)
        
        # check the below line
        n_correct = n_correct + (torch.max(outputs, 1)[1].view(label.size()) == label).sum().item()

        n_total = n_total + label.size(0)
        n_correct = float(n_correct)
        train_acc = n_correct/n_total
        train_acc_list.append(train_acc)
        print n_correct, n_total, float(n_correct/n_total)
        
        # Calculate loss
        loss = criterion(outputs, label)
        train_loss.append(loss.item())
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        c = c+1
        
#      # Validation
#     with torch.no_grad():
#         correct = 0
#         total = 0

#         for j, (images, labels) in enumerate(validation_loader):

#             images = images.view(-1, sequence_length, input_size)
#             labels = labels
#             # outputs is the probabiltites, predicted is the final class
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
# #             print ("Validation")
# #             print (predicted, labels)
#             total = total + labels.size(0)
#             correct += (torch.max(outputs, 1)[1].view(labels.size()) == labels).sum().item()
#             d_loss = criterion(outputs, labels)
#             valid_loss.append(d_loss)
#             valid_acc = correct/total
#             valid_acc_list.append(valid_acc)
        
    print ("Training accuracy, Training loss")  
    print (t, sum(train_acc_list)/len(train_acc_list), sum(train_loss)/len(train_loss))  
    

#     print ("Epoch time:")
#     print (time.time() - epoch_start)
#     epoch_start = time.time()

# print ("Mean Training Accuracy, Mean Validation Accuracy")
# print (sum(train_acc_list)/len(train_acc_list), sum(valid_acc_list)/len(valid_acc_list))
# details = "hidden_size:" + str(hidden_size) + ",learning_rate:" + str(learning_rate) + ",dropout:" + str(rec_dropout)
# print (details)

plt.figure() 
plt.plot(train_loss)
plt.title("Training loss " + str(details))
    
plt.figure()
plt.plot(train_acc_list)
plt.title("Training accuracy " + str(details))            

# plt.figure()
# plt.plot(valid_loss)
# plt.title("Validation loss " + str(details))

# plt.figure()
# plt.plot(valid_acc_list)
# plt.title("Validation accuracy " + str(details))

# plt.figure()
# plt.plot(train_acc_list, label = 'training')
# plt.plot(valid_acc_list, label = 'validation')
# plt.title("Training and Validation accuracy" + str(details))


