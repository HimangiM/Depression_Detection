import numpy as np
import torch
import torch.utils.data as data_utils
from tqdm import tqdm_notebook
import parameters
import random
from collections import namedtuple
from lstm_class import RNN
import torch.nn as nn
import matplotlib.pyplot as plt


input_size = parameters.input_size
batch_size = parameters.batch_size
hidden_size = parameters.hidden_size
num_epochs = parameters.num_epochs
num_layers = parameters.num_layers
num_classes = parameters.num_classes
rec_dropout = parameters.rec_dropout
learning_rate = parameters.learning_rate
sequence_length = parameters.sequence_length


_input_train = torch.load("input_train_300_normalized.pt")
_label_train = torch.load("label_train_300_normalized.pt")
_input_validation = torch.load("input_validation_300_normalized.pt")
_label_validation = torch.load("label_validation_300_normalized.pt")


_input_train = np.array(_input_train)
_label_train = np.array(_label_train)


discard_size = _input_train.shape[0] % batch_size

discard_idx = []
for i in range(0, discard_size):
    discard_idx.append(random.randint(0, _input_train.shape[0]))
    
discard_idx = sorted(discard_idx)
discard_idx = list(reversed(discard_idx))

for i in (discard_idx):
    _input_train = np.delete(_input_train, i, 0)
    _label_train = np.delete(_label_train, i, 0)
   

_input_train = torch.Tensor(_input_train)
_label_train = torch.Tensor(_label_train)
_label_train = (_label_train.type(torch.LongTensor))

train = data_utils.TensorDataset(_input_train, _label_train)
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)
validation = data_utils.TensorDataset(_input_validation, _label_validation)
validation_loader = data_utils.DataLoader(validation, shuffle=True)
total_step = len(train_loader)

loss = 0

model = RNN(input_size, hidden_size, num_layers, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

batchTuple = namedtuple("batchTuple", "feature label batch_size")
for epoch in (range(num_epochs)):
    n_correct, n_total = 0, 0
    train_loss = []
    valid_loss = []
    train_acc_list = []
    valid_acc_list = []
    # i is the counter, ith batch, j is the value of batch
    # Training
    for i,(feature, label) in enumerate(train_loader):
        feature = feature.reshape(-1, sequence_length, input_size)
#         print (feature.shape)

        batch = batchTuple(feature = feature, label = label, batch_size = batch_size)
        
        # Forward pass
        outputs = model(feature)
#         print (outputs.shape)
#         print (label.shape)

        label = label.reshape(batch_size)

        # Calculate train accuracy
        _, predicted_t = torch.max(outputs.data, 1)
        n_correct += (torch.max(outputs, 1)[1].view(label.size()) == label).sum().item()
        n_total = n_total + label.size(0)
        train_acc = n_correct/n_total
        train_acc_list.append(train_acc)
        
        # Calculate loss
        loss = criterion(outputs, label)
        train_loss.append(loss.item())
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
     # Validation
    with torch.no_grad():
        correct = 0
        total = 0

        for j, (images, labels) in enumerate(validation_loader):

            images = images.reshape(-1, sequence_length, input_size)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct += (torch.max(outputs, 1)[1].view(labels.size()) == label).sum().item()
            d_loss = criterion(outputs, labels)
            valid_loss.append(d_loss)
                        
        valid_acc = correct/total
        valid_acc_list.append(valid_acc)
        
    print ("Epoch", "Training accuracy, Training loss, Validation loss, Validation Accuracy")  
    print (epoch, sum(train_acc_list)/len(train_acc_list), sum(train_loss)/len(train_loss), sum(valid_loss)/len(valid_loss), sum(valid_acc_list)/len(valid_acc_list))  
    

#     print ("Epoch time:")
#     print (time.time() - epoch_start)
#     epoch_start = time.time()

            
details = "hidden_size:" + str(hidden_size) + ",learning_rate:" + str(learning_rate) + ",dropout:" + str(rec_dropout)
print (details)

plt.figure() 
plt.plot(train_loss)
plt.title("Training loss " + str(details))
    
plt.figure()
plt.plot(train_acc_list)
plt.title("Training accuracy " + str(details))            

plt.figure()
plt.plot(valid_loss)
plt.title("Validation loss " + str(details))

plt.figure()
plt.plot(valid_acc_list)
plt.title("Validation accuracy " + str(details))

