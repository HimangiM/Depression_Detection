import numpy as np
import torch
import os
from random import shuffle

#file import
import parameters 
from feature_class import faceFeatures, concatFrames

sequence_length = parameters.sequence_length 
feature_len = parameters.feature_len

print ("Training data preprocessing....")
# csv_files = ["300_P_new/300_P1.csv", "302_P_new/302_P2.csv","300_P_new/300_P2.csv"]
csv_files_train = []
for filename in os.listdir("./frames"):
    if filename != "test" and filename != "validation": 
        for framefile in os.listdir("./frames/"+filename):
            file = filename + "/" + framefile
            csv_files_train.append(file)
# print (csv_files_train)
shuffle(csv_files_train)
print (len(csv_files_train))

_input = np.zeros((sequence_length, feature_len), dtype="float32")
_label = np.ones((1), dtype="int32")

data = concatFrames(root_dir = "./frames/", csv_files = csv_files_train, _input = _input, _label = _label)
_input, _label = data._concat_()

_input_train = torch.Tensor(np.array(_input))
_label_train = torch.Tensor(np.array(_label))
_label_train = (_label_train.type(torch.LongTensor))


torch.save(_input_train, "input_train_300_normalized.pt")
torch.save(_label_train, "label_train_300_normalized.pt")
