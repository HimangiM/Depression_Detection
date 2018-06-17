import numpy as np
import torch
import os
#file import
import parameters
from feature_class import faceFeatures, concatFrames

sequence_length = parameters.sequence_length
feature_len = parameters.feature_len

print ("Test data preprocessing....")

csv_files_test = []
for filename in os.listdir("./frames"):
    if filename == "test": 
        for framefile in os.listdir("./frames/"+filename):
            file = filename + "/" + framefile
            csv_files_test.append(file)
            
# print (len(csv_files_test))
_input_ = np.zeros((sequence_length, feature_len), dtype="float32")
_label_ = np.ones((1), dtype="int32")
data_test = concatFrames(root_dir = "./frames/", csv_files = csv_files_test, _input = _input_, _label = _label_)
_input_test, _label_test = data_test._concat_()
_input_test = torch.Tensor(np.array(_input_test))
_label_test = torch.Tensor(np.array(_label_test))
_label_test = (_label_test.type(torch.LongTensor))

torch.save(_input_test, "input_test_300_normalized.pt")
torch.save(_label_test, "label_test_300_normalized.pt")

print (_input_test.shape)