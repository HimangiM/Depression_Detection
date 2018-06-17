import parameters
import torch
import numpy as np
import os
#file import
from feature_class import faceFeatures, concatFrames


sequence_length = parameters.sequence_length
feature_len = parameters.feature_len

print ("Validation data preprocessing....")

csv_files_validation = []
for filename in os.listdir("./frames"):
    if filename == "validation": 
        for framefile in os.listdir("./frames/"+filename):
            file = filename + "/" + framefile
            csv_files_validation.append(file)
            
# print (len(csv_files_validation))
_input_ = np.zeros((sequence_length, feature_len), dtype="float32")
_label_ = np.ones((1), dtype="int32")
data_validation = concatFrames(root_dir = "./frames/", csv_files = csv_files_validation, _input = _input_, _label = _label_)
_input_validation, _label_validation = data_validation._concat_()
_input_validation = torch.Tensor(np.array(_input_validation))
_label_validation = torch.Tensor(np.array(_label_validation))
_label_validation = (_label_validation.type(torch.LongTensor))

torch.save(_input_validation, "input_validation_300_normalized.pt")
torch.save(_label_validation, "label_validation_300_normalized.pt")

print (_input_validation.shape)