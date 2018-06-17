import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import pandas as pd
import parameters #file import


rec_dropout = parameters.rec_dropout
sequence_length = parameters.sequence_length
input_size = parameters.input_size
batch_size = parameters.batch_size
feature_len = parameters.feature_len

class faceFeatures(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.features_frame = pd.read_csv(root_dir + csv_file)
        self.transform = transform
        self.csv_file = csv_file

    def __len__(self):
        return len(self.features_frame)
 
    def __getitem__(self):
        features = np.zeros((sequence_length, feature_len), dtype="float32")
        label = np.ones((1), dtype="int32")
        
        all_features = self.features_frame.iloc[:, 3:-1].values
        diff = sequence_length - all_features.shape[0]
        
        if (diff < 0):
            rows = all_features.shape[0]
            row_idx = 0
            print (rows)
            while(row_idx + sequence_length <= rows):
                if (row_idx == 0):
                    features = all_features[row_idx:row_idx+sequence_length,:]
                    print ("First matrix shape:" + str(features.shape))
                    label = self.features_frame.iloc[1, -1]
                else:
                    features = np.vstack((features, all_features[row_idx:row_idx+sequence_length,:]))
                    label = np.vstack((label, self.features_frame.iloc[1, -1]))
                    print ("Subsequent matrix shape:" + str(features.shape))

                row_idx = row_idx + sequence_length
                
                
            new_diff = sequence_length - (all_features.shape[0] - row_idx)
            print ("Difference is: " + str(new_diff) + str(self.csv_file))
            features_zeroes = np.zeros((new_diff, all_features.shape[1]))
            second_features = np.append(all_features[row_idx:all_features.shape[0],:], features_zeroes, axis = 0)
            features = np.vstack((features, second_features))
            print ("Final matrix shape:" + str(features.shape))

            label = np.vstack((label, self.features_frame.iloc[1,-1]))

            features = features.reshape((-1, sequence_length, feature_len))


        else:
            features_zeroes1 = np.zeros((int(diff/2), all_features.shape[1]))
            features_zeroes2 = np.zeros((int(diff/2)+1, all_features.shape[1]))
            temp_features = np.append(features_zeroes1, all_features, axis = 0)
            if (diff % 2 == 0):
                temp_features = np.append(temp_features, features_zeroes1, axis = 0)
            else:
                temp_features = np.append(temp_features, features_zeroes2, axis = 0)
            features = temp_features
            print ("Single feature" + str(features.shape))
            features = features.reshape(-1, sequence_length, input_size)
#             print (self.features_frame.iloc[1, -1])
            label = np.array([self.features_frame.iloc[1,-1]])   
#         print (features, label)
        print ("Final shape:" + str(features.shape))
        return (features, label)


class concatFrames(Dataset):
    # Initialize the list of csv files
    def __init__(self, root_dir, _input, _label, csv_files = []):
        self.csv_files = csv_files
        self.root_dir = root_dir
        self._input = _input
        self._label = _label
        
    # Create tensor of frames
    def _concat_(self):
        data = faceFeatures(self.root_dir, self.csv_files[0])
        self._input, self._label = data.__getitem__()
        print (self._input.shape)
        # self._label = data.__getitem__()[1]
        
        for i in range(1, len(self.csv_files)):
            print (self.csv_files[i])
            data = faceFeatures(self.root_dir, self.csv_files[i])
            _feature_data, _label_data = data.__getitem__()
            for j in _feature_data:
                print (j.shape, self._input.shape)
                j = j.reshape(-1, sequence_length, input_size)
                self._input = self._input.reshape(-1, sequence_length, input_size)
                print (j.shape, self._input.shape)
                self._input = np.vstack((j, self._input))
            for j in _label_data:
                self._label = np.vstack((j, self._label))

        self._input = self._input.reshape((-1, sequence_length, feature_len))
        self._label = self._label.reshape((-1))
        print ("Concat :" + str(self._input.shape) + str(self._label.shape))
        return (self._input, self._label)

    # Get the tensor by index
    def __getitem__(self, idx):
        frame_name = self.csv_files[idx]
        frame_features = self._input[idx]
        frame_label = self._label[idx]
        return (frame_features, frame_label) 





