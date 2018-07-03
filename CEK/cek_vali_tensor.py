
# coding: utf-8

# In[91]:


import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image


# In[92]:


input_size = 4096
direc = "/home/himangi/Bosch/CEK/Emotion/validation"
directory = "/home/himangi/Bosch/CEK/CEK_64_64_augmented_class_duplicate/validation"


# In[93]:


def check_label(session, video):
    if os.path.exists(direc + "/" + session + "/" + video):
        for file in os.listdir(direc + "/" + session + "/" + video):
            f = open(direc + "/" + session + "/" + video + "/" + file, "r")
            label = int(float(f.read().strip()))
            # print (label)
            return label
    
    return None


# In[94]:


class image_data(Dataset):
    def __init__(self, session, video):
        self.session = session
        self.video = video
    def __getitem__(self):
        if (check_label(self.session, self.video) != None):
            feature = np.empty((input_size,))
            first_image = 0
            c = 0
            for image in os.listdir(directory + "/" + self.session + "/" + self.video):
                img = Image.open(directory + "/" + self.session + "/" + self.video + "/" + image)
                img = np.array(img)                
                img_ = np.array(img).flatten()
                img = [i for i in img_]
                feature = np.vstack((feature, img))
                if first_image == 0:
                    first_image = 1
                    feature = np.delete(feature, 0, 0)
                c = c+1
#             for txtfile in os.listdir(direc + "/" + self.session + "/" + self.video):
#                 f = open(directory + "/" + self.session + "/" + self.video + "/" + txtfile)
#                 label = ((f.read().strip()))
            label = check_label(self.session, self.video)
            video_len = c
            return (feature, int(label), video_len) 
        else:
            return None


# In[103]:


first_vid = 0
_label = np.ones((1), dtype="int32")

for file in os.listdir(directory):
    for video in os.listdir(directory + "/" + file):
        data = image_data(file, video).__getitem__()
        if (data == None):
            print ("None")
            print (file, video)
        else:
            if first_vid == 0:
                data = image_data(file, video).__getitem__()
                feature = data[0]
                _label = data[1]
                feature = np.array(feature)
                _label = np.array(_label)
                feature = feature.reshape(1, feature.shape[0], feature.shape[1])
                first_vid = 1
                print (feature.shape, _label.shape)
            else:
                data = image_data(file, video).__getitem__()
                print ("Here", data[0].shape)
                f = data[0]
                f = np.array(f)
                f = f.reshape(1, f.shape[0], f.shape[1])
                feature  = np.vstack((feature, f))
                label = data[1]
                _label = np.vstack((_label, label))

feature = torch.Tensor(feature)
_label = torch.Tensor(_label)            
print (feature.shape, _label.shape)
torch.save(feature, "feature_vali_cek_64_aug_class_dup_norm.pt")
torch.save(_label, "label_vali_cek_64_aug_class_dup_norm.pt")