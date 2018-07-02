import torch
import numpy as np
import os
import cv2
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
import torch.utils.data as data_utils
from tqdm import tqdm_notebook
import pdb
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import operator

input_size = 4096
max_image_size = 71

direc = "/home/himangi/Bosch/CEK/CEK_64_64_augmented_class_duplicate"
l = []

def check_label(session, video):
	file = "/home/himangi/Bosch/CEK/label_path.csv"
	with open(file, "r") as infile:
		infile.readline()
		for line in infile.readlines():
			token = line.strip().split("\t")
			if token[0][-8:] == session + "/" + video or token[0][-9:] == session + "/" + video:
				return token[1]

	infile.close()
	return None


res = check_label("S010", "004A")
print (res)


class image_data(Dataset):
	def __init__(self, session, video):
		self.session = session
		self.video = video
	def __getitem__(self):
		if (check_label(self.session, self.video) != None):
			feature = np.empty((input_size,))
			first_image = 0
			c = 0
			for image in os.listdir(direc + "/" + self.session + "/" + self.video):
				img = Image.open(direc + "/" + self.session + "/" + self.video + "/" + image)
				img_ = np.array(img).flatten()
				img = [i/255 for i in img_]
#                 print (img.shape, feature.shape)
				feature = np.vstack((feature, img))
				if first_image == 0:
					first_image = 1
					feature = np.delete(feature, 0, 0)
				c = c+1
			label = check_label(self.session, self.video)
			video_len = c
			return (feature, int(label), video_len) 
		else:
			return None


data = image_data("S010", "004A").__getitem__()
print (data[0].shape, data[1], data[2])

first_vid = 0
_label = np.ones((1), dtype="int32")
for session in os.listdir(direc):
    if session != "validation" and session != "test":
        for video in os.listdir(direc + "/" + session):
            print (session, video)
            if first_vid == 0:
                data = image_data(session, video).__getitem__()
                print (data[0].shape)
                if data != None:
                    rows = max_image_size-data[0].shape[0]
                    img = np.pad(data[0], [(0,rows),(0,0)], 'constant', constant_values=(0))
                    feature = img
                    feature = feature.reshape(1, feature.shape[0], feature.shape[1])
                    label = data[1]
                    first_vid = 1
                    print (feature.shape)
            else:
                data = image_data(session, video).__getitem__()
                if data != None:
                    rows = max_image_size-data[0].shape[0]
                    img = np.pad(data[0], [(0,rows),(0,0)], 'constant', constant_values=(0))
                    img = img.reshape(1, img.shape[0], img.shape[1])
                    print (feature.shape, img.shape)
                    feature = np.vstack((feature,img))
                    label = np.vstack((label,data[1]))
                # stack the labels
feature = torch.Tensor(np.array(feature))
label = torch.Tensor(np.array(label))

print (feature.shape, label.shape)

torch.save(feature, "feature_cek_64_aug_class_dup_norm.pt")
torch.save(label, "label_cek_64_aug_class_dup_norm.pt")