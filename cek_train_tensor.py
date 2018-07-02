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


sequence_length = 71
input_size = 4096
hidden_size = 256
num_layers = 2
num_classes = 8
batch_size = 5
learning_rate = 0.01
rec_dropout = 0.05

direc = "/home/himangi/Bosch/CEK/CEK_64_64_augmented_class_duplicate"
label_direc = "/home/himangi/Bosch/CEK/Emotion"
l = []
max_image_size = 71

d = dict()

class image_data(Dataset):
	def __init__(self, session, video):
		self.session = session
		self.video = video
	def __getitem__(self):
		if (os.path.exists(label_direc + "/" + self.session + "/" + self.video)) and (os.listdir(label_direc + "/" + self.session + "/" + self.video)):
			feature = np.empty((input_size,))
			first_image = 0
			c = 0
			for image in os.listdir(direc + "/" + self.session + "/" + self.video):
				
				img = Image.open(direc + "/" + self.session + "/" + self.video + "/" + image)
				img = np.array(img).flatten()
#                 print (img.shape, feature.shape)
				feature = np.vstack((feature, img))
				if first_image == 0:
					first_image = 1
					feature = np.delete(feature, 0, 0)
				c = c+1

			for file in os.listdir(label_direc + "/" + self.session + "/" + self.video):
				f = open(label_direc + "/" + self.session + "/" + self.video + "/" + file, "r")
				label = np.array(int(float(f.read().strip())))

			video_len = c
			return (feature, label, video_len) 
		else:
			return None


for file in os.listdir(direc):
	if file != "test" and file != "validation":
		for folder in os.listdir(direc + "/" + file):
			c = 0
			# print (file, folder)

			for image in os.listdir(direc + "/" + file + "/" + folder):
				# print (image)
				c = c+1
			d[str(file + "/" + folder)] = c


# print d


d_files = sorted(d.items(), key=operator.itemgetter(1), reverse = True)
print d_files

name_first = d_files[0][0]
print str(name_first)

for image in os.listdir(direc + "/" + str(name_first)):
	token = str(name_first).strip().split("/")
	# print token[0], token[1]
	data = image_data(token[0], token[1]).__getitem__()
	feature = data[0]
	feature = feature.reshape(1, feature.shape[0], feature.shape[1])
	label = data[1]
	video_len = data[2]

print (name_first, feature, label, video_len)

# print (feature.shape, label.shape)

for i in range(1, len(d_files)):
	token = str(d_files[i][0]).strip().split("/")
	data = image_data(token[0], token[1]).__getitem__()
	if (data!=None):
		rows = max_image_size-data[0].shape[0]

		img = np.pad(data[0], [(0,rows),(0,0)], 'constant', constant_values=(0))
		img = img.reshape(1, img.shape[0], img.shape[1])
		# print (feature.shape, img.shape, label.shape)
		feature = np.vstack((feature,img))
		label = np.vstack((label,data[1]))
		video_len = np.vstack((video_len, data[2]))
		print (d_files[i][0], token[0], token[1], d_files[i][1], rows)


print (feature.shape, label.shape, video_len.shape)

'''
first_vid = 0
count = 0
_label = np.ones((1), dtype="int32")
for session in os.listdir(direc):
	if session != "validation" and session != "test":
		for video in os.listdir(direc + "/" + session):
			print (session, video)
			if first_vid == 0:
				data = image_data(session, video).__getitem__()
				if data != None:
					rows = max_image_size-data[0].shape[0]
					img = np.pad(data[0], [(0,rows),(0,0)], 'constant', constant_values=(0))
					feature = img
					feature = feature.reshape(1, feature.shape[0], feature.shape[1])
					label = data[1]
					first_vid = 1
					count = 1
#                     print (label.shape)
			else:
				if count < 30:
					data = image_data(session, video).__getitem__()
					if data != None:
						rows = max_image_size-data[0].shape[0]
						img = np.pad(data[0], [(0,rows),(0,0)], 'constant', constant_values=(0))
						img = img.reshape(1, img.shape[0], img.shape[1])
						print (feature.shape, img.shape, label.shape)
						feature = np.vstack((feature,img))
						label = np.vstack((label,data[1]))
						count = count + 1

#                 # stack the labels
feature = torch.Tensor(np.array(feature))
label = torch.Tensor(np.array(label))


print (feature.shape, label.shape)


'''
torch.save(feature, "feature_cek_64_aug_class.pt")
torch.save(label, "label_cek_64_aug_class.pt")
torch.save(video_len, "video_len_cek_64_aug_class.pt")
