import torch
import numpy as np
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader

N, D_in, H, D_out = 50, 37800, 64, 27
batch_size = N

_input_train = torch.load("input_train_100_27.pt")
_label_train = torch.load("label_train_100_27.pt")

_input_train = np.array(_input_train)
_label_train = np.array(_label_train)

_input = np.array(_input_train[0].flatten())
_label = np.zeros(D_out)
_label[_label_train[0]] = 1

for i in range(1, len(_input_train)):
	a = np.array(_input_train[i].flatten())
	_input = np.vstack((_input, a))
	d = np.zeros(D_out)
	d[_label_train[i]] = 1
	_label = np.vstack((_label, d))

print (_input.shape, _label.shape)

x = torch.Tensor(_input)
y = torch.Tensor(_label)

train = data_utils.TensorDataset(x, y)
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

model = torch.nn.Sequential(torch.nn.Linear(D_in, H), torch.nn.ReLU(), torch.nn.Linear(H, D_out))

loss_fn = torch.nn.MSELoss(size_average = False)

learning_rate = 1e-4

for t in range(100):
	for i,(feature, label) in enumerate(train_loader):
		y_pred = model(feature)
		loss = loss_fn(y_pred, label)

		print (t, loss.item())

		model.zero_grad()

		loss.backward()

		with torch.no_grad():
			for param in model.parameters():
				param = param - learning_rate * param.grad