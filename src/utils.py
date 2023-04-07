import os, sys, random
import csv
import torch
import numpy as np
import PIL.Image as Image
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

def load_paths_from_csv(data_folder_path: str) -> tuple[list, list, list]:
	"""
	A Function to load the labels.csv file and get 3 lists of 
	[label_id, file_path, label_name, card_type, dataset_type]
	for training, validation and testing datasets
	"""
	file = open(os.path.join(data_folder_path, 'labels.csv'))
	full_list = list(csv.reader(file, delimiter=','))
	file.close()

	del full_list[0]

	for item in full_list:
		item[1] = os.path.join(data_folder_path, item[1])

	list_train = [x for x in full_list if x[-1] == 'train']
	list_val = [x for x in full_list if x[-1] == 'valid']
	list_test = [x for x in full_list if x[-1] == 'test']

	return list_train, list_val, list_test

def create_label_dict(list1: list) -> dict:
	"""
	Using a list of labels and filepaths, create a label dict 
	associating label_id with label_name
	"""
	dict1 = {}
	for item in list1:
		if item[0] in dict1:
			continue
		else:
			dict1[item[0]] = item[2]

	return dict1

def modify_label_dict(list1: list, dict1: dict) -> dict:
	"""
	Using a different list, check if all classes have been covered in the label dict
	"""
	for item in list1:
		if item[0] in dict1:
			continue
		else:
			dict1[item[0]] = item[2]

	return dict1

class dataloader(Dataset):
	"""
	Dataset class to read img file from filepath 
	and return Pillow Image object with its label_id
	"""
	def __init__(self, list1, transform=None):
		self.list1 = list1
		self.transform = transform
	
	def __getitem__(self, index):
		list2 = self.list1[index]
		lab = list2[0]
		png = Image.open(list2[1]).convert('RGB') # ori: RGB, do not convert to numpy, keep it as PIL image to apply transform

		if self.transform:
			png = self.transform(png)

		return png, lab

	def __len__(self):
		return len(self.list1)

def get_mean_and_std(data_list: list) -> tuple[float, float]:
    '''
	Compute the mean and std value of dataset.
	'''
    dataset = dataloader(data_list)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def parallelize_model(model, device):
	if torch.cuda.is_available():
		model = model.to(device)
		model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
		cudnn.benchmark = True
		return model

def unparallelize_model(model):
	try:
		while 1:
			# to avoid nested dataparallel problem
			model = model.module
	except AttributeError:
		pass
	return model

def plot_tra_val_loss(tra_val_loss):
	epochs = list(range(1,len(tra_val_loss)+1))
	train_loss = [x[0] for x in tra_val_loss]
	val_loss = [x[1] for x in tra_val_loss]
	plt.plot(x=epochs, y=train_loss, label='Train Loss')
	plt.plot(x=epochs, y=val_loss, label='Validation Loss')
	plt.title('Training and Validation Loss per epoch')
	plt.ylabel('Loss')
	plt.xlabel('Epochs')
	plt.legend()
	plt.show()

