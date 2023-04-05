import os, sys, random
import csv
import torch
import numpy as np
import PIL.Image as Image

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
	def __init__(self, imgs, transform=None):
		self.imgs = imgs
		self.transform = transform
	
	def __getitem__(self, index):
		img = self.imgs[index]
		lab = np.array([int(int(img[1]) > 0)])[0]
		png = Image.open(img[0]).convert('RGB') # ori: RGB, do not convert to numpy, keep it as PIL image to apply transform

		if self.transform:
			png = self.transform(png)

		return png, lab

	def __len__(self):
		return len(self.imgs)

