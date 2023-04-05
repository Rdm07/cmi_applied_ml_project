import os, sys, time, random
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.optim as optim

from torchvision import transforms
from sklearn.metrics import accuracy_score, hamming_loss, roc_curve, auc, f1_score

# Local Imports
from utils import *
# from models.models import *

# parser = argparse.ArgumentParser(description='Playing Cards Classification Project')
# parser.add_argument('--lr', default=5e-6, type=float, help='Learning rate')
# parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
# parser.add_argument('--trainer', default='adam', type=str, help='Optimizer')
# parser.add_argument('--batch_size', default=256, type=int, help="Number of datapoints used in one batch")
# parser.add_argument('--num_workers', default=8, type=int, help='Number of CPU threads to be used')
# parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs in training')
# parser.add_argument('--check_after', default=1, type=int, help='Check the network after check_after epoch')
# parser.add_argument('--checkpoint', type=int, help='Folder to save model checkpoints', required=True)
# parser.add_argument('--data', type=str, help="Folder containing all subfolders of training/validation/testing data",
#                     required=True)
# parser.add_argument('--model', default='none', help="Pre-trained model name", required=False)
# args = parser.parse_args()

# Setting Random Seed
# rand_seed = 42
# if rand_seed is not None:
#     np.random.seed(rand_seed)
#     torch.manual_seed(rand_seed)
#     torch.cuda.manual_seed(rand_seed)

# Checking if cuda is available
use_gpu = torch.cuda.is_available()
print('Using GPU: ', use_gpu)

if use_gpu == True:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Setting frequency with which the training info will be printed
freq_print = 100

# Loading file paths and labels
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
print(data_path)
train_list, val_list, test_list = load_paths_from_csv(data_path)
labels_dict = create_label_dict(train_list)

# Getting input image size
img1 = Image.open(train_list[0][1]).convert('RGB')
input_size = img1.size

print(input_size)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(22),
        transforms.CenterCrop(350),
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),  # simple data augmentation
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),

    'val': transforms.Compose([
        transforms.CenterCrop(350),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}