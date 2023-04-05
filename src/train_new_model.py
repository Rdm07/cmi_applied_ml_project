import os, sys, time, random
import argparse

from torchvision import transforms
from sklearn.metrics import accuracy_score, hamming_loss, roc_curve, auc, f1_score
from utils import *

parser = argparse.ArgumentParser(description='Playing Cards Classification Project')
parser.add_argument('--lr', default=5e-6, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--trainer', default='adam', type=str, help='optimizer')
parser.add_argument('--batch_size', default=256, type=int, help="Number of datapoints used in one batch")
parser.add_argument('--num_workers', default=8, type=int, help='Number of CPU threads to be used')
parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs in training')
parser.add_argument('--check_after', default=1, type=int, help='check the network after check_after epoch')
parser.add_argument('--data', type=str, default='none', help="path to the folder containing all subfolders of training/testing data", required=False)
parser.add_argument('--data_list', type=str, default='none', help="text file containing the training/testing folder", required=False)
parser.add_argument('--model_folder', type=str, default='none', help="path to folder containing all pre-trained models", required=False)
parser.add_argument('--model_name', type=str, default='none', help="pre-trained model name", required=False)
args = parser.parse_args()