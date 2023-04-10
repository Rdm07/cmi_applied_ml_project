import os, sys, time, random
import argparse, copy

import numpy as np
import torch.optim as optim
import torch.nn as nn

from torchvision import transforms
from sklearn.metrics import accuracy_score, hamming_loss, roc_auc_score, f1_score
from time import strftime

# Local Imports
from utils import *

parser = argparse.ArgumentParser(description='Playing Cards Classification Project')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
parser.add_argument('--trainer', default='sgd', type=str, help='Optimizer')
parser.add_argument('--batch_size', default=256, type=int, help="Number of datapoints used in one batch")
parser.add_argument('--num_workers', default=8, type=int, help='Number of CPU threads to be used')
parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs in training')
parser.add_argument('--check_after', default=1, type=int, help='Run validation on network after check_after epoch')
parser.add_argument('--checkpoint', type=str, help='Folder to save model checkpoints')
parser.add_argument('--data_folder', type=str, help="Folder containing all subfolders of training/validation/testing data")
parser.add_argument('--model_folder', type=str, help="Folder containing saved model")
parser.add_argument('--model_name', type=str, help="Saved model name")
parser.add_argument('--last_layer_nodes', type=int, help='Number of Classes (Nodes in last layer)')
parser.add_argument('--mean', nargs="+", type=float, default=None, help="Mean of channels from pre-trained model")
parser.add_argument('--std', nargs="+", type=float, default=None, help="Std Dev of channels from pre-trained model")
args = parser.parse_args()

print('Batch Size: {}'.format(args.batch_size))

# Setting Random Seed
rand_seed = 42
if rand_seed is not None:
	np.random.seed(rand_seed)
	torch.manual_seed(rand_seed)
	torch.cuda.manual_seed(rand_seed)

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
data_path = args.data_folder
train_list, val_list, test_list = load_paths_from_csv(data_path)
labels_dict = create_label_dict(train_list)

# Getting input image size
img1 = Image.open(train_list[0][1]).convert('RGB')
input_size = img1.size

# Calculating Mean, Std Dev for training images
mean, std = args.mean, args.std

if mean is None or std is None:
	train_set = dataloader(train_list, transform=transforms.ToTensor())
	mean, std = get_mean_and_std(train_set)

# Data Transforms
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

# Random Shuffling lists
random.shuffle(train_list)
random.shuffle(val_list)
random.shuffle(test_list)

# Creating train and val datasets and DataLoader objects
train_set = dataloader(train_list, transform = data_transforms['train'])
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)

val_set = dataloader(val_list, transform = data_transforms['val'])
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)

# Function to run evaluation for a model on a given dataloader object
def run_validation(model, criterion, val_loader):
	model.eval()
	pred_labels = []
	orig_labels = []

	ntotal = 0
	running_loss = 0.0
	with torch.no_grad():
		for ix, batch in enumerate(val_loader):
			inputs, targets = batch

			inputs = inputs.to(torch.float).to(device)
			targets = targets.to(torch.long).to(device)
			output = model(inputs)
			if type(output) == tuple:
				output,_ = output
			
			_, preds = torch.max(output.data, 1)

			loss = criterion(output, targets)
			ntotal += output.size(0)
			running_loss += loss.item() * output.size(0)

			targets = torch.squeeze(targets).data.cpu().tolist()
			orig_labels = orig_labels + targets

			preds = torch.squeeze(preds).data.cpu().tolist()
			pred_labels = pred_labels + preds
	
	orig_labels = np.array(orig_labels).reshape(-1,1)
	pred_labels = np.array(pred_labels).reshape(-1,1)

	val_loss = running_loss/ntotal
	val_ham = (1 - hamming_loss(orig_labels, pred_labels))
	val_acc = accuracy_score(orig_labels, pred_labels)
	val_f1 = f1_score(orig_labels, pred_labels, average='macro')

	return val_loss, val_ham, val_acc, val_f1, orig_labels, pred_labels

# Loading the model to be trained
model_folder = args.model_folder
model_path = os.path.join(model_folder, args.model_name)

# Function to train a given model on a given train and val dataloader object
def run_training(model, criterion, num_epochs, trainer = args.trainer, train_loader = train_loader, val_loader = val_loader): 
	best_f1 = 0
	best_epoch = 0
	train_val_loss_epoch = []
	start_training = time.time()

	for epoch in range(num_epochs):

		### TRAINING ###
		start = time.time()

		if epoch < num_epochs//6: lr = args.lr
		elif epoch < num_epochs//3: lr = args.lr/2
		elif epoch < num_epochs//2: lr = args.lr/4
		elif epoch < 3*(num_epochs//4): lr = args.lr/10
		else: lr = args.lr/20

		if epoch > 2:
			for param in model.parameters():
				param.requires_grad = True


		if trainer == 'adam':
			optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
		elif trainer == 'sgd':
			optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=args.weight_decay)
		else:
			raise Exception('Select optimiser (trainer) as adam or sgd')
		
		print('Epoch {}/{}'.format(epoch + 1, num_epochs))
		print('lr: {:.6f}'.format(lr))
		print('-' * 50)

		model.train()
		ntotal = 0
		running_loss = 0.0
		running_corrects = 0
		for ix, batch in enumerate(train_loader):
			inputs, targets = batch

			inputs = inputs.to(torch.float).to(device)
			targets = targets.to(torch.long).to(device)

			optimizer.zero_grad()
			output = model(inputs)
			if type(output) == tuple:
				output,_ = output
			
			_, preds = torch.max(output.data, 1)
			loss = criterion(output, targets)
			loss.backward()
			optimizer.step()

			ntotal += output.size(0)
			running_loss += loss.item() * output.size(0)
			running_corrects += torch.sum(preds == targets.data)

			if (ix + 1) % freq_print == 0:
				print('| Epoch:[{}][{}/{}]\tTrain_Loss: {:.4f}\tAccuracy: {:.4f}\tTime: {:.2f} mins'.format(
					epoch + 1, ix + 1, len(train_loader.dataset)//args.batch_size, running_loss/ntotal, 
					running_corrects.item()/ntotal, (time.time() - start)/60.0))

			sys.stdout.flush()

		### VALIDATION ###
		if (epoch + 1) % args.check_after == 0:
			model.eval()
			start = time.time()
			val_loss, val_ham, val_acc, val_f1, orig_labels, pred_labels = run_validation(model, criterion, val_loader = val_loader)
			train_val_loss_epoch.append([running_loss/ntotal, val_loss])
			print("Epoch: {}/{}\tVal_Loss: {:.4f}\tAccuracy: {:.4f}\tF1-score: {:.4f}\t{:.3f}mins".format(
					(epoch + 1), num_epochs, val_loss, val_acc, val_f1, (time.time() - start)/60.0))
			start = time.time()

			# Save model if auc best auc score is acheived (yet)
			if val_f1 > best_f1:
				print('Saving model')
				best_f1 = val_f1
				best_epoch = epoch
				best_model = copy.deepcopy(model)
				state = {
					'model': best_model,
					'auc': best_f1,
					'args': args,
					'lr': lr,
					'saved_epoch': epoch,
				}

				save_point = os.path.join(args.checkpoint)
				if not os.path.isdir(os.path.join(save_point)):
					os.mkdir(save_point)

				saved_model_name = args.model_name
				torch.save(state, os.path.join(save_point, saved_model_name + '_' + str(round(best_f1,3)) + '_' + str(epoch) + '.t7'))
				print('=======================================================================')

	time_elapsed = time.time() - start_training
	print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
	print('Best validation F1 score: {:4f} at epoch: {}'.format(best_f1, best_epoch))

	plot_tra_val_loss(train_val_loss_epoch)

def main():
	sys.setrecursionlimit(10000)

	print("Loading model at  %s..." % model_path)
	model = torch.load(model_path)

	for param in model.parameters():
		param.requires_grad = False

	num_in = model.fc.in_features
	model.fc = nn.Linear(num_in, 53)

	model.eval()
	model.to(device)
	model = unparallelize_model(model)
	model = parallelize_model(model, device)
	print(model)
	
	##################
	print('Start training ... ')
	criterion = nn.CrossEntropyLoss().to(device)
	run_training(model, criterion, num_epochs=args.num_epochs, trainer = args.trainer, train_loader=train_loader, val_loader=val_loader)

if __name__ == "__main__":
	main()