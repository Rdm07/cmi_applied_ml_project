import os, random, warnings
import mlflow, argparse

from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, matthews_corrcoef
from mlflow.tracking import MlflowClient
from torchvision import transforms

# Local Imports
from utils import *
from predict import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Playing Cards Classification Project')
parser.add_argument('--batch_size', default=256, type=int, help="Number of datapoints used in one batch")
parser.add_argument('--num_workers', default=8, type=int, help='Number of CPU threads to be used')
parser.add_argument('--data_folder', type=str, help="Folder containing all subfolders of training/validation/testing data")
parser.add_argument('--checkpoint_folder', type=str, help="Folder containing trained model")
parser.add_argument('--log_folder', type=str, help="Folder for log")
parser.add_argument('--model_name', type=str, help="Saved model name")
parser.add_argument('--model_name_pretty', type=str, help="Model name for MLFlow")
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

# Loading data
data_path = args.data_folder
train_list, val_list, test_list = load_paths_from_csv(data_path)

# Calculating Mena, Std Dev for training images
mean, std = args.mean, args.std

if mean is None or std is None:
	train_set = dataloader(train_list, transform=transforms.ToTensor())
	mean, std = get_mean_and_std(train_set)

# Data Transforms
data_transforms = {
	'train': transforms.Compose([
		transforms.RandomRotation(22),
		transforms.RandomHorizontalFlip(),  # simple data augmentation
		transforms.RandomVerticalFlip(),
		transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)]),

	'val': transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
	]),
}

random.shuffle(test_list)

# Creating test dataset and dataLoader objects
test_set = dataloader(test_list, transform = data_transforms['val'])
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)

# Loading the model to be trained
model_folder = args.checkpoint_folder
model_path = os.path.join(model_folder, args.model_name)

# Function to evaluate models
def eval_model(test_labels, pred_labels):
	# Computing Accuracy
	acc_sc = accuracy_score(test_labels, pred_labels)

	# Computing multi-class f1_score 
	f1_sc = f1_score(test_labels, pred_labels, average='macro')
	
	# Computing Cohen-Kappa Score
	cp_sc = cohen_kappa_score(test_labels, pred_labels)

	# Computing Mathew's Correlation co-efficient
	mcr_sc = matthews_corrcoef(test_labels, pred_labels)

	return (["Accuracy", "F1-Score", "Cohen-Kappa Score", "Matthews-Correlation-Coefficient"], [acc_sc, f1_sc, cp_sc, mcr_sc])

# Function to Log Metrics
def storing_metric(md_name, test_labels, pred_labels):

	client = MlflowClient()
	#log into MLflow

	#Set storage directory
	mlflow.set_tracking_uri(os.path.join(args.log_folder, 'MLFlow_Logs/mlruns'))

	#set experiment
	mlflow.set_experiment('Classification of Playing Cards using pre-trained CNN models')
	
	with mlflow.start_run() as run: 

		#Log parameters
		mlflow.log_param("Model",md_name)
		#Running the model
		metric_label, model_metrics = eval_model(test_labels, pred_labels)
		
		#Logging metrics
		for i in range(len(model_metrics)):
			mlflow.log_metric(metric_label[i], model_metrics[i])
		
def main():
	sys.setrecursionlimit(10000)

	print("Loading model checkpoint at  %s..." % model_path)
	checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
	model = checkpoint['model']

	model.eval()
	model.to(device)
	model = unparallelize_model(model)
	model = parallelize_model(model, device)
	
	##################
	print('Start predicting ... ')
	orig_labels, pred_labels = run_test(model=model, test_loader=test_loader, device=device)

	storing_metric(args.model_name_pretty, orig_labels, pred_labels)

if __name__ == "__main__":
	main()