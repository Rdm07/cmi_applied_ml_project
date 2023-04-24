import torchvision, pickle
import torch.nn.functional as F

from torchvision import transforms

# Local Imports
from utils import *

os.chdir(os.path.dirname(__file__))

with open('../data/label_dict', 'rb') as handle:
    label_dict = pickle.load(handle)

# Data Transforms
data_transforms = {
	'val': transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
}

# Function to generate predictions for each model on the test dataset
def run_test_loader(model, test_loader, device):
	model.eval()
	pred_labels = []
	orig_labels = []

	ntotal = 0
	running_loss = 0.0
	with torch.no_grad():
		for ix, batch in enumerate(test_loader):
			inputs, targets = batch

			inputs = inputs.to(torch.float).to(device)
			targets = targets.to(torch.long).to(device)
			output = model(inputs)
			# For InceptionNet_v3
			if type(output) == torchvision.models.inception.InceptionOutputs:
				output,_ = output
			
			_, preds = torch.max(output.data, 1)

			targets = torch.squeeze(targets).data.cpu().tolist()
			orig_labels = orig_labels + targets

			preds = torch.squeeze(preds).data.cpu().tolist()
			pred_labels = pred_labels + preds
	
	orig_labels = np.array(orig_labels).reshape(-1,1)
	pred_labels = np.array(pred_labels).reshape(-1,1)

	return orig_labels, pred_labels

def classify_image(model, image_path, device, label_dict=label_dict):
	model.eval()

	img = Image.open(image_path).convert('RGB')
	png = img.resize((224,224))
	png = data_transforms['val'](png).unsqueeze(0)

	inputs = png.to(torch.float).to(device)
	output = model(inputs)

	# For InceptionNet_v3
	if type(output) == torchvision.models.inception.InceptionOutputs:
		output,_ = output
	
	pred_prob = F.softmax(output.data, 1)
	top_p, top_class = pred_prob.topk(1, dim = 1)

	pred_class = int(torch.squeeze(top_class).data.cpu())
	pred_prob = round(float(torch.squeeze(top_p).data.cpu()), 5)*100

	pred_class_lab = label_dict[pred_class]

	return pred_class_lab, pred_prob