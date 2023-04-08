import os, sys, random
import torch, torchvision
import torch.nn as nn
import torchvision.models as models

from models import *

file_dir = os.path.dirname(__file__)

# File used to save your model to the saved_models folder

# Template to save a pytorch model
# torch.save(model_name, os.path.join(file_dir, 'saved_models/model1.pt'))

