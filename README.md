# Applied ML Project: Playing Cards Classification (Jan-April 2023)

This repo is for training and testing Deep Learning models for classifying playing card images

# Dependencies
All the dependancies can be installed using pip install -r requirements.txt from the root folder

# Running Codes Instructions
- Bash or Batch Files are in folder scripts, including training and testing
- Change the training parameters in scripts/run_training.sh or .bat file

# Training (From Scratch)
- For an original model architecture, specify the class in models/models.py file
- Create a model instance of the required architecture (original or PyTorch templates) in save_model.py and save as .pt file in the models/saved_models folder (a code template for saving the file is included in the python script)
- Changing the model name in scripts/run_training.sh (or .bat), run the script to train the model
- Find the complete terminal output during training in the training_log folder

# Training (Transfer Learning)
- train_transfer.py yet to be written

# Testing
- test.py yet to be written