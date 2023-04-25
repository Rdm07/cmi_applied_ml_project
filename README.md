# Applied ML Project: Playing Cards Classification (Jan-April 2023)

This repo is for training and testing Deep Learning models for classifying playing card images

# Dependencies
All the dependancies can be installed using pip install -r requirements.txt from the root folder

# Running Codes Instructions
- Bash Files are in folder scripts, including training and testing
- Change the training parameters in scripts/run_training_new.sh

# Training (From Scratch)
- For an original model architecture, specify the class in models/models.py file
- Create a model instance of the required architecture (original or PyTorch templates) in save_model.py and save as .pt file in the models/saved_models folder (a code template for saving the file is included in the python script)
- Changing the model name in scripts/run_training_new.sh, run the script to train the model
- Find the complete terminal output during training in the models/training_log folder

# Training (Transfer Learning)
- For a pre-trained model, save an instance of the model in saved_models
- Changing the model name in scripts/run_training_pretrained.sh (or .bat), run the script to train the model
- Find the complete terminal output during the training in the models/training_log folder

# MLFlow Experiment
- Run run_test_mlflow.sh to run prediction on the test dataset for a specific model checkpoint and log the metrics in mlflow

# Flask App
- Run run_flask.sh to run the flask on 127.0.0.1:5000 
- Upload any card image and click submit
- The app will return the predicted class and its confidence for the same