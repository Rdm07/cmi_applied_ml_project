import os, sys, torch
import urllib.request

from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
from PIL import Image

# Local imports
from app import app

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from predict import classify_image

# Checking for GPU
use_gpu = torch.cuda.is_available()
print('Using GPU: ', use_gpu)

if use_gpu == True:
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

os.chdir(os.path.dirname(__file__))

# Importing Model
chkpt_path = "../models/model_checkpoint/model_efficient_net.pt_0.992_9.t7"
checkpoint = torch.load(chkpt_path, map_location=lambda storage, loc: storage)
model = checkpoint['model']
model.to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def submit_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            pred_class_lab, pred_prob = classify_image(model, file, device)
            # img = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            return render_template('result.html', pred_class=pred_class_lab, pred_prob=pred_prob)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)