import os, sys, random

from flask import Flask

os.chdir(os.path.dirname(__file__))

upload_folder = './upload_folder'
template_folder = './template'

app = Flask(__name__, template_folder= template_folder)
app.secret_key = "password"
app.config['UPLOAD_FOLDER'] = upload_folder