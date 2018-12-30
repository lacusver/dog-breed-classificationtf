from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from glob import glob

#from san import *
from extract_bottleneck_features import *


from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.wsgi import WSGIServer


app = Flask(__name__)


MODEL_PATH = 'inceptionv3_model_21.hdf5'


model = load_model(MODEL_PATH)
dog_names = [item[21:-1] for item in sorted(glob('dataset/training_set/*/'))]
file=open('dataset/rus_name/dogbreeds.txt','r')
rus_dog_names=[item[0:-1] for item in file]
breeds=zip(dog_names,rus_dog_names)
breeds=dict(breeds)
model._make_predict_function()          
print('Model loaded. Start serving...')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    bottleneck_feature=extract_InceptionV3(x)
    preds = model.predict(bottleneck_feature)
    return preds

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

      
        preds = model_predict(file_path, model)
        breed=dog_names[np.argmax(preds)]
        rus_breed=breeds.get(str(breed))
        result = str(rus_breed)               
        return result
    return None


if __name__ == '__main__':
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
