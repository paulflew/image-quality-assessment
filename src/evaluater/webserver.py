
import os
import argparse
from utils.utils import calc_mean_score
from handlers.model_builder import Nima
from flask import Flask, flash, request, redirect
import numpy as np
from PIL import Image

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# build models and load weights
WEIGHTS_DIR = '/tmp'
nima_technical = Nima('MobileNet', weights=None)
nima_aesthetic = Nima('MobileNet', weights=None)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            image = Image.open(file)
            if image.size != (224, 224):
                image = image.resize((224, 224))
            return process(image)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


def process(image):
    # prepare image
    X = np.array([np.asarray(image)])
    X = nima_technical.preprocessing_function()(X)

    # get predictions
    technical_predictions = nima_technical.nima_model(X)
    aesthetic_predictions = nima_aesthetic.nima_model(X)

    # calc mean scores and return
    return {'technical': calc_mean_score(technical_predictions), 'aesthetic': calc_mean_score(aesthetic_predictions)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights-dir', help='path of weights files', required=False, default=WEIGHTS_DIR)
    args = parser.parse_args()

    nima_technical.build()
    nima_technical.nima_model.load_weights(os.path.join(args.weights_dir, 'technical.hdf5'))
    nima_aesthetic.build()
    nima_aesthetic.nima_model.load_weights(os.path.join(args.weights_dir, 'aesthetic.hdf5'))

    if __debug__:
        app.debug = True
        app.run()
    else:
        from waitress import serve
        serve(app, host="0.0.0.0", port=8080)
