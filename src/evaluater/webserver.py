
import os
import glob
import json
import argparse
from utils.utils import calc_mean_score, save_json
from handlers.model_builder import Nima
from handlers.data_generator import TestDataGenerator
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

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
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return process(app.config['UPLOAD_FOLDER'], filename)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


def process(image_dir, image_file, img_format='jpg'):
    samples = [{'image_id': os.path.basename(image_file).split('.')[0]}]

    # initialize data generator
    data_generator = TestDataGenerator(samples, image_dir, 64, 10, nima_technical.preprocessing_function(), img_format=img_format)

    # get predictions
    technical_predictions = nima_technical.nima_model.predict(data_generator, verbose=1 if __debug__ else data_generator)
    aesthetic_predictions = nima_aesthetic.nima_model.predict(data_generator, verbose=1 if __debug__ else data_generator)

    # calc mean scores and add to samples
    for i, sample in enumerate(samples):
        sample['technical'] = calc_mean_score(technical_predictions[i])
        sample['aesthetic'] = calc_mean_score(aesthetic_predictions[i])

    return samples


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
