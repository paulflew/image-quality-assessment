
import os
import glob
import json
import argparse
from utils.utils import calc_mean_score, save_json
from handlers.model_builder import Nima
from handlers.data_generator import TestDataGenerator


def image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]

    return img_dir, [{'image_id': img_id}]


def image_dir_to_json(img_dir, img_type='jpg'):
    img_paths = glob.glob(os.path.join(img_dir, '*.'+img_type))

    samples = []
    for img_path in img_paths:
        img_id = os.path.basename(img_path).split('.')[0]
        samples.append({'image_id': img_id})

    return samples


def predict(model, data_generator):
    return model.predict(data_generator, workers=8, use_multiprocessing=True, verbose=1)


def main(weights_dir, image_source, predictions_file, img_format='jpg'):
    # load samples
    if os.path.isfile(image_source):
        image_dir, samples = image_file_to_json(image_source)
    else:
        image_dir = image_source
        samples = image_dir_to_json(image_dir, img_type='jpg')

    # build models and load weights
    nima_technical = Nima('MobileNet', weights=None)
    nima_technical.build()
    nima_technical.nima_model.load_weights(os.path.join(weights_dir, 'technical.hdf5'))
    nima_aesthetic = Nima('MobileNet', weights=None)
    nima_aesthetic.build()
    nima_aesthetic.nima_model.load_weights(os.path.join(weights_dir, 'aesthetic.hdf5'))

    # initialize data generator
    data_generator = TestDataGenerator(samples, image_dir, 64, 10, nima_technical.preprocessing_function(),
                                       img_format=img_format)

    # get predictions
    technical_predictions = predict(nima_technical.nima_model, data_generator)
    aesthetic_predictions = predict(nima_aesthetic.nima_model, data_generator)

    # calc mean scores and add to samples
    for i, sample in enumerate(samples):
        sample['technical'] = calc_mean_score(technical_predictions[i])
        sample['aesthetic'] = calc_mean_score(aesthetic_predictions[i])

    print(json.dumps(samples, indent=2))

    if predictions_file is not None:
        save_json(samples, predictions_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights-dir', help='path of weights files', required=True)
    parser.add_argument('-is', '--image-source', help='image directory or file', required=True)
    parser.add_argument('-pf', '--predictions-file', help='file with predictions', required=False, default=None)

    args = parser.parse_args()

    main(**args.__dict__)
