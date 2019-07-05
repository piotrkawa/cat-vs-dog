import pdb
import argparse
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import text


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', type=str, default='model.h5',
                        help='Path to the model')
    parser.add_argument('--image', dest='image', type=str,
                        help='Path to the image')
    args = parser.parse_args()
    return args.model, args.image


def get_image(image_path):
    pred_image = image.load_img(image_path, target_size=(64, 64))
    pred_image = image.img_to_array(pred_image)
    pred_image = np.expand_dims(pred_image, axis=0)
    return pred_image


def perform_prediction(model, image):
    prediction = model.predict(image)[0][0]
    return "Dog" if prediction == 1 else "Cat"


def show_result(image_path, prediction):
    image = mpimg.imread(image_path)
    plt.imshow(image)
    plt.title(prediction)
    plt.show()


def main():
    model_path, image_path = parse_arguments()
    model = load_model(model_path)
    image = get_image(image_path)
    prediction = perform_prediction(model, image)
    show_result(image_path, prediction)


if __name__ == '__main__':
    main()
