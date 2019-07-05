# Cat vs Dog Classifier

The following repository contains a classifier that predicts whether given image contains a cat or a dog. 
The proposed solution uses convolutional neural network.


## Contents 
* `Cat_vs_Dog_classifier.ipynb` - Jupyter Notebook containing code with notes and conclusions regarding the whole process of creating the solution
* `animal_predictor.py` - simple command line program that allows to perform a prediction on a given image
* `model.h5` - saved model


### Prediction program

To perform a prediction `animal_predictor.py` requires path to the image that is denoted as `--image {path_to_an_image}`.<br>
Path to the model is specified by default (`model.h5`), to use another path please specify argument `--model {path_to_model}`.

Example usage: `python animal_predictor.py --image 'dataset/dog_image.jpg' --model 'model.h5'`
