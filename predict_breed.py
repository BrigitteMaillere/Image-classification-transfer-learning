# C:\Users\brigi\Documents\openclassrooms data scientist\parcours\6- Réalisez des indexations automatiques d’images\livrables>
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 



import numpy as np
import pickle

file_path = '/content/drive/My Drive/Colab Notebooks/notebooks/livrables/'
breeds = np.load(file_path + 'breeds.npy')
model = pickle.load(open(file_path + 're_traing_GVV16.sav', 'rb'))


import argparse
parser = argparse.ArgumentParser(description="tells a dog's breed from an image. Known breeds are " + str(list(breeds)))
parser.add_argument("image_path", help="path to the image to be classified. Example : predict_breed.py dog1.jpg ")
args = parser.parse_args()


from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input



def predict_breed(img) :	
    img = load_img(img, target_size=(224, 224))  # Charger l'image
    img = img_to_array(img)  # Convertir en tableau numpy
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))  # Créer la collection d'images (un seul échantillon)
    img = preprocess_input(img)  # Prétraiter l'image comme le veut VGG-16
    pred = model.predict(img)
    result = breeds[pred.argmax()]
    return result
	
print(" \n\n\n\n votre chien semble être de la race {}".format(predict_breed(file_path + args.image_path)))