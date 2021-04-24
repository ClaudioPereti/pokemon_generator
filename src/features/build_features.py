# image are weight: 256 and height: 256
import PIL
from PIL import Image
image = Image.open("/home/claudio/machine_learning_project/pokemon_generator/data/raw/pokemon/pokemon/1.png")

from PIL import Image
import numpy as np
import os
os.listdir(path)
path = '../../data/raw/pokemon/pokemon/'


list_img_pokemon = []
for img in os.listdir(path):
    image = Image.open(path + img)
    image = image.convert('RGB')
    image = np.array(image)/255.0
    list_img_pokemon.append(image)

# len(list_img_pokemon)
array_img_pokemon = np.array(list_img_pokemon)

import tensorflow as tf
pokemon_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2,
)

pokemon_datagen.fit(array_img_pokemon)
for i in range(5):
    next(pokemon_datagen.flow(array_img_pokemon,save_to_dir = '../../data/processed/'))


path_augmented = '../../data/interim/'
len(os.listdir(path_augmented))
Image.open(path_augmented + os.listdir(path_augmented)[4])

list_img_augmented_pokemon = []
for img in os.listdir(path_augmented):
    image = Image.open(path_augmented + img)
    image = image.convert('RGB')
    image = np.array(image)/255.0
    list_img_augmented_pokemon.append(image)

# len(list_img_pokemon)
array_img_augmented_pokemon = np.array(list_img_augmented_pokemon)


array_all_img_pokemon = np.vstack([array_img_pokemon,array_img_augmented_pokemon])

np.random.shuffle(array_all_img_pokemon)

array_all_img_pokemon.shape
