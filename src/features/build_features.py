# image are weight: 256 and height: 256
import sys
sys.path.append('../../config')
sys.path.append('../data')
from config import path_augmented_images
import numpy as np
import tensorflow as tf
from config import path_raw_images
from PIL import Image
from load_data import load_pokemon_image

def normalize_images(array_img):
    """normalize and return an array of images"""

    return array_img/255.0


def gen_image_augmented(batch_size = 1,path=None,save_path=None):
    """generate augmented images and save them into path"""

    pokemon_datagen = tf.keras.preprocessing.image.ImageDataGenerator(

        rescale = 1./255,
        rotation_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2,
    )

    return pokemon_datagen.flow_from_directory('../../data/raw/pokemon/',target_size = (256,256),batch_size =batch_size,class_mode=None,shuffle=False,save_to_dir = path)


# poke = gen_image_augmented()
#
# for i in range(2):
#     print(poke.next())
