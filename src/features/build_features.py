# image are weight: 256 and height: 256
import numpy as np
import tensorflow as tf

def normalize_images(array_img):
    """normalize and return an array of images"""

    return array_img/255.0


def gen_image_augmented(num_aug_images = 32,path = '../../data/interim/'):
    """generate augmented images and save them into path"""

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
        batch_size = num_aug_images,
    )

    pokemon_datagen.fit(array_img_pokemon)
    pokemon_datagen.flow(array_img_pokemon,save_to_dir = path))
