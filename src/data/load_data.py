from PIL import Image
import numpy as np
import os


def load_pokemon_image(path = '../../data/raw/pokemon/pokemon/'):
    """load png images of pokemon and return an array (,256,256,3)"""

    list_img_pokemon = []
    for img in os.listdir(path):
        image = Image.open(path + img)
        image = image.convert('RGB')
        image = np.array(image)
        list_img_pokemon.append(image)


    array_img_pokemon = np.array(list_img_pokemon)
    return array_img_pokemon


def load_pokemon_array(path='../../data/processed/',name = 'all_image_pokemon-npy'):
    """load and return array containig 256x256x3 images"""
    
    return np.load(path+name)
