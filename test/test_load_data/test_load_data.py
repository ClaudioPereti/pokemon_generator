import sys
sys.path.append('../../src/data')
import numpy as np
from load_data import load_pokemon_array, load_pokemon_image
import pytest

# Unit Test for load_data functions

@pytest.mark.parametrize('array_img_pokemon',[
    load_pokemon_array(),
    load_pokemon_image(),
]
)
# Check if output is an numpy array
def test_output_numpy_array(array_img_pokemon):
    """ Check output type """
    assert type(array_img_pokemon) == np.ndarray

@pytest.mark.parametrize('array_img_pokemon',[
    load_pokemon_array(),
    load_pokemon_image(),
]
)
# Check if the dimension output is compatible with an image dimension
def test_output_numpy_dimension_general(array_img_pokemon):
    """ Check output shape compatible with image (256,256,3) """
    assert np.shape(array_img_pokemon)[1:] == (256,256,3)


def test_aumented_data_present():
    """ Check if processed dataset contain augmented data """
    assert load_pokemon_array().shape[0] > load_pokemon_image().shape[0]
