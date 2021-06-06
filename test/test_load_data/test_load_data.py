import sys
sys.path.append('../../src/data')
import numpy as np
from load_data import load_pokemon_array, load_pokemon_image

# Unit Test for load_data functions
def test_output_numpy_array():
    """" Check output type """
    assert type(load_pokemon_image()) == np.ndarray
