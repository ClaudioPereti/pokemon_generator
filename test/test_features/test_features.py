import sys
sys.path.append('../../src/features')
import numpy as np
from build_features import normalize_images
import pytest


def test_output_type_normalize_images():
    """ Check output type """
    assert type(normalize_images(np.array([1]))) == np.ndarray

@pytest.mark.parametrize('array_numbers,expected_result',[
    (np.array([1,1,1]),np.array([1/255.0,1/255.0,1/255.0])),
    (np.array([10,10]),np.array([10/255.0,10/255.0])),
    (np.array([255,255,255,255]),np.array([1,1,1,1])),
]
)
def test_normalization_normalize_images(array_numbers,expected_result):
    """ Check normalization value """
    assert (normalize_images(array_numbers) == expected_result).all()

def test_sign_normalize_images():
    """ Check if normalization doesn't change sign """
    assert normalize_images(np.array([1])) > 0
    assert normalize_images(np.array([-1])) < 0
    assert normalize_images(np.array([0])) == 0
