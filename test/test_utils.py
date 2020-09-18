import numpy as np
import pytest
import utils
import os


utils.fix_random_seeds()

@pytest.mark.parametrize("arg, expected", [
    [
        np.array([0.0, 0.25, 0.75]),
        np.array([0.22721977, 0.29175596, 0.48102426])
    ]
    
])
def test_softmax(arg, expected):
    result = utils.softmax(arg).round(8)
    expected = expected.round(8)
    assert np.array_equal(result, expected)
    

@pytest.mark.parametrize("arg, expected", [
    [-1, 0],
    [np.array([-1.0, 1.0]), np.array([0.0, 0.0])]
])
def test_d_tanh(arg, expected):
    assert np.array_equal(utils.d_tanh(arg), expected)
    
def test_randvec():
    x = utils.randvec(10)
    assert len(x) == 10
    
def test_randmatrix():
    X = utils.randmatrix(10, 20)
    assert X.shape == (10, 20)
    
def test_safe_macro_f1():
    y = [1, 1, 2, 2, 1]
    y_pred = [1, 2, 2, 1, 1]
    utils.safe_macro_f1(y, y_pred)
@pytest.mark.parametrize("arg, expected", [
    [
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.0, 0.0], [0.0, 0.0]])
    ]
])
def test_log_of_array_ignoring_zeros(arg, expected):
    result = utils.log_of_array_ignoring_zeros(arg)
    return np.array_equal(result, expected)


def test_glove2dict():
    src_filename = os.path.join("data", "glove.6B", "glove.6B.50d.txt")
    data = utils.glove2dict(src_filename)
    assert len(data) == 400000
