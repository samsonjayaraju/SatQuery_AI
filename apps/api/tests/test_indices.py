import numpy as np

from app.remote_sensing.indices import ndbi, ndvi, ndwi


def test_normalized_difference_indices_are_bounded():
    high = np.array([[0.8, 0.4]], dtype=np.float32)
    low = np.array([[0.2, 0.4]], dtype=np.float32)
    for result in (ndvi(high, low), ndwi(high, low), ndbi(high, low)):
        assert np.all(result <= 1)
        assert np.all(result >= -1)
        assert result[0, 0] > 0
        assert result[0, 1] == 0
