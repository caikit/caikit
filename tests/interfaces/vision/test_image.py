# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for image data model and its backend/views."""
# Standard
from pathlib import Path
import io
import os

# Third Party
from PIL import Image as PILImage
import numpy as np
import pytest

# Local
from caikit.interfaces import vision as v

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures")

HAPPY_IMAGE_PATH = os.path.join(FIXTURES_DIR, "data", "sample_image.png")
SAD_IMAGE_PATH = "does/not/exist"

with open(HAPPY_IMAGE_PATH, "rb") as raw_img:
    RAW_IMG_BYTES = raw_img.read()
    PIL_IMG_DATA = PILImage.open(io.BytesIO(RAW_IMG_BYTES))
NP_UINT8_DATA = np.asarray(PIL_IMG_DATA)

IMAGE_DATA_KEY = "image_data"
VALID_INITIALIZATIONS = [
    # A path to an image
    {IMAGE_DATA_KEY: HAPPY_IMAGE_PATH},
    # Pathlib object to an image
    {IMAGE_DATA_KEY: Path(HAPPY_IMAGE_PATH)},
    # A numpy array of type np.uint8 [0-255]
    {IMAGE_DATA_KEY: NP_UINT8_DATA},
    # A PIL image
    {IMAGE_DATA_KEY: PIL_IMG_DATA},
    # Raw bytes representing a compressed image
    {IMAGE_DATA_KEY: RAW_IMG_BYTES},
]

INVALID_INITIALIZATIONS = [
    # Numpy img cast to float32
    {IMAGE_DATA_KEY: NP_UINT8_DATA.astype(np.float32)},
]

### Initialization tests
@pytest.mark.parametrize("happy_kwargs", VALID_INITIALIZATIONS)
def test_valid_image_initializations_with_kwargs(happy_kwargs):
    """Test that we can build an Caikit Image data model object directly [kwargs]."""
    im = v.data_model.Image(**happy_kwargs)
    assert isinstance(im, v.data_model.Image)
    # Ensure that no matter what, we always have the same data
    numpy_im = im.as_numpy()
    assert isinstance(numpy_im, np.ndarray)
    assert np.allclose(numpy_im, NP_UINT8_DATA)


@pytest.mark.parametrize("happy_kwargs", VALID_INITIALIZATIONS)
def test_valid_image_initializations_with_positional_args(happy_kwargs):
    """Test that we can build an Caikit Image data model object directly [positional args]."""
    im = v.data_model.Image(happy_kwargs[IMAGE_DATA_KEY])
    assert isinstance(im, v.data_model.Image)
    # Ensure that no matter what, we always have the same data
    numpy_im = im.as_numpy()
    assert isinstance(numpy_im, np.ndarray)
    assert np.allclose(numpy_im, NP_UINT8_DATA)


@pytest.mark.parametrize("sad_kwargs", INVALID_INITIALIZATIONS)
def test_invalid_initializations_with_kwargs(sad_kwargs):
    """Test that we get a ValueError if we try to init with unsupported types [kwargs]."""
    with pytest.raises(ValueError):
        v.data_model.Image(**sad_kwargs)


@pytest.mark.parametrize("sad_kwargs", INVALID_INITIALIZATIONS)
def test_invalid_initializations_with_positional_args(sad_kwargs):
    """Test that we get a ValueError if we try to init with unsupported types [positional args]."""
    with pytest.raises(ValueError):
        v.data_model.Image(sad_kwargs[IMAGE_DATA_KEY])


def test_bad_path_initialization():
    """Test that we get a FileNotFoundError if we try to load a bad path."""
    with pytest.raises(FileNotFoundError):
        v.data_model.Image(image_data=SAD_IMAGE_PATH)


def test_bad_pathlib_initialization():
    """Test that we get a FileNotFoundError if we try to load a bad pathlib object."""
    with pytest.raises(FileNotFoundError):
        v.data_model.Image(image_data=Path(SAD_IMAGE_PATH))


def test_bad_init_type():
    """Test that we produce a TypeError if we pass a bad type to dm initializer."""

    class SadType:
        pass

    with pytest.raises(TypeError):
        v.data_model.Image(SadType())


### View tests
def test_as_pil():
    """Test that we can grab a view of our DM object as a PIL image."""
    pimg = v.data_model.Image(PIL_IMG_DATA).as_pil()
    # Since we make the dm.Image using PIL as our hub format, the view references the same object
    assert pimg is PIL_IMG_DATA


def test_as_numpy():
    """Test that we can grab a view of our DM object as a Numpy array."""
    dimg = v.data_model.Image(PIL_IMG_DATA)
    expected_shape = (dimg.rows, dimg.columns, dimg.channels)
    npimg = v.data_model.Image(PIL_IMG_DATA).as_numpy()
    assert npimg.shape == expected_shape
    assert npimg.dtype == np.uint8


### Backend validation tests
def test_no_backend_property_access_fails():
    """Test that we throw good errors if we skip _backend setting."""
    sad_img = v.data_model.Image.__new__(v.data_model.Image)
    # Any property relying on a view should raise
    for prop_name in ["rows", "columns", "channels"]:
        with pytest.raises(AttributeError):
            getattr(sad_img, prop_name)
    # So should actually invoking a view method
    for view_name in ["as_numpy", "as_pil"]:
        with pytest.raises(AttributeError):
            getattr(sad_img, view_name)()


def test_from_sad_backend():
    """Test that if garbage is used for our backend, views fail."""
    sad_img = v.data_model.Image.from_backend(100)
    with pytest.raises(AttributeError):
        sad_img.as_numpy()


def test_default_format_pil():
    """Test that we can serialize a PIL image created in-memory with a lossless compression."""
    default_format = "PNG"
    img = PILImage.new("RGB", (1000, 600), color=(255, 255, 255))
    # Image should have no format information
    assert img.format is None
    # We should be able to serialize the image without any issues; we test .proto() since it's used
    # in the underlying conversion to alternate serialized formats, e.g., .json() also.
    dm_image = v.data_model.Image(img)
    proto_img = dm_image.to_proto()
    assert dm_image._backend._export_format == default_format
    # Then, if we reconstruct from the serialized proto, it should be
    # reloaded as a PNG because it was serialized as a PNG.
    assert (
        v.data_model.Image.from_proto(proto_img)._backend._export_format
        == default_format
    )


### Tests for other properties
def test_dm_props():
    """Test that our row, col, and channel properties return the correct values."""
    im = v.data_model.Image(image_data=PIL_IMG_DATA)
    assert im.rows == 100
    assert im.columns == 250
    assert im.channels == 3
