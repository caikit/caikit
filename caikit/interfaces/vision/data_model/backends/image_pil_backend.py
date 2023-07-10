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

"""Defines the backend for making Image data model objects work nicely with PIL
images. The ImagePilBackend outlines the manner in which we coerce our inputs to PIL
images, retrieve/cache data model attributes from the encapsulated PIL image, and so on.
"""

# Standard
from typing import Any, Union
import io
import os
import pathlib

# Third Party
from PIL import Image as PILImage
import numpy as np

# First Party
import alog

# Local
from caikit.core import error_handler
from caikit.core.data_model.base import DataBase
from caikit.core.data_model.data_backends import DataModelBackendBase

log = alog.use_channel("DATABACK")
error = error_handler.get(log)
PIL_SOURCE_TYPES = Union[PILImage.Image, pathlib.PosixPath, str, np.ndarray, bytes]


class ImagePilBackend(DataModelBackendBase):
    def __init__(self, image_data: PIL_SOURCE_TYPES):
        self._image_data = self.__class__.coerce_to_pil(image_data)
        # Unless the PIL image explicitly has information about its format, export
        # serialized image data as a PNG since it's not a lossy compression
        self._export_format = (
            self._image_data.format if self._image_data.format is not None else "PNG"
        )

    def get_attribute(self, data_model_class: DataBase, name: str) -> Any:
        # Convert the underlying PIL image to bytes
        if name == "image_data":
            # Save the whole contents of the image into a BytesIO object; this includes image fmt
            im_arr = io.BytesIO()
            self._image_data.save(im_arr, format=self._export_format)
            # Then return the whole value of the buffer as a bytes object
            return im_arr.getvalue()
        # Delegate to common parent logic
        return super().get_attribute(data_model_class, name)

    ##### Type coercion to PIL
    @classmethod
    def coerce_to_pil(
        cls,
        image_data: PIL_SOURCE_TYPES,
    ) -> PILImage:
        """Given an object representing an image in a wide variety of formats, force it into a
        PIL image representation.
        Supported formats:
            - [PIL.Image.Image] loaded PIL image; this is a no-op
            - [str] path to an image on disk to be loaded
            - [pathlib.PosixPath] path to an image on disk to be loaded
            - [np.ndarray] Numpy array of type uint8
            - [bytes] Image loaded into a bytes object

        Args:
            image_data:
                Raw data object to be coerced to a PIL image.
        Returns:
            PILImage.Image
                PIL image representation of the provided image data.
        """
        # If we're given a PIL image, then we have nothing to do!
        if isinstance(image_data, PILImage.Image):
            return image_data
        # Load a numpy array; [uint8]
        if isinstance(image_data, np.ndarray):
            return cls._coerce_from_numpy(image_data)
        # Load a path on disk (str or pathlib)
        if isinstance(
            image_data,
            (
                str,
                pathlib.PosixPath,
            ),
        ):
            return cls._coerce_from_path(image_data)
        # Load from a bytes object containing the whole image
        if isinstance(image_data, bytes):
            return cls._coerce_from_bytes(image_data)
        error(
            "<COR17413231E>",
            TypeError(
                f"Unsupported data type could not be coerced to PIL! {type(image_data)}"
            ),
        )

    @staticmethod
    def _coerce_from_numpy(image_data: np.ndarray) -> PILImage.Image:
        """Given image data as a Numpy array, load it as a PIL image.

        Args:
            image_data: np.ndarray
                Numpy array representing an image (uint8).
        Returns:
            PIL.Image.Image
                PIL image representation of the numpy array.
        """
        error.value_check(
            "<COR73134226E>",
            image_data.dtype == np.uint8,
            "Numpy array must be of type [np.uint8] to be loaded as a data model image",
        )
        return PILImage.fromarray(image_data)

    @classmethod
    def _coerce_from_path(cls, image_data: Union[str, pathlib.PosixPath]) -> PILImage:
        """Given a str, which we assume to be a path to an image on disk, or a Pathlib object,
        try to load it as a PIL image.

        Args:
            image_data: Union[pathlib.PosixPath, str]
                Path to be loaded.
        Returns:
            PIL.Image.Image
                PIL image representation of the image loaded from disk.
        """
        if isinstance(image_data, str) and not os.path.isfile(image_data):
            error(
                "<COR14433331E>",
                FileNotFoundError(f"Provided path [{image_data}] is not a file"),
            )
        elif isinstance(image_data, pathlib.PosixPath) and not image_data.is_file():
            error(
                "<COR14499831E>",
                FileNotFoundError(f"Provided pathlib [{image_data}] is not a file"),
            )
        return PILImage.open(image_data)

    @staticmethod
    def _coerce_from_bytes(image_data: bytes) -> PILImage:
        """Given bytes, which we assume to represent a full image, try to load it as a PIL image.
        Args:
            image_data: bytes
                binary data to be loaded as a PIL image.
        Returns:
            PIL.Image.Image
                PIL image representation of the bytes object.
        """
        return PILImage.open(io.BytesIO(image_data))

    ##### Backend views
    def as_numpy(self) -> np.ndarray:
        """Zero-copy method to produce the PIL image as a numpy array.
        Returns:
            np.ndarray
                Numpy array representation of the PIL image data.
        """
        return np.asarray(self._image_data)

    def as_pil(self) -> PILImage.Image:
        """Return a handle to our PIL image.
        Returns:
            PIL.Image.Image
                PIL image representation of this data model object.
        """
        return self._image_data
