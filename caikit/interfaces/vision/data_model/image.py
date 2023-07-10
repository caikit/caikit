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

"""Data structures for representing images."""

# Third Party
from PIL import Image as PILImage
import numpy as np

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from .backends import ImagePilBackend
from .package import VISION_PACKAGE
from caikit.core import DataObjectBase, dataobject, error_handler

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(package=VISION_PACKAGE)
class Image(DataObjectBase):
    """Data model for an image object; this stores the image in the backend as a PIL image, with
    convenience views to interact with the Image as other formats as needed.
    """

    image_data: Annotated[bytes, FieldNumber(1)]

    def __init__(self, *args, **kwargs):
        # For now, we always delegate to the PIL image backend, because it's the only one.
        # In the future, we may have more views etc, but for now, images as PIL Images is
        # the hub format.
        #
        # NOTE: Currently, the args / kwargs here should match up with the proto class, because
        # .from_proto() requires it.
        self._backend = ImagePilBackend(*args, **kwargs)

    def as_numpy(self) -> np.ndarray:
        """Convert the Image data model to a Numpy Array; here, we delegate to the backend.
        Produced images are of type uint8.
        Note that the produced object will have shape (Rows, Cols, Channels).
        Returns:
            np.ndarray
                Numpy array(uint8) representation of this data model object.
        """
        self._check_initialization()
        return self._backend.as_numpy()

    def as_pil(self) -> PILImage.Image:
        """Convert the Image data model to a PIL image. Since we use PIL images as our hub format,
        this is simply returning a handle to the internally stored PIL image.
        Returns:
            PIL.Image
                PIL image representation of this data model object.
        """
        self._check_initialization()
        return self._backend.as_pil()

    def _check_initialization(self):
        """Ensure that we have a _backend; throw an attribute error if we don't. This will always
        be the case if we go through the initializer, but may occur if we call __new__() and use
        this class incorrectly.
        """
        if not isinstance(getattr(self, "_backend", None), ImagePilBackend):
            error(
                "<COR14411171E>",
                AttributeError(
                    "Unable to create view; missing PIL backend initialization"
                ),
            )

    ### Other attributes
    @property
    def rows(self) -> int:
        """Grab the number of rows in the image.
        Returns:
            int
                Number of rows in the underlying image.
        """
        return self.as_pil().height

    @property
    def columns(self) -> int:
        """Grab the number of columns in the image.
        Returns:
            int
                Number of columns in the underlying image.
        """
        return self.as_pil().width

    @property
    def channels(self) -> int:
        """Grab the number of channels in the image.
        Returns:
            int
                Number of channels in the underlying image.
        """
        return len(self.as_pil().getbands())
