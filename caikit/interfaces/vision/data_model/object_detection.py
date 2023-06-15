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
"""Data structures for text object detection in images."""

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from ....core import DataObjectBase, dataobject
from ...common.data_model import ProducerId
from .package import VISION_PACKAGE

log = alog.use_channel("DATAM")

@dataobject(package=VISION_PACKAGE)
class BoundingBox(DataObjectBase):
    xmin: Annotated[int, FieldNumber(1)]
    xmax: Annotated[int, FieldNumber(2)]
    ymin: Annotated[int, FieldNumber(3)]
    ymax: Annotated[int, FieldNumber(4)]

@dataobject(package=VISION_PACKAGE)
class DetectedObject(DataObjectBase):
    score: Annotated[str, FieldNumber(1)]
    box: Annotated[BoundingBox, FieldNumber(2)]
    score: Annotated[float, FieldNumber(3)]

@dataobject(package=VISION_PACKAGE)
class ObjectDetectionResult(DataObjectBase):
    detected_objects: Annotated[list[BoundingBox], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]
