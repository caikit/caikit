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
"""Data structures for text classification.
"""

# Standard
from enum import Enum
from typing import List, Union
import collections.abc

# Third Party
import numpy as np

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from ....core.data_model import DataObjectBase, dataobject
from ....core.toolkit.errors import DataValidationError, error_handler
from ...common.data_model import ProducerId

log = alog.use_channel("DATAM")


@dataobject(package="caikit_data_model.nlp")
class ModelType(Enum):
    """
    A single hierarchical NLC prediction.
    """

    MULTI_TARGET = 0
    SINGLE_TARGET = 1


error = error_handler.get(log)


@dataobject(package="caikit_data_model.nlp")
class ClassInfo(DataObjectBase):
    class_name: Annotated[str, FieldNumber(1)]
    confidence: Annotated[float, FieldNumber(2)]

    """A single classification prediction."""

    def __init__(self, class_name, confidence):
        """Construct a new classification prediction.

        Args:
            class_name:  str
                The labels in our classification hierarchy.
            confidence:  int | float | np.number
                Confidence-like score in [0, 1] for this class.
        """
        error.type_check("<NLP09900624E>", str, class_name=class_name)
        error.type_check("<NLP03131624E>", np.number, int, float, confidence=confidence)
        error.value_check(
            "<NLP27713002E>",
            0.0 <= confidence <= 1.0,
            "`confidence` of `{}` not between 0 and 1",
            confidence,
        )

        super().__init__()
        self.class_name = class_name
        # Always convert confidence into a float, since this is what the protobuf spec expects.
        self.confidence = float(confidence)

    def __lt__(self, other):
        """Compare classifications by score."""
        return self.confidence < other.confidence


@dataobject(package="caikit_data_model.nlp")
class ClassificationPrediction(DataObjectBase):
    """A Classification prediction generated from a document and consisting multiple classes."""

    classes: Annotated[List[ClassInfo], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]

    """The result of a classification prediction."""

    def __init__(self, classes, producer_id=None):
        """Construct a new classification prediction.

        Args:
            classes:  list(ClassInfo)
                The classification (predictions) associated with this
                classification prediction.
            producer_id:  ProducerId or None
                The block that produced this classification prediction.
        """
        error.type_check_all("<NLP43800581E>", ClassInfo, classes=classes)
        error.type_check(
            "<NLP50868756E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        super().__init__()
        self.classes = sorted(classes, reverse=True)
        self.producer_id = producer_id


@dataobject(package="caikit_data_model.nlp")
class ClassificationTrainRecord(DataObjectBase):
    """A Classification Training record consisting of a single train instance."""

    text: Annotated[str, FieldNumber(1)]
    labels: Annotated[List[str], FieldNumber(2)]

    """A Train record for Classification."""

    def __init__(self, text, labels):
        """Construct a new Classification Train record.

        Args:
            text: string
                The raw text associated with this train record.
            labels: list(string)
                The class labels associated with this train record.
        """
        error.type_check("<NLP43811580E>", str, text=text)
        error.type_check_all("<NLP42812582E>", str, labels=labels)

        super().__init__()
        self.text = text
        self.labels = labels

    @classmethod
    def from_data_obj(
        cls,
        data_obj: Union[dict, list, tuple, "ClassificationTrainRecord"],
        data_item_number: int = None,
    ):
        """Constructs a ClassificationTrainRecord instance from a train instance

        Args:
            data_obj: dict, list, tuple or instance of ClassificationTrainRecord
                A single train instance representation
            data_item_number: int or None
                An index for the train instance. Can be used when converting objects of a stream

        Returns:
            ClassificationTrainRecord: Instance of ClassificationTrainRecord
        """
        if isinstance(data_obj, ClassificationTrainRecord):
            return data_obj

        if isinstance(data_obj, dict):
            # Check if keys do not exist
            missing_keys = set(cls.fields) - set(data_obj.keys())
            if len(missing_keys) > 0:
                message = "Data item is missing required key(s): {} ".format(
                    missing_keys
                )
                raise DataValidationError(message, item_number=data_item_number)
            try:
                return cls.from_json(data_obj)
            except (ValueError, TypeError) as e:
                raise DataValidationError(e, item_number=data_item_number)

        elif isinstance(data_obj, collections.abc.Sequence):
            if len(data_obj) < 2:
                message = "Expected data item {} to have a mimimum of 2 elements but contained {} elements".format(
                    data_item_number, len(data_obj)
                )
                raise DataValidationError(message, item_number=data_item_number)
            text = data_obj[0]
            if isinstance(data_obj[1], collections.abc.Sequence) and not isinstance(
                data_obj[1], str
            ):
                labels = data_obj[1]
            else:
                # Iterable can be a tuple or list of string labels
                labels = list(data_obj[1:])
            try:
                return cls(text=text, labels=labels)
            except (ValueError, TypeError) as e:
                raise DataValidationError(e, item_number=data_item_number)
