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
"""Data structures for classification representations"""

# Standard
from typing import List, Optional

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from ....core import DataObjectBase, dataobject
from .package import NLP_PACKAGE
from .text_generation import TokenStreamDetails

log = alog.use_channel("DATAM")


@dataobject(package=NLP_PACKAGE)
class ClassificationTrainRecord(DataObjectBase):
    text: Annotated[str, FieldNumber(1)]
    labels: Annotated[List[str], FieldNumber(2)]


@dataobject(package=NLP_PACKAGE)
class Classification(DataObjectBase):
    label: Annotated[str, FieldNumber(1)]
    score: Annotated[float, FieldNumber(2)]


@dataobject(package=NLP_PACKAGE)
class ClassificationResult(DataObjectBase):
    results: Annotated[List[Classification], FieldNumber(1)]


# NOTE: Annotated[This is meant to align with the HuggingFace token classification task:
# https://huggingface.co/docs/transformers/tasks/token_classification#inference
# The field `word` does not necessarily correspond to a single "word",
# and `entity` may not always be applicable beyond "entity" in the NER
# (named entity recognition) sense
@dataobject(package=NLP_PACKAGE)
class TokenClassification(DataObjectBase):
    start: Annotated[int, FieldNumber(1)]
    end: Annotated[int, FieldNumber(2)]
    word: Annotated[str, FieldNumber(3)]  # could be thought of as text
    entity: Annotated[str, FieldNumber(4)]  # could be thought of as label
    entity_group: Annotated[
        str, FieldNumber(5)
    ]  # could be thought of as aggregate label, if applicable
    score: Annotated[float, FieldNumber(6)]
    token_count: Annotated[Optional[int], FieldNumber(7)]


@dataobject(package=NLP_PACKAGE)
class TokenClassificationResult(DataObjectBase):
    results: Annotated[List[TokenClassification], FieldNumber(1)]


# Streaming result that indicates up to where in stream is processed
@dataobject(package=NLP_PACKAGE)
class StreamingTokenClassificationResult(TokenClassificationResult):
    # Result index up to which text is processed
    processed_index: Annotated[int, FieldNumber(2)]


@dataobject(package=NLP_PACKAGE)
class ClassifiedGeneratedTextResult(DataObjectBase):
    text: Annotated[str, FieldNumber(1)]
    results: Annotated[Optional[List[TokenClassification]], FieldNumber(2)]
    details: Annotated[
        Optional[TokenStreamDetails], FieldNumber(3)
    ]  # Should this be for stream


class ClassifiedGeneratedTextStreamResult(ClassifiedGeneratedTextResult):
    processed_index: Annotated[int, FieldNumber(4)]
