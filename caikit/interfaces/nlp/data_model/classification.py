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
import alog

# Local
from ....core import DataObjectBase, dataobject
from .package import NLP_PACKAGE
from .text_generation import TokenStreamDetails

log = alog.use_channel("DATAM")


@dataobject(package=NLP_PACKAGE)
class ClassificationTrainRecord(DataObjectBase):
    text: str
    labels: List[str]


@dataobject(package=NLP_PACKAGE)
class Classification(DataObjectBase):
    label: str
    score: float


@dataobject(package=NLP_PACKAGE)
class ClassificationResult(DataObjectBase):
    results: List[Classification]


# NOTE: This is meant to align with the HuggingFace token classification task:
# https://huggingface.co/docs/transformers/tasks/token_classification#inference
# The field `word` does not necessarily correspond to a single "word",
# and `entity` may not always be applicable beyond "entity" in the NER
# (named entity recognition) sense
@dataobject(package=NLP_PACKAGE)
class TokenClassification(DataObjectBase):
    start: int
    end: int
    word: str  # could be thought of as text
    entity: str  # could be thought of as label
    entity_group: str  # could be thought of as aggregate label, if applicable
    score: float
    token_count: Optional[int]


@dataobject(package=NLP_PACKAGE)
class TokenClassificationResult(DataObjectBase):
    results: List[TokenClassification]


# Streaming result that indicates up to where in stream is processed
@dataobject(package=NLP_PACKAGE)
class StreamingTokenClassificationResult(TokenClassificationResult):
    # Result index up to which text is processed
    processed_index: int


@dataobject(package=NLP_PACKAGE)
class ClassifiedGeneratedTextResult(DataObjectBase):
    text: str
    results: Optional[List[TokenClassification]]
    details: Optional[TokenStreamDetails]  # Should this be for stream


class ClassifiedGeneratedTextStreamResult(ClassifiedGeneratedTextResult):
    processed_index: int
