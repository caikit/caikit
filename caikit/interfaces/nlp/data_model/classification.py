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
from enum import Enum
from typing import List, Optional

# Third Party
import numpy as np

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from ....core import DataObjectBase, dataobject
from .package import NLP_PACKAGE
from .text_generation import FinishReason, GeneratedToken

log = alog.use_channel("DATAM")


@dataobject(package=NLP_PACKAGE)
class InputWarningReason(Enum):
    UNSUITABLE_INPUT = 0


@dataobject(package=NLP_PACKAGE)
class InputWarning(DataObjectBase):
    """Input Warning data object, which returns a reason and message associated with warnings
    to issue to a user that causes errors (such as failed text generation)
    """

    id: Annotated[InputWarningReason, FieldNumber(1)]  # id of input error
    message: Annotated[str, FieldNumber(2)]  # Error message detailing Warning


@dataobject(package=NLP_PACKAGE)
class ClassificationTrainRecord(DataObjectBase):
    """A classification training record consisting of a single train instance."""

    text: Annotated[str, FieldNumber(1)]  # Text to be classified
    labels: Annotated[
        List[str], FieldNumber(2)
    ]  # Class labels to be learnt for the text


@dataobject(package=NLP_PACKAGE)
class ClassificationResult(DataObjectBase):
    """A single classification prediction."""

    label: Annotated[str, FieldNumber(1)]  # Predicted relevant class name
    score: Annotated[
        float, FieldNumber(2)
    ]  # The confidence-like score of this prediction in [0, 1]


@dataobject(package=NLP_PACKAGE)
class ClassificationResults(DataObjectBase):
    """Classification results generated from a text and consisting multiple classes."""

    results: Annotated[
        List[ClassificationResult], FieldNumber(1)
    ]  # List of classifications for a text


# NOTE: This is meant to align with the HuggingFace token classification task:
# https://huggingface.co/docs/transformers/tasks/token_classification#inference
# The field `word` does not necessarily correspond to a single "word",
# and `entity` may not always be applicable beyond "entity" in the NER
# (named entity recognition) sense
@dataobject(package=NLP_PACKAGE)
class TokenClassificationResult(DataObjectBase):
    """A single token classification prediction."""

    start: Annotated[int, FieldNumber(1)]  # Beginning offset of the token
    end: Annotated[int, FieldNumber(2)]  # Ending offset of the token
    word: Annotated[str, FieldNumber(3)]  # Text referenced by this token
    entity: Annotated[
        str, FieldNumber(4)
    ]  # Predicted relevant class name for the token
    entity_group: Annotated[str, FieldNumber(5)]  # Aggregate label, if applicable
    score: Annotated[
        float, FieldNumber(6)
    ]  # The confidence-like score of this classification prediction in [0, 1]
    token_count: Annotated[
        Optional[int], FieldNumber(7)
    ]  # Length of tokens in the text


@dataobject(package=NLP_PACKAGE)
class TokenClassificationResults(DataObjectBase):
    """Token classification results generated from a text and consisting multiple classes."""

    results: Annotated[List[TokenClassificationResult], FieldNumber(1)]


@dataobject(package=NLP_PACKAGE)
class TokenClassificationStreamResult(TokenClassificationResults):
    """
    Streaming token classification results that indicates up to where in stream is processed.
    """

    processed_index: Annotated[
        int, FieldNumber(2)
    ]  # Result index up to which text is processed
    start_index: Annotated[int, FieldNumber(3)]  # Result start index for processed text


@dataobject(package=NLP_PACKAGE)
class ClassifiedGeneratedTextResult(DataObjectBase):
    """Classification result on text produced by a text generation model, contains
    information from the original text generation output as well as the result of
    classification on the generated text.
    """

    @dataobject(package=NLP_PACKAGE)
    class TextGenTokenClassificationResults(DataObjectBase):
        input: Annotated[Optional[List[TokenClassificationResult]], FieldNumber(10)]
        output: Annotated[Optional[List[TokenClassificationResult]], FieldNumber(20)]

    generated_text: Annotated[Optional[str], FieldNumber(1)]  # The generated text
    token_classification_results: Annotated[
        Optional[TextGenTokenClassificationResults], FieldNumber(2)
    ]  # Token classification results for input and generated text
    finish_reason: Annotated[
        Optional[FinishReason], FieldNumber(3)
    ]  # Reason as to why text generation stopped
    generated_token_count: Annotated[
        Optional[int], FieldNumber(4)
    ]  # Length of generated tokens sequence
    seed: Annotated[
        Optional[np.uint64], FieldNumber(5)
    ]  # The random seed used for text generation
    input_token_count: Annotated[Optional[int], FieldNumber(6)]
    warnings: Annotated[
        Optional[List[InputWarning]], FieldNumber(9)
    ]  # Warning to user in the event of input errors
    tokens: Annotated[Optional[List[GeneratedToken]], FieldNumber(10)]
    input_tokens: Annotated[Optional[List[GeneratedToken]], FieldNumber(11)]


@dataobject(package=NLP_PACKAGE)
class ClassifiedGeneratedTextStreamResult(ClassifiedGeneratedTextResult):
    """
    Streaming classification on generated text result that indicates up to where in stream
    is processed.
    """

    processed_index: Annotated[
        Optional[int], FieldNumber(7)
    ]  # Result index up to which text is processed
    start_index: Annotated[int, FieldNumber(8)]  # Result start index for processed text
