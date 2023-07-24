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
"""Data structures for text generation representations"""

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
from ...common.data_model import ProducerId
from .package import NLP_PACKAGE

log = alog.use_channel("DATAM")


@dataobject(package=NLP_PACKAGE)
class FinishReason(Enum):
    NOT_FINISHED = 0
    MAX_TOKENS = 1
    EOS_TOKEN = 2
    CANCELLED = 3
    TIME_LIMIT = 4
    STOP_SEQUENCE = 5
    TOKEN_LIMIT = 6
    ERROR = 7


@dataobject(package=NLP_PACKAGE)
class GeneratedTextResult(DataObjectBase):
    generated_text: Annotated[str, FieldNumber(1)]
    generated_tokens: Annotated[int, FieldNumber(2)]
    finish_reason: Annotated[FinishReason, FieldNumber(3)]
    producer_id: Annotated[ProducerId, FieldNumber(4)]


@dataobject(package=NLP_PACKAGE)
class GeneratedToken(DataObjectBase):
    text: Annotated[str, FieldNumber(1)]
    id: Annotated[Optional[np.uint32], FieldNumber(2)]
    logprob: Annotated[Optional[float], FieldNumber(3)]
    special: Annotated[Optional[bool], FieldNumber(4)]


@dataobject(package=NLP_PACKAGE)
class TokenStreamDetails(DataObjectBase):
    finish_reason: Annotated[FinishReason, FieldNumber(1)]
    generated_tokens: Annotated[np.uint32, FieldNumber(2)]
    seed: Annotated[np.uint64, FieldNumber(3)]


@dataobject(package=NLP_PACKAGE)
class GeneratedTextStreamResult(DataObjectBase):
    generated_text: Annotated[str, FieldNumber(1)]
    tokens: Annotated[Optional[List[GeneratedToken]], FieldNumber(2)]
    details: Annotated[Optional[TokenStreamDetails], FieldNumber(3)]
