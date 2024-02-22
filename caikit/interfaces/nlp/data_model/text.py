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
"""Data structures for text representations"""

# Standard
from typing import List, Optional

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from ....core import DataObjectBase, dataobject
from .package import NLP_PACKAGE

log = alog.use_channel("DATAM")


@dataobject(package=NLP_PACKAGE)
class Token(DataObjectBase):
    """Tokens here are the basic units of text. Tokens can be characters, words,
    sub-words, or other segments of text or code, depending on the method of
    tokenization chosen or the task being implemented.
    """

    start: Annotated[int, FieldNumber(1)]  # Beginning offset of the token
    end: Annotated[int, FieldNumber(2)]  # Ending offset of the token
    text: Annotated[str, FieldNumber(3)]  # Text referenced by this token


@dataobject(package=NLP_PACKAGE)
class TokenizationResults(DataObjectBase):
    """Tokenization result generated from a text."""

    results: Annotated[Optional[List[Token]], FieldNumber(1)]
    # The number of tokens
    # Note: Field number 4 chosen due to Fields 2 and 3 used below
    token_count: Annotated[Optional[int], FieldNumber(4)]


@dataobject(package=NLP_PACKAGE)
class TokenizationStreamResult(TokenizationResults):
    """
    Streaming tokenization result that indicates up to where in stream is processed.
    """

    processed_index: Annotated[
        int, FieldNumber(2)
    ]  # Result index up to which text is processed
    start_index: Annotated[int, FieldNumber(3)]  # Result start index for processed text
