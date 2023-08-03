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
from typing import List

# First Party
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

    start: int
    end: int
    text: str


@dataobject(package=NLP_PACKAGE)
class TokenizationResult(DataObjectBase):
    results: List[Token]
