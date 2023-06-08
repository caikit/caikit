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
from typing import List

# First Party
import alog

# Local
from ....core import DataObjectBase, dataobject
from ...common.data_model import ProducerId
from .package import NLP_PACKAGE

log = alog.use_channel("DATAM")


@dataobject(package=NLP_PACKAGE)
class StopReason(Enum):
    NOT_FINISHED = 0
    MAX_TOKENS = 1
    EOS_TOKEN = 2
    CANCELLED = 3
    TIME_LIMIT = 4
    STOP_SEQUENCE = 5


@dataobject(package=NLP_PACKAGE)
class GeneratedResult(DataObjectBase):
    generated_text: str
    generated_token_count: int
    stop_reason: StopReason
    producer_id: ProducerId
