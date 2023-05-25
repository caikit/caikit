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
"""Data structures for text generation.
"""
# Standard
from enum import Enum

# Third Party
import numpy as np

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from ....core.data_model import DataObjectBase, dataobject
from ....core.toolkit.errors import error_handler
from ...common.data_model import ProducerId

log = alog.use_channel("DATAM")


@dataobject(package="caikit_data_model.nlp")
class StopReason(Enum):
    """
    Syntax analysis parsers that can be invoked.
    """

    NOT_FINISHED = 0
    MAX_TOKENS = 1
    EOS_TOKEN = 2
    CANCELLED = 3
    TIME_LIMIT = 4
    STOP_SEQUENCE = 5
    ERROR = 6


error = error_handler.get(log)


@dataobject(package="caikit_data_model.nlp")
class GeneratedResult(DataObjectBase):
    """A text generation prediction from an input prompt."""

    text: Annotated[str, FieldNumber(1)]
    stop_reason: Annotated[StopReason, FieldNumber(2)]
    generated_token_count: Annotated[np.uint64, FieldNumber(3)]
    producer_id: Annotated[ProducerId, FieldNumber(4)]

    def __init__(
        self, text, stop_reason=None, generated_token_count=None, producer_id=None
    ):
        """Text generation result.

        Args:
            text: str
                The generated string text.
            stop_reason: (int) enums.StopReason
                Enum value corresponding to a Stop Reason tag.  None (default)
                indicates that this value has not been set.
            generated_token_count: int or None
                The generated token sequence count.  None (default) indicates
                that this value has not been set.
            producer_id: ProducerId or None
                Block that generates text
        """
        error.type_check("<NLP12543906E>", str, text=text)
        error.type_check(
            "<NLP41543906E>", int, allow_none=True, stop_reason=stop_reason
        )
        error.type_check(
            "<NLP21243906E>",
            int,
            allow_none=True,
            generated_token_count=generated_token_count,
        )
        error.type_check(
            "<NLP19248586E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        object.__setattr__(self, "_text", text)
        object.__setattr__(self, "_stop_reason", stop_reason)
        object.__setattr__(self, "_generated_token_count", generated_token_count)
        object.__setattr__(self, "_producer_id", producer_id)
