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
"""Data structures for text similarity predictions.
"""
# Standard
from typing import List

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from ....core.data_model import DataObjectBase, dataobject
from ....core.toolkit.errors import error_handler
from ...common.data_model import ProducerId

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(package="caikit_data_model.nlp")
class TextSimilarityPrediction(DataObjectBase):
    scores: Annotated[List[float], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]

    """The result of a similarity scores prediction."""

    def __init__(self, scores, producer_id=None):
        """Construct a new similarity scores prediction.

        Args:
            scores:  list(float)
                The similarity scores associated with this
                similarity prediction.
            producer_id:  ProducerId or None
                The block that produced this similarity prediction.
        """
        error.type_check("<NLP57445742E>", list, scores=scores)
        error.type_check(
            "<NLP85436555E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        super().__init__()
        self.scores = scores[:]
        self.producer_id = producer_id
