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
"""Data structures for embedding vector representations"""

# Standard
from typing import List, Optional

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from ...common.data_model import ProducerId
from caikit.core import DataObjectBase, dataobject
from caikit.core.exceptions import error_handler

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(package="caikit_data_model.caikit_nlp")
class SentenceSimilarityScores(DataObjectBase):
    """Scores for a sentence similarity task"""

    scores: Annotated[List[float], FieldNumber(1)]


@dataobject(package="caikit_data_model.caikit_nlp")
class SentenceSimilarityResult(DataObjectBase):
    """Result for sentence similarity task"""

    result: Annotated[SentenceSimilarityScores, FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]
    input_token_count: Annotated[Optional[int], FieldNumber(3)]


@dataobject(package="caikit_data_model.caikit_nlp")
class SentenceSimilarityResults(DataObjectBase):
    """Results list for sentence similarity tasks"""

    results: Annotated[List[SentenceSimilarityScores], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]
    input_token_count: Annotated[Optional[int], FieldNumber(3)]
