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

# Standard
from typing import List, Optional

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber

# Local
from ...common.data_model import ProducerId
from caikit.core import DataObjectBase, dataobject
from caikit.core.data_model.json_dict import JsonDict


@dataobject(package="caikit_data_model.caikit_nlp")
class RerankScore(DataObjectBase):
    """The score for one document (one query)"""

    document: Annotated[Optional[JsonDict], FieldNumber(1)]
    index: Annotated[int, FieldNumber(2)]
    score: Annotated[float, FieldNumber(3)]
    text: Annotated[Optional[str], FieldNumber(4)]


@dataobject(package="caikit_data_model.caikit_nlp")
class RerankScores(DataObjectBase):
    """Scores for a query in a rerank task.
    This is a list of n ReRankScore where n is based on top_n documents and each score indicates
    the relevance of that document for this query. Results are ordered most-relevant first.
    """

    query: Annotated[Optional[str], FieldNumber(1)]
    scores: Annotated[List[RerankScore], FieldNumber(2)]


@dataobject(package="caikit_data_model.caikit_nlp")
class RerankResult(DataObjectBase):
    """Result for one query in a rerank task.
    This is a list of n ReRankScore where n is based on top_n documents and each score indicates
    the relevance of that document for this query. Results are ordered most-relevant first.
    """

    result: Annotated[RerankScores, FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]
    input_token_count: Annotated[Optional[int], FieldNumber(3)]


@dataobject(package="caikit_data_model.caikit_nlp")
class RerankResults(DataObjectBase):
    """Results list for rerank tasks (supporting multiple queries).
    For multiple queries, each one has a RerankQueryResult (ranking the documents for that query).
    """

    results: Annotated[List[RerankScores], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]
    input_token_count: Annotated[Optional[int], FieldNumber(3)]
