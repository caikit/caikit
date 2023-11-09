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
from typing import List

# Local
from caikit.core import TaskBase, task
from caikit.core.data_model.json_dict import JsonDict
from caikit.interfaces.nlp.data_model.reranker import (
    RerankPredictions,
    RerankQueryResult,
)


@task(
    required_parameters={
        "documents": List[JsonDict],
        "query": str,
    },
    output_type=RerankQueryResult,
)
class RerankTask(TaskBase):
    """Returns an ordered list ranking the most relevant documents for the query

    Required parameters:
        query: The search query
        documents: JSON documents containing "text" or alternative "_text" to search
    Returns:
        The top_n documents in order of relevance (most relevant first).
        For each, a score and document index (position in input) is returned.
        The original document JSON is returned depending on optional args.
        The top_n optional parameter limits the results when used.
    """


@task(
    required_parameters={
        "documents": List[JsonDict],
        "queries": List[str],
    },
    output_type=RerankPredictions,
)
class RerankTasks(TaskBase):
    """Returns an ordered list for each query ranking the most relevant documents for the query

    Required parameters:
        queries: The search queries
        documents: JSON documents containing "text" or alternative "_text" to search
    Returns:
        Results in order of the queries.
        In each query result:
            The query text is optionally included for visual convenience.
            The top_n documents in order of relevance (most relevant first).
            For each, a score and document index (position in input) is returned.
            The original document JSON is returned depending on optional args.
            The top_n optional parameter limits the results when used.
    """
