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
"""
This module holds the Task definitions for all common NLP tasks
"""

# Standard
from typing import Iterable, List

# Local
from ...core import TaskBase, task
from ...core.data_model.json_dict import JsonDict
from .data_model import SentenceSimilarityResult, SentenceSimilarityResults
from .data_model.classification import (
    ClassificationResults,
    ClassifiedGeneratedTextResult,
    ClassifiedGeneratedTextStreamResult,
    TokenClassificationResults,
    TokenClassificationStreamResult,
)
from .data_model.embedding_vectors import EmbeddingResult, EmbeddingResults
from .data_model.reranker import RerankResult, RerankResults
from .data_model.text import TokenizationResults, TokenizationStreamResult
from .data_model.text_generation import GeneratedTextResult, GeneratedTextStreamResult


@task(
    unary_parameters={"text": str},
    unary_output_type=GeneratedTextResult,
    streaming_output_type=Iterable[GeneratedTextStreamResult],
)
class TextGenerationTask(TaskBase):
    """The Text Generation Task is responsible for taking input prompting text
    and generating additional text from that prompt.
    """


@task(
    required_parameters={"text": str},
    output_type=ClassificationResults,
)
class TextClassificationTask(TaskBase):
    """The text classification task is responsible for assigning a label or class to text."""


@task(
    unary_parameters={"text": str},
    streaming_parameters={"text_stream": Iterable[str]},
    unary_output_type=TokenClassificationResults,
    streaming_output_type=Iterable[TokenClassificationStreamResult],
)
class TokenClassificationTask(TaskBase):
    """The token classification task is responsible for assigning a label to individual
    tokens in a document.
    """


@task(
    unary_parameters={"text": str},
    streaming_parameters={"text_stream": Iterable[str]},
    unary_output_type=TokenizationResults,
    streaming_output_type=Iterable[TokenizationStreamResult],
)
class TokenizationTask(TaskBase):
    """The tokenization task is responsible for splitting a document into tokens."""


@task(
    unary_parameters={"text": str},
    unary_output_type=ClassifiedGeneratedTextResult,
    streaming_output_type=Iterable[ClassifiedGeneratedTextStreamResult],
)
class ClassificationWithTextGenerationTask(TaskBase):
    """The classification with text generation task is responsible for taking
    input prompting text, generating additional text from that prompt and classifying
    the generated text based on detectors.
    """


@task(
    required_parameters={"text": str},
    output_type=EmbeddingResult,
)
class EmbeddingTask(TaskBase):
    """Return a text embedding for the input text string"""


@task(
    required_parameters={"texts": List[str]},
    output_type=EmbeddingResults,
)
class EmbeddingTasks(TaskBase):
    """Return a text embedding for each text string in the input list"""


@task(
    required_parameters={
        "documents": List[JsonDict],
        "query": str,
    },
    output_type=RerankResult,
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
    output_type=RerankResults,
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


@task(
    required_parameters={"source_sentence": str, "sentences": List[str]},
    output_type=SentenceSimilarityResult,
)
class SentenceSimilarityTask(TaskBase):
    """Compare the source_sentence to each of the sentences.
    Result contains a list of scores in the order of the input sentences.
    """


@task(
    required_parameters={"source_sentences": List[str], "sentences": List[str]},
    output_type=SentenceSimilarityResults,
)
class SentenceSimilarityTasks(TaskBase):
    """Compare each of the source_sentences to each of the sentences.
    Returns a list of results in the order of the source_sentences.
    Each result contains a list of scores in the order of the input sentences.
    """
