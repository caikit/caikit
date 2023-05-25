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
"""Data structures for topic modeling.
"""
# Standard
from typing import List

# Third Party
import numpy as np

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from ....core.data_model import DataObjectBase, dataobject
from ....core.toolkit import error_handler
from ...common.data_model import ProducerId
from .text_primitives import NGram

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(package="caikit_data_model.nlp")
class TopicPhrase(DataObjectBase):
    text: Annotated[str, FieldNumber(1)]
    distance: Annotated[float, FieldNumber(2)]

    """A string sequence of multiple words coming from the original document
    and a distance from the topic center.
    """

    def __init__(self, text, distance):
        """Construct a single TopicPhrase consisting on a piece of text and a value.

        Args:
            text: str
                A string of text consisting of few or many words.
            distance: float
                A value indicating how far the associated text is to the cluster.
        """
        super().__init__()

        error.type_check("<NLP11507333E>", str, text=text)
        error.type_check("<NLP90124148E>", float, relevance=distance)

        self.text = text
        self.distance = distance


@dataobject(package="caikit_data_model.nlp")
class Topic(DataObjectBase):
    name: Annotated[str, FieldNumber(1)]
    score: Annotated[float, FieldNumber(2)]
    size: Annotated[np.uint32, FieldNumber(3)]
    ngrams: Annotated[List[NGram], FieldNumber(4)]
    snippets: Annotated[List[TopicPhrase], FieldNumber(5)]
    sentences: Annotated[List[TopicPhrase], FieldNumber(6)]
    producer_id: Annotated[ProducerId, FieldNumber(7)]

    """A Topic description of a document."""

    def __init__(
        self, name, score, size, ngrams, snippets=None, sentences=None, producer_id=None
    ):
        """Construct a single Topic instance.

        Args:
            name: str
                Text of n concatenated words to generally describe generally the topic.
            score: float
                Value describing the cohesiveness of the text to the topic title.
            size: int
                The number of documents that are classified as this topic.
            ngrams: list(NGram)
                A set of NGram values with each internal n-gram being a set of strings.
            snippets: list(TopicPhrase)
                A set of partial sentence text tokens and their overall cluster distance.
            sentences: list(TopicPhrase)
                A set of total sentences composed of text tokens and their overall cluster distance.
            producer_id:  ProducerId or None
                The block that produced this classification prediction.
        """
        super().__init__()

        error.type_check("<NLP25114056E>", str, name=name)
        error.type_check("<NLP29506904E>", float, score=score)
        error.type_check("<NLP39392665E>", int, size=size)
        # Should be a list of n-grams
        error.type_check("<NLP50038122E>", list, allow_none=True, ngrams=ngrams)
        error.type_check_all("<NLP66260011E>", NGram, ngrams=ngrams)

        # Both of these should be lists of topic-phrases
        error.type_check("<NLP51138722E>", list, allow_none=True, snippets=snippets)
        error.type_check_all("<NLP33210011E>", TopicPhrase, snippets=snippets)
        error.type_check("<NLP93964097E>", list, allow_none=True, sentences=sentences)
        error.type_check_all("<NLP33210311E>", TopicPhrase, sentences=sentences)

        self.name = name
        self.score = score
        self.size = size
        self.ngrams = ngrams
        self.snippets = [] if snippets is None else snippets
        self.sentences = [] if sentences is None else sentences
        self.producer_id = producer_id


@dataobject(package="caikit_data_model.nlp")
class TopicsPrediction(DataObjectBase):
    topics: Annotated[List[Topic], FieldNumber(1)]

    """A set of Topics describing the original document."""

    def __init__(self, topics):
        """Construct a topic Predictions.

        Args:
            topics: list (Topic)
                A list of instantiated Topics.
        """
        super().__init__()

        error.type_check("<NLP13527993E>", list, topics=topics)

        self.topics = topics
