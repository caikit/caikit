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
"""Data structures for keyword extraction.
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
from ....core.toolkit.errors import error_handler
from ...common.data_model import ProducerId
from . import syntax, text_primitives

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(package="caikit_data_model.nlp")
class Keyword(DataObjectBase):
    """A single keyword extraction from a document"""

    text: Annotated[str, FieldNumber(1)]
    relevance: Annotated[float, FieldNumber(2)]
    mentions: Annotated[List[text_primitives.Span], FieldNumber(3)]
    count: Annotated[np.uint32, FieldNumber(4)]

    """A single keyword prediction."""

    def __init__(self, text, relevance, mentions, document=None, count=None):
        """Construct a new keyword prediction.

        Args:
            text:  str
                The canonical text for this keyword.
            relevance:  float
                Relevance score, relative to the document it was extracted from, for this keyword,
                in the range [0, 1].
            mentions:  list(2-tuple) or list(text_primitives.Span)
                The mention spans describing the locations of this keyword in the document.
            document:  str or RawDocument or SyntaxPrediction or None
                The document that this mention refers to.  Used to extract the text for
                spans.  None (default) indicates that this field is not set.

        Notes:
            Text for mention spans will be extracted from the (optional) document argument.
            This extraction will overwrite any text already in the span objects.
        """
        error.type_check("<NLP74029391E>", str, text=text)
        error.value_check(
            "<NLP03946336E>",
            0.0 <= relevance <= 1.0,
            "`relevance` of `{}` is not between 0 and 1",
            relevance,
        )
        error.type_check_all(
            "<NLP38779508E>", tuple, text_primitives.Span, mentions=mentions
        )
        error.type_check(
            "<NLP15716500E>",
            str,
            syntax.RawDocument,
            syntax.SyntaxPrediction,
            allow_none=True,
            document=document,
        )

        super().__init__()
        self.text = text
        self.relevance = relevance

        # convert mentions from 2-tuples to spans if the first one is a tuple
        self.mentions = (
            mentions
            if not mentions or isinstance(mentions[0], text_primitives.Span)
            else [text_primitives.Span(*mention) for mention in mentions]
        )

        # sort mentions by mention offsets
        self.mentions = sorted(self.mentions)
        if count is None:
            self.count = len(self.mentions)
        else:
            self.count = count

        if document is not None:
            text = (
                document.text
                if isinstance(document, (syntax.RawDocument, syntax.SyntaxPrediction))
                else document
            )

            for mention in self.mentions:
                mention.slice_and_set_text(text)

    @classmethod
    def from_proto(cls, proto):
        text = proto.text
        relevance = proto.relevance
        mentions = [
            text_primitives.Span.from_proto(mention) for mention in proto.mentions
        ]
        count = proto.count
        return cls(text=text, relevance=relevance, mentions=mentions, count=count)

    def fill_proto(self, proto):
        proto.text = self.text
        proto.relevance = self.relevance
        proto.mentions.extend([mention.to_proto() for mention in self.mentions])
        proto.count = self.count
        return proto

    def __lt__(self, other):
        """Compare keywords by relevance score."""
        return self.relevance < other.relevance


@dataobject(package="caikit_data_model.nlp")
class KeywordsPrediction(DataObjectBase):
    """The full set of keywords extracted for a document"""

    keywords: Annotated[List[Keyword], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]

    """The result of a keywords prediction."""

    def __init__(self, keywords, producer_id=None):
        """Construct a new keywords prediction.

        Args:
            keywords:  list(Keyword)
                The keywords (predictions) assocatied with this
                keywords prediction.
            producer_id:  ProducerId or None
                The block that produced this keywords prediction.
        """
        error.type_check_all("<NLP81954717E>", Keyword, keywords=keywords)
        error.type_check(
            "<NLP84926549E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        super().__init__()
        self.keywords = sorted(keywords, reverse=True)
        self.producer_id = producer_id

    def get_keyword_texts(self):
        """Get a string representations of each keyword in this prediction.

        Returns:
            list(str):
                A list containing the string representation of each keyword.
        """
        return [keyword.text for keyword in self.keywords]
