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
"""Data structures for noun phrase extraction.
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
from . import text_primitives

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(package="caikit_data_model.nlp")
class NounPhrase(DataObjectBase):
    """Representation of a noun phrase and any associated annotations."""

    span: Annotated[text_primitives.Span, FieldNumber(1)]

    """A single noun phrase."""

    def __init__(self, span):
        """Construct a new noun phrase.

        Args:
            span:  Span or 2-tuple (int, int)
                The location of this noun phrase in terms (begin, end) utf
                codepoint offsets into the text.
        """
        error.type_check("<NLP89554815E>", tuple, text_primitives.Span, span=span)

        super().__init__()
        self.span = text_primitives.Span(*span) if isinstance(span, tuple) else span

    @classmethod
    def from_proto(cls, proto):
        span = text_primitives.Span.from_proto(proto.span)
        return cls(span=span)

    def fill_proto(self, proto):
        self.span.fill_proto(proto.span)
        return proto

    @property
    def text(self):
        """The string text spanned by this noun phrase."""
        return self.span.text


@dataobject(package="caikit_data_model.nlp")
class NounPhrasesPrediction(DataObjectBase):
    """The full set of noun phrases extracted for a document"""

    noun_phrases: Annotated[List[NounPhrase], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]

    """The result of a noun_phrases prediction."""

    def __init__(self, noun_phrases, producer_id=None):
        """Construct a new noun_phrases prediction.

        Args:
            noun_phrases:  list(noun_phrase)
                The noun_phrases (predictions) assocatied with this
                noun_phrases prediction.
            producer_id:  ProducerId or None
                The block that produced this noun_phrases prediction.
        """
        error.type_check_all("<NLP43515375E>", NounPhrase, noun_phrases=noun_phrases)
        error.type_check(
            "<NLP21319008E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        super().__init__()
        self.noun_phrases = sorted(noun_phrases, key=lambda k: k.span.begin)
        self.producer_id = producer_id
