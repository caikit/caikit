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
"""Basic primtivies for representing text including the `Span` primitive.
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

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(package="watson_core_data_model.nlp")
class Span(DataObjectBase):
    """Span within a given body of text represented as start and end.
    This is the fundamental data structure for representing a region of text.
    Note:  Spans are always in relative to a text document, i.e., not in reference
    to a sentence or paragraph, et cetra.  These other structures can be
    cross-referenced instead, which keeps our notion of a span consistent."""

    begin: Annotated[np.uint32, FieldNumber(1)]
    end: Annotated[np.uint32, FieldNumber(2)]
    text: Annotated[str, FieldNumber(3)]

    """A span object describing a location, in terms of utf codepoints, into a text document."""

    def __init__(self, begin, end, text=""):
        """Construct a new text span.

        Args:
            begin:  int
                The starting offset into the text.
            end:  int
                The end offset into the text.
            text:  str
                The text contained by this token.  This should generally not be set manually and may
                be automatically filled in by other data structures.  Instead, the span should be
                considered the ground truth for the content and of this token in reference to a
                document.  Empty string (default) indicates that this value has not been set.

        Notes:
            In general, a Span is the basic currency of a part of a text document.  The actual text
            is typically extracted and stored automatically by other data structures and is *not*
            populated in protobufs.
        """
        # NOTE:  We do not validate the types passed to `Span` for performance reasons.  Since large
        #   numbers of `Span` objects may be created very quickly, type checks cause a burdonsome
        #   performance slowdown.  The value checks are necessary, however, to ensure that only
        #   valid spans are constructed.
        error.value_check(
            "<NLP76707577E>", 0 <= begin <= end, "invalid span `({}, {})`", begin, end
        )
        if text and (len(text) != (end - begin)):
            error(
                "<NLP36052454E>",
                ValueError(
                    "invalid `text` length of `{}` for span `({}, {})`".format(
                        len(text), begin, end
                    )
                ),
            )

        # NOTE: We do not call parent constructor here for performance reasons.
        # Instead, we set and initialize all necessary fields explicitly in the following.
        # super().__init__()

        # NOTE:  We do not validate fields definition here for performance reasons.
        object.__setattr__(self, "_begin", begin)
        object.__setattr__(self, "_end", end)
        object.__setattr__(self, "_text", text)

    @classmethod
    def from_proto(cls, proto):
        return cls(proto.begin, proto.end, proto.text)

    def fill_proto(self, proto):
        proto.begin = self.begin
        proto.end = self.end
        proto.text = self.text

        return proto

    def slice_and_set_text(self, text, text_length=None):
        """Extract the text referenced by this Span object and place it into the text attribute,
        overwriting whatever was previously there.

        Args:
            text:  str
                The text for a larger document that this span is
                offset into.  The text will be extracted from this
                argument.
            text_length:  None | int
                The length of the document; this override can be useful to avoid execessive calls
                to len() for a given document.

        Examples:
            > token = dm.Token(dm.Span(0,5))
            > token.slice_and_set_text("Hello World!");
            > token.text
            'Hello'
        """
        text_len = text_length if text_length is not None else len(text)
        error.value_check(
            "<NLP95348099E>",
            self.end <= text_len,
            "span `({}, {})` end is beyond end of text with length `{}`",
            self.begin,
            self.end,
            text_len,
        )

        self.text = text[self.begin : self.end]
        return self

    def __call__(self):
        """Calling a locatable object directly returns a tuple(int, int)
        representation of this span's begin and end offsets.
        """
        return self.begin, self.end

    def __lt__(self, other):
        """Self comes before other, compare by begin offsets and break ties with end offset."""
        return self.begin < other.begin or (
            self.begin == other.begin and self.end < other.end
        )

    def __le__(self, other):
        """Self comes before or equals other, compare by begin offsets and break ties with
        end offset.
        """
        return self.begin < other.begin or (
            self.begin == other.begin and self.end <= other.end
        )

    def __gt__(self, other):
        """Self comes after other, compare by begin offsets and break ties with end offset."""
        return self.begin > other.begin or (
            self.begin == other.begin and self.end > other.end
        )

    def __ge__(self, other):
        """Self comes after or equals other, compare by begin offsets and break ties with
        end offset.
        """
        return self.begin > other.begin or (
            self.begin == other.begin and self.end >= other.end
        )

    def __ne__(self, other):
        """Self is not the same as other, i.e., begin and end are not exactly the same."""
        return self.begin != other.begin or self.end != other.end

    def __eq__(self, other):
        """Self is the same as other, i.e., begin and end are exactly the same."""
        return self.begin == other.begin and self.end == other.end

    def __contains__(self, other):
        """Self is inside of other."""
        return other.begin >= self.begin and other.end <= self.end

    def __len__(self):
        """The length of this span, end - begin."""
        return self.end - self.begin

    def __hash__(self):
        """Span is hashable, where we hash its string representation"""
        return hash(str(self))

    def __repr__(self):
        """A string representation of this span."""
        return "({}, {})".format(self.begin, self.end)

    def overlaps(self, other):
        """Check for any overlap with another span."""
        return self.begin < other.end and other.begin < self.end


@dataobject(package="watson_core_data_model.nlp")
class NGram(DataObjectBase):
    texts: Annotated[List[str], FieldNumber(1)]
    relevance: Annotated[float, FieldNumber(2)]

    """a set of n-grams with each n-gram series of strings and an associated weight"""

    def __init__(self, texts, relevance=None):
        """Construct an NGram instance consisting of a series of words and their relevance

        Args:
            texts: list (str)
                a single or multiple words
            relevance: float
                an optional score in [0, 1] indicating how relevant
                this ngram is to a document or document collection
        """
        # NOTE:  We do not validate types here for performance reasons.

        # NOTE: We do not call parent constructor here for performance reasons.
        # Instead, we set and initialize all necessary fields explicitly in the following.
        # super().__init__()

        # NOTE:  We do not validate fields definition here for performance reasons.
        object.__setattr__(self, "_texts", list(texts))
        object.__setattr__(self, "_relevance", relevance)
