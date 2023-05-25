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
"""Data structures for concept prediction.
"""

# Standard
from typing import List
import re

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
class Concept(DataObjectBase):
    """A single concept."""

    text: Annotated[str, FieldNumber(1)]
    relevance: Annotated[float, FieldNumber(2)]
    dbpedia_resource: Annotated[str, FieldNumber(3)]

    """A single concept."""

    # regex for verifying the URI format of dbpedia resources
    dbpedia_format_regex = re.compile(
        r"^http(:?s)?://(:?..+\.)?dbpedia\.org/resource/(:?.+)$"
    )

    def __init__(self, text, relevance=0.0, dbpedia_resource=""):
        """Construct a new concept.

        Args:
            text:  str
                The canonical text name for this concept.
            relevance:  float
                The relevance of this concept in the range [0, 1] relative to the document
                it was extracted from.  The default relevance scores is 0.0.
            dbpedia_resource:  str (url)
                Text of URL link to the corresponding DBPedia resource.
        """
        error.type_check(
            "<NLP76174661E>", str, text=text, dbpedia_resource=dbpedia_resource
        )
        error.value_check(
            "<NLP98684345E>",
            0.0 <= relevance <= 1.0,
            "`relevance` of `{}` is not between 0 and 1",
            relevance,
        )
        error.value_check(
            "<NLP76383322E>",
            not dbpedia_resource or self.dbpedia_format_regex.match(dbpedia_resource),
            "`dbpedia_resource` of `{}` is invalid",
            dbpedia_resource,
        )

        super().__init__()
        self.text = text
        self.relevance = relevance
        self.dbpedia_resource = dbpedia_resource

    def __lt__(self, other):
        """Compare concepts by relevance."""
        return self.relevance < other.relevance


@dataobject(package="caikit_data_model.nlp")
class ConceptsPrediction(DataObjectBase):
    """A concepts prediction generated from a document and consisting multiple concepts."""

    concepts: Annotated[List[Concept], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]

    """The result of a concepts prediction contains a number of predicted concepts relative to a
    given document.
    """

    def __init__(self, concepts, producer_id=None):
        """Construct a new concepts prediction.

        Args:
            concepts:  list(Concept)
                The concepts assocatied with this predicted to be relevant to a given document.
            producer_id:  ProducerId or None
                The block that produced this concepts prediction.
        """
        error.type_check_all("<NLP49786309E>", Concept, concepts=concepts)
        error.type_check(
            "<NLP57582755E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        super().__init__()
        self.concepts = sorted(concepts, reverse=True)
        self.producer_id = producer_id
