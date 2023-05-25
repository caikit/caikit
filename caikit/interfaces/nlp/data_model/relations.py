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
"""Data structures for relation recognition.
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
from . import entities, syntax

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(package="caikit_data_model.nlp")
class RelationMention(DataObjectBase):
    """Representation of a relationship mention."""

    type: Annotated[str, FieldNumber(1)]
    mention1: Annotated[entities.EntityMention, FieldNumber(2)]
    mention2: Annotated[entities.EntityMention, FieldNumber(3)]
    confidence: Annotated[float, FieldNumber(4)]
    producer_id: Annotated[ProducerId, FieldNumber(5)]
    subtype: Annotated[str, FieldNumber(6)]
    text: Annotated[str, FieldNumber(7)]

    """A relation mention consists of a pair of mentions and attributes representing the semantic
    relation between them, including type, subtype, etc. By default, is a binary relation.
    """

    # variable `type` is a builtin but cannot be renamed now without breaking the API
    # pylint: disable=redefined-builtin
    def __init__(
        self,
        type,
        mention1,
        mention2,
        producer_id=None,
        confidence=0.0,
        subtype="",
        text=None,
    ):
        """Constructs a new relation mention.

        Args:
            type:  str
                The type of the relation. Examples include 'parentOf', 'locatedAt'.
            mention1:  EntityMention
                The first mention
            mention2:  EntityMention
                The second mention
            producer_id:  ProducerId or None
                The block that produced this entities prediction.  None (default) indicates
                that this value is not set.
            confidence:  float
                The confidence score in the range [0, 1] that this mention is correct.
                The default confidence is 0.0.
            relation_subtype:  str
                The subtype of this relation mention (if defined by the type system).
                The empty string (default) denotes that this values is not set.
            text:  str or RawDocument or SyntaxPrediction or None
                The document that this mention refers to.  Used to extract the text for
                spans.  None (default) indicates that this field is not set.
        """
        error.type_check("<NLP92786787E>", str, type=type, subtype=subtype)
        error.type_check("<NLP11157666E>", entities.EntityMention, mention1=mention1)
        error.type_check("<NLP89010993E>", entities.EntityMention, mention2=mention2)
        error.value_check(
            "<NLP67780376E>",
            0.0 <= confidence <= 1.0,
            "`confidence` of `{}` is not between 0 and 1",
            confidence,
        )
        error.type_check("<NLP72418709E>", str, subtype=subtype)
        error.type_check(
            "<NLP00779192E>",
            str,
            syntax.RawDocument,
            syntax.SyntaxPrediction,
            allow_none=True,
            text=text,
        )

        self.type = type
        self.producer_id = producer_id
        self.mention1 = mention1
        self.mention2 = mention2
        self.confidence = confidence
        self.subtype = subtype
        self.text = (
            text.text
            if isinstance(text, (syntax.RawDocument, syntax.SyntaxPrediction))
            else text
        )


@dataobject(package="caikit_data_model.nlp")
class RelationMentionsPrediction(DataObjectBase):
    """Representation of a list of relationship mentions."""

    relation_mentions: Annotated[List[RelationMention], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]

    """A relation mention prediction is set of relation mentions extracted from a document."""

    def __init__(self, relation_mentions, producer_id=None):
        error.type_check_all(
            "<NLP50296922E>", RelationMention, relation_mentions=relation_mentions
        )
        error.type_check(
            "<NLP87490045E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        self.relation_mentions = relation_mentions
        self.producer_id = producer_id

    def get_relation_types(self):
        """Returns a list containing the str relation type for each relation mention."""
        return [relation.type for relation in self.relation_mentions]

    def get_relation_pairs_by_type(self):
        """Returns a dict that maps relation types to a list of relations each in the format
        ((mention1_text, mention1_type), (mention2_text, mention2_type))
        """
        relations_by_type = {}
        for relation in self.relation_mentions:
            relations_by_type.setdefault(relation.type, []).append(
                (
                    (relation.mention1.text, relation.mention1.type),
                    (relation.mention2.text, relation.mention2.type),
                )
            )
        return relations_by_type

    def get_relation_pairs(self):
        """Get a list of all relations in the format
        ((mention1_text, mention1_type), (mention2_text, mention2_type))
        """
        return [
            (
                (relation.mention1.text, relation.mention1.type),
                (relation.mention2.text, relation.mention2.type),
                relation.type,
            )
            for relation in self.relation_mentions
        ]


@dataobject(package="caikit_data_model.nlp")
class Relation(DataObjectBase):
    """Representation of a relation between entities.
    Entity representation with RelationMention is subject to change with a possible EntityPrediction"""

    entity1: Annotated[entities.Entity, FieldNumber(1)]
    entity2: Annotated[entities.Entity, FieldNumber(2)]
    relation_mentions: Annotated[List[RelationMention], FieldNumber(3)]
    type: Annotated[str, FieldNumber(4)]
    confidence: Annotated[float, FieldNumber(5)]

    """Construct a new relation -- a list of relation mentions that have the properties
    - all mention1 are in the same entity
    - all mention2 are in the same entity
    - all relations have the same type
    """

    # variable `type` is a builtin but cannot be renamed now without breaking the API
    # pylint: disable=redefined-builtin
    def __init__(self, entity1, entity2, relation_mentions, type, confidence=0.0):
        """Constructs a new relation.

        Args:
            entity1:  watson_nlp.data_model.entities.Entity
                Entity 1 in the entities pair of the relation (relation from)
            entity2:  watson_nlp.data_model.entities.Entity
                Entity 2 of the relation (relation to)
            relation_mentions:  list(RelationMention)
                A list of relation mentions that represent the same relation in a document
            type:  str
                The type of the relation (should be shared among relations)
            confidence:  float
                The confidence of the relation (average of the confidences of the relations)
        """
        error.type_check(
            "<NLP91543375E>", entities.Entity, entity1=entity1, entity2=entity2
        )
        error.type_check_all(
            "<NLP32926332E>", RelationMention, relation_mentions=relation_mentions
        )
        error.type_check("<NLP95921220E>", str, type=type)
        error.value_check(
            "<NLP32791216E>",
            0.0 <= confidence <= 1.0,
            "`confidence` of `{}` is not between 0 and 1",
            confidence,
        )

        self.entity1 = entity1
        self.entity2 = entity2
        self.relation_mentions = relation_mentions
        self.type = type
        self.confidence = confidence


@dataobject(package="caikit_data_model.nlp")
class RelationsPrediction(DataObjectBase):
    """Representation of a list of relations between entities."""

    relations: Annotated[List[Relation], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]

    """A relation prediction contains a number of relations in a document."""

    def __init__(self, relations, producer_id=None):
        """Construct a new RelationsPrediction.

        Args:
            relations:  list(Relation)
                A list of predicted relations extracted from a document.
            producer_id:  ProducerId or None
                The block that produced this relations prediction.
        """
        error.type_check_all("<NLP64346885E>", Relation, relations=relations)
        error.type_check(
            "<NLP64844035E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        self.relations = relations
        self.producer_id = producer_id

    def get_relation_types(self):
        """Returns a list containing the str relation type for each relation."""
        return [relation.type for relation in self.relations]

    def get_relation_pairs_by_type(self):
        """Returns a dict that maps relation types to a list of relations each in the format
        { type: [((entity1_text, entity1_type), (entity2_text, entity2_type))] }
        """
        relations_by_type = {}
        for relation in self.relations:
            relations_by_type.setdefault(relation.type, []).append(
                (
                    (relation.entity1.text, relation.entity1.type),
                    (relation.entity2.text, relation.entity2.type),
                )
            )
        return relations_by_type

    def get_relation_pairs(self):
        """Get a list of all relations in the format
        [(entity1_text, entity1_type), (entity2_text, entity2_type), relation)]
        """
        return [
            (
                (relation.entity1.text, relation.entity1.type),
                (relation.entity2.text, relation.entity2.type),
                relation.type,
            )
            for relation in self.relations
        ]
