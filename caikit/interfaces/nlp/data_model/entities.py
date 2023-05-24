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
"""Data structures for named entity recognition.
"""
# Standard
from enum import Enum
from typing import List

# Third Party
import numpy as np

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from ....core.data_model import DataObjectBase, dataobject
from ....core.toolkit.errors import DataValidationError, error_handler
from ...common.data_model import ProducerId
from . import syntax, text_primitives

log = alog.use_channel("DATAM")


@dataobject(package="watson_core_data_model.nlp")
class EntityMentionType(Enum):
    """
    Enum for mention type.
    """

    MENTT_UNSET = 0
    MENTT_NAM = 1
    MENTT_NOM = 2
    MENTT_PRO = 3
    MENTT_NONE = 4


@dataobject(package="watson_core_data_model.nlp")
class EntityMentionClass(Enum):
    """
    Enum for mention class.
    """

    MENTC_UNSET = 0
    MENTC_SPC = 1
    MENTC_NEG = 2
    MENTC_GEN = 3


error = error_handler.get(log)


@dataobject(package="watson_core_data_model.nlp")
class EntityMention(DataObjectBase):
    """Representation of an entity mention with an extracted type and a confidence score."""

    span: Annotated[text_primitives.Span, FieldNumber(1)]
    type: Annotated[str, FieldNumber(2)]
    producer_id: Annotated[ProducerId, FieldNumber(3)]
    confidence: Annotated[float, FieldNumber(4)]
    mention_type: Annotated[EntityMentionType, FieldNumber(5)]
    mention_class: Annotated[EntityMentionClass, FieldNumber(6)]
    role: Annotated[str, FieldNumber(7)]

    """An entity mention is a single, specific reference to an entity in a text document."""

    # variable `type` is a builtin but cannot be renamed now without breaking the API
    # pylint: disable=redefined-builtin
    def __init__(
        self,
        span,
        type,
        producer_id=None,
        confidence=0.0,
        mention_type=0,
        mention_class=0,
        role="",
        document=None,
    ):
        """Construct a new entity mention.

        Args:
            span:  Span or 2-tuple (int, int)
                The location of this mention in terms (begin, end) utf codepoint
                offsets into the text.
            type:  str
                The string label or type of this mention, e.g., Person, Location or Facility
            producer_id:  ProducerId or None
                The block that produced this entities prediction.  None (default) indicates
                that this value is not set.
            confidence:  float
                The confidence score in the range [0, 1] that this mention is correct.
                The default confidence is 0.0.
            mention_type:  int (enums.EntityMentionType)
                Denotes if this mention is named, nominal, pronominal, et cetra
                A value of 0 indicates that this field is not set.
            mention_class:  int (enums.EntityMentionClass)
                Denotes if this mention is specific, negated, et cetra
                A value of 0 indicates that this field is not set.
            role:  str
                The role of this mention.
                The empty string (default) denotes that this values is not set.
            document:  str or RawDocument or SyntaxPrediction or None
                The document that this mention refers to.  Used to extract the text for
                spans.  None (default) indicates that this field is not set.

        Notes:
            Text for mention spans will be extracted from the (optional) document argument.
            This extraction will overwrite any text already in the span objects.
        """
        error.type_check("<NLP63170053E>", tuple, text_primitives.Span, span=span)
        error.type_check("<NLP62433615E>", str, type=type, role=role)
        error.type_check(
            "<NLP62194002E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )
        error.value_check(
            "<NLP07139230E>",
            0.0 <= confidence <= 1.0,
            "`confidence` of `{}` is not between 0 and 1",
            confidence,
        )
        error.type_check(
            "<NLP39500312E>",
            int,
            mention_type=mention_type,
            mention_class=mention_class,
        )
        error.type_check(
            "<NLP20994245E>",
            str,
            syntax.RawDocument,
            syntax.SyntaxPrediction,
            allow_none=True,
            document=document,
        )

        super().__init__()
        self.span = text_primitives.Span(*span) if isinstance(span, tuple) else span
        self.type = type
        self.producer_id = producer_id
        self.confidence = confidence
        self.mention_type = mention_type
        self.mention_class = mention_class
        self.role = role

        if document is not None:
            text = (
                document.text
                if isinstance(document, (syntax.RawDocument, syntax.SyntaxPrediction))
                else document
            )

            self.span.slice_and_set_text(text)

    @property
    def text(self):
        """String text spanned by this entity mention."""
        return self.span.text

    @alog.logged_function(log.debug2)
    def compare_priority(
        self, other, model_priorities=None, favor_types=None, disfavor_types=None
    ):
        """Determine if this mention has higher priority than the other. If so, return True.

        Args:
            other:  EntityMention
                Other mention to compare priority against.
            model_priorities:  list
                List of `producer_id`s in order of priority. First `producer_id` in the list has the
                highest preference.
            favor_types:  list
                List of types that should be used for favoring (aka prefer anything over these
                types).
            disfavor_types:  list
                List of types that should be used for disfavoring (aka prefer anything over these
                types).

        Returns:
            bool
                True if `self` is higher priority than `other`.

        NOTE:
            Will lowercase all types for comparison. Also:
            Logic follows:
                1) check favor_types
                2) check disfavor_types
                3) check model priority
                4) prefer longer spans (only thing that runs if no kwargs provided)
                5) if nothing resulted in `return`, assume `self` has priority and return `True`
        """
        # there are a lot, but its for the best
        # pylint: disable=too-many-return-statements

        my_type = self.type.lower()
        other_type = other.type.lower()

        # 1) check favor_types and return if applies
        if favor_types:
            favor_types = [f_type.lower() for f_type in favor_types]

            # prefer favor_types over anything; do nothing if both are in or both are out
            self_in_favor = my_type in favor_types
            other_in_favor = other_type in favor_types

            if self_in_favor and not other_in_favor:
                return True
            if not self_in_favor and other_in_favor:
                return False

        # 2) check disfavor_types and return if applies
        if disfavor_types:
            disfavor_types = [d_type.lower() for d_type in disfavor_types]

            # prefer anything over disfavor_types; do nothing if both are in or both are out
            self_in_disfavor = my_type in disfavor_types
            other_in_disfavor = other_type in disfavor_types

            if not self_in_disfavor and other_in_disfavor:
                return True
            if self_in_disfavor and not other_in_disfavor:
                return False

        # 3) higher priority (aka smaller number) is better; skip if no map provided; also requires
        #       producer ids
        if model_priorities and self.producer_id and other.producer_id:
            error.type_check("<NLP48736174E>", list, model_priorities=model_priorities)

            my_priority = model_priorities.index(self.producer_id)
            other_priority = model_priorities.index(other.producer_id)

            if my_priority < other_priority:
                return True
            if my_priority > other_priority:
                return False

        # 4) prefer longer spans
        my_len = len(self.span)
        other_len = len(other.span)
        if my_len > other_len:
            return True
        if my_len < other_len:
            return False

        # 5) default to current being better if no above rule matches
        return True


@dataobject(package="watson_core_data_model.nlp")
class EntityMentionsPrediction(DataObjectBase):
    """An entity mentions prediction generated from a document and consisting of multiple entity mentions."""

    mentions: Annotated[List[EntityMention], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]

    """An entity mentions prediction contains a number of entity mentions extracted from a document."""

    def __init__(self, mentions, producer_id=None):
        """Construct a new EntityMentionsPrediction.

        Args:
            mentions:  list(EntityMentions)
                A list of predicted entities extracted from a document.

            producer_id:  ProducerId or None
                The block that produced this entities prediction.
        """
        error.type_check_all("<NLP66268518E>", EntityMention, mentions=mentions)
        error.type_check(
            "<NLP33831155E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        # mentions are sorted by span, i.e., order of occurence in the document
        self.mentions = sorted(mentions, key=lambda mention: mention.span)
        self.producer_id = producer_id

    def __add__(self, other):
        """Combine two different EntitiesPrediction's."""
        return EntityMentionsPrediction(
            self.mentions + other.mentions, self.producer_id + other.producer_id
        )

    def get_mention_types(self):
        """Returns a list of str mention types, one for each mention."""
        return [mention.type for mention in self.mentions]

    def get_mention_texts(self):
        """Returns a list of str mention texts, one for each mention."""
        return [mention.text for mention in self.mentions]

    def get_mention_pairs(self):
        """Returns a list of all mentions in as tuples with the format (mention_text, mention_type)."""
        return [(mention.text, mention.type) for mention in self.mentions]

    @alog.logged_function(log.debug)
    def resolve_conflicts(
        self, model_priorities=None, favor_types=None, disfavor_types=None
    ):
        """Resolve conflicts using model_priorities (if provided). It will sort the mentions and
        then use the `EntityMention.compare_priority` method to resolve conflicts (in case of
        overlap).

        Args:
            model_priorities:  list
                List of `producer_id`s in order of priority. First `producer_id` in the list has the
                highest preference.
            favor_types:  list
                List of types that should be used for favoring (aka prefer anything over these
                types). Defaults to being skipped.
            disfavor_types:  list
                List of types that should be used for disfavoring (aka prefer anything over these
                types). Defaults to being skipped.

        Returns:
            EntityMentionsPrediction
                Re-assigns `self.mentions` and returns the same object for consumption by user if
                desired.
        """
        # in-place sort of mentions by span, begin first and break ties by end
        self.mentions.sort(key=lambda x: x.span)

        # zero or one mention implies no conflicts; do nothing
        if len(self.mentions) < 2:
            return EntityMentionsPrediction(self.mentions)

        prev = self.mentions[0]
        resolved_mentions = [prev]

        for mention in self.mentions[1:]:
            # if there is no overlap, append and continue!
            if not mention.span.overlaps(prev.span):
                resolved_mentions.append(mention)

            else:
                current_wins = mention.compare_priority(
                    prev,
                    model_priorities=model_priorities,
                    favor_types=favor_types,
                    disfavor_types=disfavor_types,
                )
                # replace previous mention with current one if current conflicts and wins
                if current_wins:
                    resolved_mentions[-1] = mention

            prev = mention

        return EntityMentionsPrediction(resolved_mentions)


@dataobject(package="watson_core_data_model.nlp")
class EntityDisambiguation(DataObjectBase):
    """Entity disambiguation links an entity to external or additional resources."""

    name: Annotated[str, FieldNumber(1)]
    subtypes: Annotated[List[str], FieldNumber(2)]
    dbpedia_resource: Annotated[str, FieldNumber(3)]

    """An entity disambiguation links an entity to external or additional resources."""

    def __init__(self, name, subtypes=None, dbpedia_resource=""):
        """Construct a new entity disambiguation.

        Args:
            name:  str
                The canonical disambiguated entity name.

            subtypes:  list(str) or None
                A list of string subtype labels.  A value of None (default)
                indicates that this value is not set.

            dbpedia_resource:  str
                A string URI reference to a dbpedia resource, e.g.,
                http://dbpedia.org/resource/Barack_Obama
        """
        error.type_check(
            "<NLP63435450E>", str, name=name, dbpedia_resource=dbpedia_resource
        )
        error.type_check_all("<NLP72758750E>", str, allow_none=True, subtypes=subtypes)

        self.name = name
        self.subtypes = [] if subtypes is None else subtypes
        self.dbpedia_resource = "" if dbpedia_resource is None else dbpedia_resource


@dataobject(package="watson_core_data_model.nlp")
class Entity(DataObjectBase):
    """Message representing a merged entity aggregated from one or more mentions.

    NOTE: The canonical fields need not correlate to any individual mention,
    allowing for canonicalization and/or disambiguation in the aggregation
    process."""

    mentions: Annotated[List[EntityMention], FieldNumber(1)]
    text: Annotated[str, FieldNumber(2)]
    type: Annotated[str, FieldNumber(3)]
    confidence: Annotated[float, FieldNumber(5)]
    relevance: Annotated[float, FieldNumber(6)]
    disambiguation: Annotated[EntityDisambiguation, FieldNumber(7)]

    """An entity with merged mentions, a canonical name and a type."""

    # variable `type` is a builtin but cannot be renamed now without breaking the API
    # pylint: disable=redefined-builtin
    def __init__(
        self, mentions, text, type, confidence=0.0, relevance=0.0, disambiguation=None
    ):
        """Construct a new entity.

        Args:
            mentions:  list(EntityMention)
                A list of mentions refering to this entity.
            text:  str
                The canonical name of this entity.
            type:  str
                The string label or type of this entity, e.g., Person, Location or Facility
            confidence:  float
                The confidence score in the range [0, 1] that this entity is correct.
                The default confidence is 0.0.
            relevance:  float
                The relevance of this entity in the range [0, 1] relative to the document
                it was extracted from.  The default relevance scores is 0.0.
            disambiguation:  EntityDisambiguation or None
                A disambiguation, a.k.a. link, from this entity to external resources.
        """
        error.type_check_all("<NLP37785536E>", EntityMention, mentions=mentions)
        error.type_check("<NLP45172917E>", str, text=text, type=type)
        error.value_check(
            "<NLP57324547E>",
            0.0 <= confidence <= 1.0,
            "`confidence` of `{}` is not between 0 and 1",
            confidence,
        )
        error.value_check(
            "<NLP94788594E>",
            0.0 <= relevance <= 1.0,
            "`relevance` of `{}` is not between 0 and 1",
            relevance,
        )
        error.type_check(
            "<NLP51180404E>",
            EntityDisambiguation,
            allow_none=True,
            disambiguation=disambiguation,
        )

        self.mentions = mentions
        self.text = text
        self.type = type
        self.confidence = confidence
        self.relevance = relevance
        self.disambiguation = disambiguation


@dataobject(package="watson_core_data_model.nlp")
class EntitiesPrediction(DataObjectBase):
    """An entities prediction generated from a document and consisting of multiple entities."""

    entities: Annotated[List[Entity], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]

    """An entities prediction contains a number of entities extracted from a document."""

    def __init__(self, entities, producer_id=None):
        """Construct a new EntitiesPrediction.

        Args:
            entities:  list(Entity)
                A list of predicted entities extracted from a document.
            producer_id:  ProducerId or None
                The block that produced this entities prediction.
        """
        error.type_check_all("<NLP09725430E>", Entity, entities=entities)
        error.type_check(
            "<NLP28562262E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        self.entities = entities
        self.producer_id = producer_id

    def get_entity_types(self):
        """Return a list of str entity types, one for each entity."""
        return [entity.type for entity in self.entities]

    def get_entity_texts(self):
        """Return a list of str entity texts, one for each entity."""
        return [entity.text for entity in self.entities]

    def get_entity_mention_counts(self):
        """Return a list of 3-tuples, one for each entity, each with the following format
        (entity_text, entity_type, entity_count) where count is the number of entity mentions.
        """
        return [
            (entity.text, entity.type, len(entity.mentions)) for entity in self.entities
        ]

    def find_mentions(self, entity_text, entity_type):
        """Return a list of 2-tuples with the format (mention_text, mention_type) corresponding to
        entity mentions that match the arguments entity_text and entity_type.
        """
        return [
            (mention.text, mention.type)
            for entity in self.entities
            for mention in entity.mentions
            if entity.text == entity_text and entity.type == entity_type
        ]


@dataobject(package="watson_core_data_model.nlp")
class EntityMentionAnnotation(DataObjectBase):
    """An Entity Mention Annotation consisting of text, type and span of mention."""

    text: Annotated[str, FieldNumber(1)]
    type: Annotated[str, FieldNumber(2)]
    location: Annotated[text_primitives.Span, FieldNumber(3)]

    """An entity mention annotation"""

    def __init__(self, text, type, location):
        """Construct a new Entity Mention Annotation.

        Args:
            text: string
                The text of the entity mention.
            type: string
                The type of the entity mention.
            location: text_primitives.Span or tuple(begin: int, end: int)
                Span of the entity mention.
        """
        error.type_check("<NLP96844580E>", str, text=text)
        error.type_check("<NLP67844580E>", str, type=type)
        error.type_check(
            "<NLP86150053E>", tuple, text_primitives.Span, location=location
        )

        super().__init__()
        self.text = text
        self.type = type
        self.location = (
            text_primitives.Span(*location) if isinstance(location, tuple) else location
        )


@dataobject(package="watson_core_data_model.nlp")
class EntityMentionsTrainRecord(DataObjectBase):
    """A train record consisting of raw text and entity mentions."""

    text: Annotated[str, FieldNumber(1)]
    mentions: Annotated[List[EntityMentionAnnotation], FieldNumber(2)]
    language_code: Annotated[str, FieldNumber(3)]
    id: Annotated[np.uint64, FieldNumber(4)]

    """A Train record for Entity Mentions"""

    def __init__(self, text, mentions, language_code=None, id=None):
        """Construct a new Entity Mentions Train record.

        Args:
            text: string
                The raw text associated with this train record.
            mentions: list(EntityMentionAnnotation)
                The entity mention annotations in this train record.
            language_code: Optional(str)
                The language code of the text in the train record.
            id: Optional(int)
                The ID associated with the train record.
        """
        error.type_check("<NLP13844580E>", str, text=text)
        error.type_check_all(
            "<NLP92872582E>", EntityMentionAnnotation, mentions=mentions
        )
        error.type_check(
            "<NLP99844580E>", str, allow_none=True, language_code=language_code
        )

        super().__init__()
        self.text = text
        self.mentions = mentions
        self.language_code = language_code
        self.id = id

    @classmethod
    def from_data_obj(
        cls,
        data_obj: dict,
        data_item_number: int = None,
    ):
        """Constructs a EntityMentionsTrainRecord instance from a train instance

        Args:
            data_obj: dict, list, tuple or instance of EntityMentionsTrainRecord
                A single train instance representation
            data_item_number: int or None
                An index for the train instance. Can be used when converting objects of a stream

        Returns:
            EntityMentionsTrainRecord: Instance of EntityMentionsTrainRecord
        """

        if isinstance(data_obj, EntityMentionsTrainRecord):
            return data_obj

        if isinstance(data_obj, dict):
            # Check if required keys do not exist
            missing_keys = (
                set(cls.fields) - {"language_code", "id"} - set(data_obj.keys())
            )
            if len(missing_keys) > 0:
                message = "Data item is missing required key(s): {} ".format(
                    missing_keys
                )
                raise DataValidationError(message, item_number=data_item_number)
            try:
                return cls.from_json(data_obj)
            except (ValueError, TypeError) as e:
                raise DataValidationError(e, item_number=data_item_number)
