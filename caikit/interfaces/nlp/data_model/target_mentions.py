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
"""Data structures for target mention extraction.
"""

# Standard
from typing import List
import collections

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from ....core.data_model import DataObjectBase, dataobject
from ....core.toolkit.errors import error_handler
from ...common.data_model import ProducerId
from . import entities, text_primitives

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(package="caikit_data_model.nlp")
class TargetPhrases(DataObjectBase):
    """Set of targets to search for mention matches of"""

    targets: Annotated[List[str], FieldNumber(1)]

    """Collection of target phrases"""

    def __init__(self, targets):
        """Construct a TargetPhrases from a list of strings

        Args:
            targets:  list(str)
                List of target phrase strings
        """
        error.type_check_all("<NLP57911994E>", str, targets=targets)

        super().__init__()
        self.targets = targets


@dataobject(package="caikit_data_model.nlp")
class TargetMentions(DataObjectBase):
    """Single set of matched target mention spans"""

    spans: Annotated[List[text_primitives.Span], FieldNumber(1)]

    """List of mention spans for the mentions of a given target"""

    def __init__(self, spans):
        """Construct a set of mentions for a single target

        Args:
            spans:  list(tuple[2]) or list(dm.Span)
                The spans for the target mentions
        """
        error.type_check_all("<NLP58861164E>", text_primitives.Span, tuple, spans=spans)

        super().__init__()
        self.spans = [
            span
            if isinstance(span, text_primitives.Span)
            else text_primitives.Span(*span)
            for span in spans
        ]

    @classmethod
    def from_entity_mentions(cls, entity_mentions):
        """Convert a list of entity mentions into corresponding target mentions.

        Args:
            entity_mentions:  list(dm.EntityMention)
                A list of entity mentions to convert to target mentions

        Returns:
            dm.TargetMentions
                The target mentions (spans) derived from the provided entity mentions
        """
        error.type_check_all(
            "<NLP87443612E>", entities.EntityMention, entity_mentions=entity_mentions
        )

        return TargetMentions([mention.span for mention in entity_mentions])


@dataobject(package="caikit_data_model.nlp")
class TargetMentionsPrediction(DataObjectBase):
    """A set of matched spans for a set of spans"""

    targets: Annotated[List[TargetMentions], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]

    """Predicted mention offsets for a set of targets"""

    def __init__(self, targets, producer_id=None):
        """Construct a set of targets

        Args:
            targets: list(list(tuple[2])) or list(TargetMentions)
                The set of mention spans for each target analyzed
            producer_id:  ProducerId or None
                The block that produced this target mentions prediction.
        """
        error.type_check("<NLP19403954E>", collections.abc.Iterable, targets=targets)
        error.type_check(
            "<NLP41021853E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        super().__init__()
        self.targets = [
            mentions
            if isinstance(mentions, TargetMentions)
            else TargetMentions(mentions)
            for mentions in targets
        ]
        self.producer_id = producer_id

    def __add__(self, other):
        """Concatenate another TargetMentionsPrediction with this one.

        Args:
            other: TargetMentionsPrediction
                The other TargetMentionsPrediction whose targets are to be merged

        Returns:
            TargetMentionsPrediction containing the merged targets
        """
        error.type_check("<NLP20945328E>", TargetMentionsPrediction, other=other)

        return TargetMentionsPrediction(
            self.targets + other.targets,
            ProducerId("TargetMentionsPrediction.__add__", "0.0.0"),
        )

    @classmethod
    def from_entities_prediction(cls, entities_prediction):
        """Convert an EntitiesPrediction into a corresponding TargetMentionsPrediction.

        Args:
            entity_mentions_prediction:  dm.EntitiesPrediction
                The EntitiesPrediction to convert

        Returns:
            dm.TargetMentionsPrediction
                Derived target mentions prediction
        """
        error.type_check(
            "<NLP53777920E>",
            entities.EntitiesPrediction,
            entities_prediction=entities_prediction,
        )

        return TargetMentionsPrediction(
            [
                TargetMentions.from_entity_mentions(entity.mentions)
                for entity in entities_prediction.entities
            ],
            producer_id=entities_prediction.producer_id,
        )
