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
"""Data structures for text categorization.
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

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(package="caikit_data_model.nlp")
class RelevantText(DataObjectBase):
    text: Annotated[str, FieldNumber(1)]

    """A single explanation of a categories prediction."""

    def __init__(self, text):
        """Construct the explanatory n-gram extracted from the text.

        Args:
            text: the explanatory n-gram extracted from the text
        """
        error.type_check("<NLP21907715E>", str, text=text)

        super().__init__()
        self.text = text


@dataobject(package="caikit_data_model.nlp")
class Category(DataObjectBase):
    """A single hierarchical category label and confidence."""

    labels: Annotated[List[str], FieldNumber(1)]
    score: Annotated[float, FieldNumber(2)]
    explanation: Annotated[List[RelevantText], FieldNumber(3)]

    """A single category prediction."""

    def __init__(self, labels, score, explanation=None):
        """Construct a new category prediction.

        Args:
            labels:  list(str)
                The labels in our categories hierarchy listed sequentially.
                For example, ["activities", "sports", "hocky"]
            score:  float
                Confidence-like score in [0, 1] for this category.
        """
        error.type_check_all("<NLP18592253E>", str, labels=labels)
        error.value_check(
            "<NLP75023820E>",
            0.0 <= score <= 1.0,
            "`score` of `{}` not between 0 and 1",
            score,
        )

        super().__init__()
        self.labels = labels
        self.score = score
        self.explanation = [] if explanation is None else explanation

    def __lt__(self, other):
        """Compare categories by score."""
        return self.score < other.score

    def get_label_hierarchy_as_str(self):
        """Convert the list of labels to a string.

        Returns:
            str
                String, where hierarchy levels are separated by backslashes.
        Example:
            [foo, bar, baz] -> /foo/bar/baz
        """
        return "/" + "/".join(self.labels)

    @staticmethod
    def extract_subhierarchy_from_str(str_labels, level):
        """Given a stringified joined list of labels, retrieve the subhierarchy up to a given
        level.

        Args:
            str_labels: str
                String, where hierarchy levels are separated by backslashes.
            level: int
                Level of subhierarchy to extract. If None is provided, returns the whole label.
        Returns:
            str:
                String representation of the subhierarchy of the given level.
        Example:
            Given labels = '/foo/bar/baz', level = 2 -> '/foo/bar'
        """
        error.type_check("<NLP18213113E>", str, str_labels=str_labels)
        error.type_check("<NLP18332313E>", int, allow_none=True, level=level)
        error.value_check(
            "<NLP70584996E>",
            level is None or level > 0,
            "Level must be an integer x > 0 or None",
        )
        if not level:
            return str_labels
        return "/".join(str_labels.split("/")[: level + 1])


@dataobject(package="caikit_data_model.nlp")
class CategoriesPrediction(DataObjectBase):
    """A categories prediction generated from a document and consisting multiple category labels."""

    categories: Annotated[List[Category], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]

    """The result of a categories prediction."""

    def __init__(self, categories, producer_id=None):
        """Construct a new categories prediction.

        Args:
            categories:  list(Category)
                The categories (predictions) associated with this
                categories prediction.
            producer_id:  ProducerId or None
                The block that produced this categories prediction.
        """
        error.type_check_all("<NLP42343606E>", Category, categories=categories)
        error.type_check(
            "<NLP13148586E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        super().__init__()
        self.categories = sorted(categories, reverse=True)
        self.producer_id = producer_id
