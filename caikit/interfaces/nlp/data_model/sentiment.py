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
"""Data structures for sentiment analysis.
"""
# Standard
from enum import Enum
from typing import Dict, List

# First Party
from py_to_proto.dataclass_to_proto import Annotated, FieldNumber
import alog

# Local
from ....core.data_model import DataObjectBase, dataobject, enums, protobufs
from ....core.toolkit.errors import error_handler
from ...common.data_model import ProducerId
from . import text_primitives

log = alog.use_channel("DATAM")
error = error_handler.get(log)


@dataobject(package="caikit_data_model.nlp")
class SentimentLabel(Enum):
    """
    The full set of labels that can be applied as a top-level sentiment classification.
    """

    SENT_UNSET = 0
    SENT_NEGATIVE = 1
    SENT_NEUTRAL = 2
    SENT_POSITIVE = 3


@dataobject(package="caikit_data_model.nlp")
class Sentiment(DataObjectBase):
    """The full sentiment result expressed in a given context (document or target)"""

    score: Annotated[float, FieldNumber(1)]
    label: Annotated[SentimentLabel, FieldNumber(2)]
    mixed: Annotated[bool, FieldNumber(3)]
    target: Annotated[str, FieldNumber(4)]

    """Sentiment associated with a single span or document."""

    def __init__(self, score, label, mixed=False, target=""):
        """Construct a new sentiment.

        Args:
            score:  float
                The sentiment score in the range of [-1, 1].  A negative value indicates negative
                sentiment and a positive value indicates positive sentiment.
            label:  int (SentimentLabel)
                The enum label for this sentiment, e.g., `SENT_UNSET`, `SENT_POSITIVE`,
                `SENT_NEGATIVE` or `SENT_NEUTRAL`.
            mixed:  bool
                True if the sentiment is *both* positive and negative.
            target:  str
                The string representation of the target for this sentiment.  If set to the empty
                string (default) then it is assumed that this sentiment is for an entire document.
        """
        error.value_check(
            "<NLP46385322E>",
            -1.0 <= score <= 1.0,
            "`score` of `{}` is not between -1 and 1".format(score),
        )
        error.type_check("<NLP84706804E>", int, label=label)
        error.type_check("<NLP76211808E>", bool, mixed=mixed)
        error.type_check("<NLP26615773E>", str, target=target)

        super().__init__()
        self.score = score
        self.label = label
        self.mixed = mixed
        self.target = target


@dataobject(package="caikit_data_model.nlp")
class AtomicSentiment(DataObjectBase):
    """An individual, atomic sentiment mention over a given region of the input (could be a sentence, a paragraph, a section of text within a sentence, etc.)"""

    span: Annotated[text_primitives.Span, FieldNumber(1)]
    score: Annotated[float, FieldNumber(2)]
    label: Annotated[SentimentLabel, FieldNumber(3)]

    """Sentiment associated with a single span or document."""

    def __init__(self, span, score, label):
        """Construct a new sentiment.
        Args:
            span:   Span
                Location of the region of interest within the input document
            score:  float
                The sentiment score in the range of [-1, 1].  A negative value
                indicates negative sentiment and a positive value indicates
                positive sentiment.
            label:  int (SentimentLabel)
                The enum label for this sentiment, e.g., SENT_UNSET, SENT_POSITIVE,
                SENT_NEGATIVE, SENT_NEUTRAL.
        """
        error.value_check(
            "<NLP36942974E>",
            -1.0 <= score <= 1.0,
            "`score` of `{}` is not between -1 and 1".format(score),
        )
        error.type_check("<NLP45831210E>", text_primitives.Span, span=span)
        error.type_check("<NLP31052295E>", int, label=label)

        super().__init__()
        self.span = span
        self.score = score
        self.label = label


@dataobject(package="caikit_data_model.nlp")
class AggregateSentimentPrediction(DataObjectBase):
    """A aggregate sentiment prediction for a number of atomic sentiments within the document."""

    score: Annotated[float, FieldNumber(1)]
    label: Annotated[SentimentLabel, FieldNumber(2)]
    mixed: Annotated[bool, FieldNumber(3)]
    target: Annotated[str, FieldNumber(4)]
    sentiment_mentions: Annotated[List[AtomicSentiment], FieldNumber(5)]
    producer_id: Annotated[ProducerId, FieldNumber(6)]

    """A sentiment prediction for either a document or a target."""

    label_dict = {
        SentimentLabel(SentimentLabel.SENT_UNSET.value): "Unset",
        SentimentLabel(SentimentLabel.SENT_NEGATIVE.value): "Negative",
        SentimentLabel(SentimentLabel.SENT_NEUTRAL.value): "Neutral",
        SentimentLabel(SentimentLabel.SENT_POSITIVE.value): "Positive",
    }

    def __init__(
        self,
        label,
        score=None,
        mixed=None,
        target="",
        sentiment_mentions=None,
        producer_id=None,
    ):
        """Construct a new sentiment prediction.
        Args:
            score:  float
                Overall (aggregated) score
                The sentiment score in the range of [-1, 1].  A negative value
                indicates negative sentiment and a positive value indicates
                positive sentiment.
            label:  int (SentimentLabel)
                Overall (aggregated) label
                The enum label for this sentiment, e.g., SENT_UNSET, SENT_POSITIVE,
                SENT_NEGATIVE, SENT_NEUTRAL.
            mixed: bool
                Were there mentions containing a mixture of polarities?
            target:  str
                The string representation of the target for this sentiment.
                If set to the empty string (default) then it is assumed that
                this sentiment is for an entire document.
            sentiment_mentions: (list(AtomicSentiment))
                Individual atomic sentiments that went into producing the aggregated sentiment
            producer_id:  ProducerId or None
                The block that produced this syntax analysis.
        """
        error.value_check(
            "<NLP55269291E>",
            (score is None) or (-1.0 <= score <= 1.0),
            "`score` of `{}` is not between -1 and 1".format(score),
        )
        error.type_check("<NLP32699803E>", int, label=label)
        error.type_check("<NLP15839765E>", bool, mixed=mixed, allow_none=True)
        error.type_check("<NLP41888984E>", str, target=target)
        error.type_check_all(
            "<NLP01273330E>",
            AtomicSentiment,
            allow_none=True,
            sentiment_mentions=sentiment_mentions,
        )
        error.type_check(
            "<NLP82980057E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        super().__init__()
        self.score = score
        self.label = label
        self.mixed = mixed
        self.target = target
        self.sentiment_mentions = sentiment_mentions
        self.producer_id = producer_id

    def prettify_document_sentiment(self):
        """Get the document sentiment object with score, label, mixed and target.

        Returns:
            dict
                {'label': str, 'score': float, 'mixed': boolean, 'target': str}
        """
        return {
            "label": self.label_dict[self.label],
            "mixed": self.mixed,
            "score": self.score,
        }

    def prettify_sentence_based_sentiment(self):
        """Get the document sentiment object with score, label, mixed and target.

        Returns:
            dict
                {'label': str, 'score': float, 'mixed': boolean, 'target': str}
        """
        return [
            {
                "label": self.label_dict[target.label],
                "score": target.score,
                "target": target.span.text,
            }
            for target in self.sentiment_mentions
        ]


@dataobject(package="caikit_data_model.nlp")
class SentimentProb(DataObjectBase):
    """An individual, "atomic" sentiment mention over a given region of the input (could be a sentence, a paragraph, a section of text within a sentence, etc.)"""

    positive: Annotated[float, FieldNumber(1)]
    neutral: Annotated[float, FieldNumber(2)]
    negative: Annotated[float, FieldNumber(3)]

    """An individual, "atomic" sentiment mention over a given region of the input"""

    def __init__(self, positive, neutral, negative):
        """Construct a new sentiment probability.

        Args:
            positive:  float
                The positive score in the range of [0, 1].
            neutral:  float
                The neutral score in the range of [0, 1].
            negative:  float
                The negative probability in the range of [0, 1].
        """
        error.type_check(
            "<NLP88885619E>",
            float,
            positive=positive,
            neutral=neutral,
            negative=negative,
        )
        error.value_check(
            "<NLP01747123E>",
            0.0 <= positive <= 1.0,
            "`positive` of `{}` is not between 0 and 1",
            positive,
        )
        error.value_check(
            "<NLP23223969E>",
            0.0 <= neutral <= 1.0,
            "`neutral` of `{}` is not between 0 and 1",
            neutral,
        )
        error.value_check(
            "<NLP58843469E>",
            0.0 <= negative <= 1.0,
            "`negative` of `{}` is not between 0 and 1",
            negative,
        )
        super().__init__()
        self.positive = positive
        self.neutral = neutral
        self.negative = negative


@dataobject(package="caikit_data_model.nlp")
class SentimentMention(DataObjectBase):
    span: Annotated[text_primitives.Span, FieldNumber(1)]
    sentimentprob: Annotated[SentimentProb, FieldNumber(2)]

    """An individual, "atomic" sentiment mention over a given region of the input"""

    def __init__(self, span, sentimentprob):
        """Construct a new sentiment.

        Args:
            span:   Span
                Location of the region of interest within the input document
            sentimentprob:  SentimentProb
                Each sentiment probability in the range of [0, 1].

        """
        error.type_check("<NLP96227271E>", text_primitives.Span, span=span)

        error.type_check("<NLP13740108E>", SentimentProb, sentimentprob=sentimentprob)

        super().__init__()
        self.span = span
        self.sentimentprob = sentimentprob


@dataobject(package="caikit_data_model.nlp")
class AggregatedSentiment(DataObjectBase):
    """A sentiment prediction for a number of atomic sentiments within the document."""

    score: Annotated[float, FieldNumber(1)]
    label: Annotated[SentimentLabel, FieldNumber(2)]
    mixed: Annotated[bool, FieldNumber(3)]
    sentiment_mentions: Annotated[List[SentimentMention], FieldNumber(4)]

    """A sentiment prediction for a number of atomic sentiments within the document."""

    def __init__(self, score, label=0, mixed=False, sentiment_mentions=None):
        """Construct a new sentiment prediction.

        Args:
            score:  float
                The score of the sentiment in the [-1, 1] range
            label:  int, default=0
                The trinary (positive, negative, neutral) classification label
            mixed:  bool, default=False
                (optional)True if the sentiment is *both* positive and negative.
            sentiment_mentions: (list(SentimentMetion)), default=None
                (optional)Individual atomic sentiment that went into producing the aggregated sentiment
        """
        if isinstance(score, self._proto_class):
            log.debug2("Handling input protobufs.AggregatedSentiment")
            label = score.label
            mixed = score.mixed
            sentiment_mentions_component = list(score.sentiment_mentions)
            score = score.score
            sentiment_mentions = []
            # If the field value is a nested message (i.e. a list wrapper), we
            # need to convert that too.
            for sm in sentiment_mentions_component:
                submsg_class_name, submsg_class = self._get_class_for_proto(sm)
                sentiment_mentions.append(submsg_class.from_proto(sm))

        error.value_check(
            "<NLP87579767E>",
            -1.0 <= score <= 1.0,
            "`score` of `{}` is not between -1 and 1".format(score),
        )
        error.type_check("<NLP38599343E>", int, label=label)
        error.type_check("<NLP83382639E>", bool, mixed=mixed)

        error.type_check_all(
            "<NLP43733502E>",
            SentimentMention,
            allow_none=True,
            sentiment_mentions=sentiment_mentions,
        )

        super().__init__()
        self.score = score
        self.label = label
        self.mixed = mixed
        self.sentiment_mentions = (
            sentiment_mentions if sentiment_mentions is not None else []
        )


@dataobject(package="caikit_data_model.nlp")
class TargetsSentimentPrediction(DataObjectBase):
    targeted_sentiments: Annotated[Dict[str, AggregatedSentiment], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]

    """A sentiment prediction for targets."""

    def __init__(self, targeted_sentiments, producer_id):
        """Construct a new sentiment prediction.
        Args:
            targeted_sentiments: dict{target: AggregatedSentiment}
                Mapping from target string to computed sentiment for the given target
            producer_id:  ProducerId or None
                The block that produced this emotion prediction.
        """
        error.type_check(
            "<NLP85733835E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        error.type_check(
            "<NLP56695324E>", dict, targeted_sentiments=targeted_sentiments
        )

        if targeted_sentiments:
            for s in targeted_sentiments.values():
                error.type_check("<NLP66394863E>", AggregatedSentiment, s=s)

        self.targeted_sentiments = targeted_sentiments
        self.producer_id = producer_id


@dataobject(package="caikit_data_model.nlp")
class SentimentPrediction(DataObjectBase):
    """The return type for all `sentiment` blocks

    A sentiment towards a document and optionally specific sentiment targets from within that document."""

    document_sentiment: Annotated[AggregatedSentiment, FieldNumber(1)]
    targeted_sentiments: Annotated[TargetsSentimentPrediction, FieldNumber(2)]
    producer_id: Annotated[ProducerId, FieldNumber(3)]

    """A sentiment prediction for either a document or a target."""

    label_dict = {
        SentimentLabel(SentimentLabel.SENT_UNSET.value): "Unset",
        SentimentLabel(SentimentLabel.SENT_NEGATIVE.value): "Negative",
        SentimentLabel(SentimentLabel.SENT_NEUTRAL.value): "Neutral",
        SentimentLabel(SentimentLabel.SENT_POSITIVE.value): "Positive",
    }

    def __init__(self, document_sentiment, targeted_sentiments, producer_id):
        """Construct a new sentiment prediction.
        Args:
            document_sentiment: (sentiment)
                Individual sentiment predictions that went into producing the aggregated document sentiment
            targeted_sentiments: TargetsSentimentPrediction
                Mapping from target string to computed sentiment for the given target
            producer_id:  ProducerId or None
                The block that produced this emotion prediction.
        """
        error.type_check(
            "<NLP04096466E>", AggregatedSentiment, document_sentiment=document_sentiment
        )

        error.type_check(
            "<NLP17303089E>",
            ProducerId,
            allow_none=True,
            producer_id=producer_id,
        )

        error.type_check(
            "<NLP03109820E>",
            TargetsSentimentPrediction,
            targeted_sentiments=targeted_sentiments,
        )

        super().__init__()
        self.document_sentiment = document_sentiment
        self.targeted_sentiments = targeted_sentiments
        self.producer_id = producer_id
