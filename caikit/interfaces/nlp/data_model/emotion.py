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
"""Data structures for emotion analysis.
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
class Emotion(DataObjectBase):
    """The full emotion expressed in a given context"""

    anger: Annotated[float, FieldNumber(1)]
    disgust: Annotated[float, FieldNumber(2)]
    fear: Annotated[float, FieldNumber(3)]
    joy: Annotated[float, FieldNumber(4)]
    sadness: Annotated[float, FieldNumber(5)]

    """Emotion associated with a single span or document."""

    def __init__(self, anger, disgust, fear, joy, sadness, target=""):
        """Construct a new emotion.

        Args:
            anger:  float
                The anger score in the range of [0, 1].
                Confidence score of predicting the doc as anger emotion.
            disgust:  float
                The disgust score in the range of [0, 1].
                Confidence score of predicting the doc as disgust emotion.
            fear:  float
                The fear score in the range of [0, 1].
                Confidence score of predicting the doc as fear emotion.
            joy:  float
                The joy score in the range of [0, 1].
                Confidence score of predicting the doc as joy emotion.
            sadness:  float
                The sadness score in the range of [0, 1].
                Confidence score of predicting the doc as sadness emotion.
        """
        error.type_check(
            "<NLP86548946E>",
            float,
            anger=anger,
            disgust=disgust,
            fear=fear,
            joy=joy,
            sadness=sadness,
        )
        error.value_check(
            "<NLP19139291E>",
            0.0 <= anger <= 1.0,
            "`anger` of `{}` is not between 0 and 1",
            anger,
        )
        error.value_check(
            "<NLP73616005E>",
            0.0 <= disgust <= 1.0,
            "`disgust` of `{}` is not between 0 and 1",
            disgust,
        )
        error.value_check(
            "<NLP23527971E>",
            0.0 <= fear <= 1.0,
            "`fear` of `{}` is not between 0 and 1",
            fear,
        )
        error.value_check(
            "<NLP74209869E>",
            0.0 <= joy <= 1.0,
            "`joy` of `{}` is not between 0 and 1",
            joy,
        )
        error.value_check(
            "<NLP13751784E>",
            0.0 <= sadness <= 1.0,
            "`sadness` of `{}` is not between 0 and 1",
            sadness,
        )
        error.type_check("<NLP39704020E>", str, target=target)

        super().__init__()
        self.anger = anger
        self.disgust = disgust
        self.fear = fear
        self.joy = joy
        self.sadness = sadness


@dataobject(package="caikit_data_model.nlp")
class EmotionMention(DataObjectBase):
    """An individual, atomic emotion mention over a given region of the
    input (could be a sentence, a paragraph, a section of text within
    a sentence, etc.)"""

    span: Annotated[text_primitives.Span, FieldNumber(1)]
    emotion: Annotated[Emotion, FieldNumber(2)]

    """Emotion associated with a single span or document"""

    def __init__(self, span, emotion):
        """Construct a new emotion.
        Args:
            span:   Span
                Location of the region of interest within the input document
            emotion: Emotion
                The emotion predictions associated with the span.
        """
        error.type_check("<NLP05809335E>", Emotion, emotion=emotion)
        error.type_check("<NLP35130066E>", text_primitives.Span, span=span)
        super().__init__()
        self.span = span
        self.emotion = emotion


@dataobject(package="caikit_data_model.nlp")
class AggregatedEmotionPrediction(DataObjectBase):
    """An emotion prediction for either a document or a target"""

    emotion: Annotated[Emotion, FieldNumber(1)]
    target: Annotated[str, FieldNumber(2)]
    emotion_mentions: Annotated[List[EmotionMention], FieldNumber(3)]

    """An emotion prediction for either a document or a target"""

    def __init__(self, emotion=None, target="", emotion_mentions=None):
        """Construct a new emotion prediction.
        Args:
            emotion: Emotion
                The overall aggregated emotions and respective aggregate confidence scores
            target:  str
                The string representation of the target for this emotion.
                If set to the empty string (default) then it is assumed that
                this emotion is for an entire document.
            emotion_mentions: (list(EmotionMention))
                Individual atomic emotions that went into producing the aggregated emotion
        """
        error.type_check("<NLP01428099E>", Emotion, emotion=emotion, allow_none=True)
        error.type_check("<NLP88111466E>", str, target=target, allow_none=True)
        error.type_check_all(
            "<NLP91537607E>",
            EmotionMention,
            allow_none=True,
            emotion_mentions=emotion_mentions,
        )
        super().__init__()
        self.emotion = emotion
        self.target = target
        self.emotion_mentions = emotion_mentions


@dataobject(package="caikit_data_model.nlp")
class EmotionPrediction(DataObjectBase):
    """An emotion prediction for a document and zero or more targets."""

    emotion_predictions: Annotated[List[AggregatedEmotionPrediction], FieldNumber(1)]
    producer_id: Annotated[ProducerId, FieldNumber(2)]

    """An emotion prediction for a document and zero or more targets."""

    def __init__(self, emotion_predictions, producer_id):
        """Construct a new targeted emotion prediction.
        Args:
            emotion_predictions: (list(AggregatedEmotionPrediction))
                List of emotions per target
            producer_id:  ProducerId or None
                The block that produced this emotion prediction.
        """
        error.type_check_all(
            "<NLP86630312E>",
            AggregatedEmotionPrediction,
            emotion_predictions=emotion_predictions,
        )
        error.type_check("<NLP14272394E>", ProducerId, producer_id=producer_id)
        super().__init__()
        self.emotion_predictions = emotion_predictions
        self.producer_id = producer_id
