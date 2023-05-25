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


# Local
from caikit.interfaces.nlp import data_model as dm

# Unit Test Infrastructure
from tests.base import TestCaseBase


class TestEmotion(TestCaseBase):
    def setUp(self):
        self.emotion = dm.Emotion(
            anger=0.47, disgust=0.15, fear=0.13, joy=0.51, sadness=0.09
        )

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.emotion))

    def test_from_proto_and_back(self):
        new = dm.Emotion.from_proto(self.emotion.to_proto())
        self.assertAlmostEqual(new.anger, self.emotion.anger)
        self.assertAlmostEqual(new.disgust, self.emotion.disgust)
        self.assertAlmostEqual(new.fear, self.emotion.fear)
        self.assertAlmostEqual(new.joy, self.emotion.joy)
        self.assertAlmostEqual(new.sadness, self.emotion.sadness)

    def test_from_json_and_back(self):
        new = dm.Emotion.from_json(self.emotion.to_json())
        self.assertAlmostEqual(new.anger, self.emotion.anger)
        self.assertAlmostEqual(new.disgust, self.emotion.disgust)
        self.assertAlmostEqual(new.fear, self.emotion.fear)
        self.assertAlmostEqual(new.joy, self.emotion.joy)
        self.assertAlmostEqual(new.sadness, self.emotion.sadness)


class TestEmotionMention(TestCaseBase):
    def setUp(self):
        emotion_targ = dm.Emotion(
            anger=0.47, disgust=0.15, fear=0.13, joy=0.51, sadness=0.09
        )
        self.emotion_mention = dm.EmotionMention(
            span=dm.Span(0, 7, text="testing"), emotion=emotion_targ
        )

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.emotion_mention))

    def test_from_proto_and_back(self):
        new = dm.EmotionMention.from_proto(self.emotion_mention.to_proto())
        self.assertEqual(new.span.begin, self.emotion_mention.span.begin)
        self.assertEqual(new.span.end, self.emotion_mention.span.end)
        self.assertAlmostEqual(new.emotion.anger, self.emotion_mention.emotion.anger)
        self.assertAlmostEqual(
            new.emotion.disgust, self.emotion_mention.emotion.disgust
        )
        self.assertAlmostEqual(new.emotion.fear, self.emotion_mention.emotion.fear)
        self.assertAlmostEqual(new.emotion.joy, self.emotion_mention.emotion.joy)
        self.assertAlmostEqual(
            new.emotion.sadness, self.emotion_mention.emotion.sadness
        )

    def test_from_json_and_back(self):
        new = dm.EmotionMention.from_json(self.emotion_mention.to_json())
        self.assertEqual(new.span.begin, self.emotion_mention.span.begin)
        self.assertEqual(new.span.end, self.emotion_mention.span.end)
        self.assertAlmostEqual(new.emotion.anger, self.emotion_mention.emotion.anger)
        self.assertAlmostEqual(
            new.emotion.disgust, self.emotion_mention.emotion.disgust
        )
        self.assertAlmostEqual(new.emotion.fear, self.emotion_mention.emotion.fear)
        self.assertAlmostEqual(new.emotion.joy, self.emotion_mention.emotion.joy)
        self.assertAlmostEqual(
            new.emotion.sadness, self.emotion_mention.emotion.sadness
        )


class TestAggregatedEmotionPrediction(TestCaseBase):
    def setUp(self):
        avg_emotion = dm.Emotion(
            anger=0.47, disgust=0.15, fear=0.13, joy=0.51, sadness=0.09
        )
        emotion_targ = dm.Emotion(
            anger=0.50, disgust=0.20, fear=0.13, joy=0.51, sadness=0.09
        )
        emotion_targ2 = dm.Emotion(
            anger=0.90, disgust=0.20, fear=0.13, joy=0.51, sadness=0.09
        )
        emotion_mention1 = dm.EmotionMention(
            span=dm.Span(0, 7, text="testing"), emotion=emotion_targ
        )
        emotion_mention2 = dm.EmotionMention(
            span=dm.Span(0, 7, text="testing"), emotion=emotion_targ2
        )
        self.emotion_prediction = dm.AggregatedEmotionPrediction(
            emotion=avg_emotion,
            target="testing",
            emotion_mentions=[emotion_mention1, emotion_mention2],
        )

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.emotion_prediction))

    def test_from_proto_and_back(self):
        new = dm.AggregatedEmotionPrediction.from_proto(
            self.emotion_prediction.to_proto()
        )
        self.assertAlmostEqual(new.emotion.anger, self.emotion_prediction.emotion.anger)
        self.assertAlmostEqual(
            new.emotion.disgust, self.emotion_prediction.emotion.disgust
        )
        self.assertEqual(new.target, self.emotion_prediction.target)
        self.assertEqual(
            len(new.emotion_mentions), len(self.emotion_prediction.emotion_mentions)
        )

    def test_from_json_and_back(self):
        new = dm.AggregatedEmotionPrediction.from_json(
            self.emotion_prediction.to_json()
        )
        self.assertAlmostEqual(new.emotion.anger, self.emotion_prediction.emotion.anger)
        self.assertAlmostEqual(
            new.emotion.disgust, self.emotion_prediction.emotion.disgust
        )
        self.assertEqual(new.target, self.emotion_prediction.target)
        self.assertEqual(
            len(new.emotion_mentions), len(self.emotion_prediction.emotion_mentions)
        )


class TestEmotionPrediction(TestCaseBase):
    def setUp(self):
        avg_emotion = dm.Emotion(
            anger=0.47, disgust=0.15, fear=0.13, joy=0.51, sadness=0.09
        )
        emotion_targ = dm.Emotion(
            anger=0.50, disgust=0.20, fear=0.13, joy=0.51, sadness=0.09
        )
        emotion_targ2 = dm.Emotion(
            anger=0.90, disgust=0.20, fear=0.13, joy=0.51, sadness=0.09
        )
        emotion_mention1 = dm.EmotionMention(
            span=dm.Span(0, 7, text="testing"), emotion=emotion_targ
        )
        emotion_mention2 = dm.EmotionMention(
            span=dm.Span(0, 7, text="testing"), emotion=emotion_targ2
        )
        emotion_prediction1 = dm.AggregatedEmotionPrediction(
            emotion=avg_emotion,
            target="testing",
            emotion_mentions=[emotion_mention1, emotion_mention2],
        )
        avg_emotion2 = dm.Emotion(
            anger=0.47, disgust=0.15, fear=0.13, joy=0.51, sadness=0.09
        )
        emotion_targ3 = dm.Emotion(
            anger=0.50, disgust=0.20, fear=0.13, joy=0.51, sadness=0.09
        )
        emotion_targ4 = dm.Emotion(
            anger=0.90, disgust=0.20, fear=0.13, joy=0.51, sadness=0.09
        )
        emotion_mention3 = dm.EmotionMention(
            span=dm.Span(0, 7, text="resting"), emotion=emotion_targ3
        )
        emotion_mention4 = dm.EmotionMention(
            span=dm.Span(0, 7, text="resting"), emotion=emotion_targ4
        )
        emotion_prediction2 = dm.AggregatedEmotionPrediction(
            emotion=avg_emotion2,
            target="resting",
            emotion_mentions=[emotion_mention3, emotion_mention4],
        )
        self.agg_emotion = dm.EmotionPrediction(
            emotion_predictions=[emotion_prediction1, emotion_prediction2],
            producer_id=dm.ProducerId(name="Test", version="1.0.0"),
        )

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.agg_emotion))

    def test_from_proto_and_back(self):
        new = dm.EmotionPrediction.from_proto(self.agg_emotion.to_proto())
        self.assertAlmostEqual(
            new.emotion_predictions[0].emotion.anger,
            self.agg_emotion.emotion_predictions[0].emotion.anger,
        )
        self.assertAlmostEqual(
            new.emotion_predictions[0].emotion.disgust,
            self.agg_emotion.emotion_predictions[0].emotion.disgust,
        )
        self.assertEqual(
            new.emotion_predictions[0].target,
            self.agg_emotion.emotion_predictions[0].target,
        )
        self.assertEqual(
            len(new.emotion_predictions), len(self.agg_emotion.emotion_predictions)
        )
        self.assertEqual(new.producer_id.name, self.agg_emotion.producer_id.name)

    def test_from_json_and_back(self):
        new = dm.EmotionPrediction.from_json(self.agg_emotion.to_json())
        self.assertAlmostEqual(
            new.emotion_predictions[0].emotion.anger,
            self.agg_emotion.emotion_predictions[0].emotion.anger,
        )
        self.assertAlmostEqual(
            new.emotion_predictions[0].emotion.disgust,
            self.agg_emotion.emotion_predictions[0].emotion.disgust,
        )
        self.assertEqual(
            new.emotion_predictions[0].target,
            self.agg_emotion.emotion_predictions[0].target,
        )
        self.assertEqual(
            len(new.emotion_predictions), len(self.agg_emotion.emotion_predictions)
        )
        self.assertEqual(new.producer_id.name, self.agg_emotion.producer_id.name)
