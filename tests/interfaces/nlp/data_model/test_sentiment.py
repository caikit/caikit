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


class TestSentiment(TestCaseBase):
    def setUp(self):
        self.sentiment = dm.Sentiment(
            score=0.89,
            label=dm.SentimentLabel.SENT_POSITIVE.value,
            mixed=False,
            target="unit tests",
        )

        self.sentiment_minimal = dm.Sentiment(
            score=0.67, label=dm.SentimentLabel.SENT_NEUTRAL.value
        )

        self.sentiment_prob = dm.SentimentProb(positive=0.4, neutral=0.2, negative=0.4)

        self.sentiment_mention = dm.SentimentMention(
            span=dm.text_primitives.Span(begin=0, end=1),
            sentimentprob=self.sentiment_prob,
        )

        self.aggregated_sentiment = dm.AggregatedSentiment(
            score=0.67,
            label=dm.SentimentLabel.SENT_NEUTRAL.value,
            mixed=False,
            sentiment_mentions=[self.sentiment_mention],
        )

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.sentiment))
        self.assertTrue(self.validate_fields(self.sentiment_minimal))
        self.assertTrue(self.validate_fields(self.sentiment_prob))
        self.assertTrue(self.validate_fields(self.sentiment_mention))
        self.assertTrue(self.validate_fields(self.aggregated_sentiment))

    def test_from_proto_and_back_Sentiment(self):
        new = dm.Sentiment.from_proto(self.sentiment.to_proto())
        self.assertAlmostEqual(new.score, self.sentiment.score)
        self.assertEqual(new.label, self.sentiment.label)
        self.assertEqual(new.mixed, self.sentiment.mixed)
        self.assertEqual(new.target, self.sentiment.target)

        new = dm.Sentiment.from_proto(self.sentiment_minimal.to_proto())
        self.assertAlmostEqual(new.score, self.sentiment_minimal.score)
        self.assertEqual(new.label, self.sentiment_minimal.label)
        self.assertEqual(new.mixed, self.sentiment_minimal.mixed)
        self.assertEqual(new.mixed, False)
        self.assertEqual(new.target, self.sentiment_minimal.target)
        self.assertEqual(new.target, "")

    def test_from_json_and_back_Sentiment(self):
        new = dm.Sentiment.from_json(self.sentiment.to_json())
        self.assertAlmostEqual(new.score, self.sentiment.score)
        self.assertEqual(new.label, self.sentiment.label)
        self.assertEqual(new.mixed, self.sentiment.mixed)
        self.assertEqual(new.target, self.sentiment.target)

        new = dm.Sentiment.from_json(self.sentiment_minimal.to_json())
        self.assertAlmostEqual(new.score, self.sentiment_minimal.score)
        self.assertEqual(new.label, self.sentiment_minimal.label)
        self.assertEqual(new.mixed, self.sentiment_minimal.mixed)
        self.assertEqual(new.mixed, False)
        self.assertEqual(new.target, self.sentiment_minimal.target)
        self.assertEqual(new.target, "")

    def test_from_proto_and_back_SentimentProb(self):
        new = dm.SentimentProb.from_proto(self.sentiment_prob.to_proto())
        self.assertAlmostEqual(new.positive, self.sentiment_prob.positive)
        self.assertAlmostEqual(new.neutral, self.sentiment_prob.neutral)
        self.assertAlmostEqual(new.negative, self.sentiment_prob.negative)

    def test_from_proto_and_back_SentimentMention(self):
        new = dm.SentimentMention.from_proto(self.sentiment_mention.to_proto())
        self.assertAlmostEqual(
            new.sentimentprob.positive, self.sentiment_mention.sentimentprob.positive
        )
        self.assertAlmostEqual(
            new.sentimentprob.neutral, self.sentiment_mention.sentimentprob.neutral
        )
        self.assertAlmostEqual(
            new.sentimentprob.negative, self.sentiment_mention.sentimentprob.negative
        )
        self.assertEqual(new.span, self.sentiment_mention.span)

    def test_from_proto_and_back_AggregatedSentiment(self):
        new = dm.AggregatedSentiment.from_proto(self.aggregated_sentiment.to_proto())
        self.assertAlmostEqual(new.score, self.aggregated_sentiment.score)
        self.assertEqual(new.label, self.aggregated_sentiment.label)
        self.assertEqual(new.mixed, self.aggregated_sentiment.mixed)
        self.assertEqual(new.mixed, False)
        self.assertAlmostEqual(
            new.sentiment_mentions[0].sentimentprob.positive,
            self.aggregated_sentiment.sentiment_mentions[0].sentimentprob.positive,
        )
        self.assertAlmostEqual(
            new.sentiment_mentions[0].sentimentprob.positive,
            self.sentiment_mention.sentimentprob.positive,
        )

    def test_from_json_and_back_SentimentProb(self):
        new = dm.SentimentProb.from_json(self.sentiment_prob.to_json())
        self.assertAlmostEqual(new.positive, self.sentiment_prob.positive)
        self.assertAlmostEqual(new.neutral, self.sentiment_prob.neutral)
        self.assertAlmostEqual(new.negative, self.sentiment_prob.negative)

    def test_from_json_and_back_SentimentMention(self):
        new = dm.SentimentMention.from_json(self.sentiment_mention.to_json())
        self.assertAlmostEqual(
            new.sentimentprob.positive, self.sentiment_mention.sentimentprob.positive
        )
        self.assertAlmostEqual(
            new.sentimentprob.neutral, self.sentiment_mention.sentimentprob.neutral
        )
        self.assertAlmostEqual(
            new.sentimentprob.negative, self.sentiment_mention.sentimentprob.negative
        )
        self.assertEqual(new.span, self.sentiment_mention.span)

    def test_from_json_and_back_AggregatedSentiment(self):
        new = dm.AggregatedSentiment.from_json(self.aggregated_sentiment.to_json())
        self.assertAlmostEqual(new.score, self.aggregated_sentiment.score)
        self.assertEqual(new.label, self.aggregated_sentiment.label)
        self.assertEqual(new.mixed, self.aggregated_sentiment.mixed)
        self.assertEqual(new.mixed, False)
        self.assertAlmostEqual(
            new.sentiment_mentions[0].sentimentprob.positive,
            self.aggregated_sentiment.sentiment_mentions[0].sentimentprob.positive,
        )
        self.assertAlmostEqual(
            new.sentiment_mentions[0].sentimentprob.positive,
            self.sentiment_mention.sentimentprob.positive,
        )


class TestSentimentPrediction(TestCaseBase):
    def setUp(self):
        sentiment_prob = dm.SentimentProb(positive=0.4, neutral=0.2, negative=0.4)

        sentiment_mention = dm.SentimentMention(
            span=dm.text_primitives.Span(begin=0, end=1), sentimentprob=sentiment_prob
        )

        sentiment_doc = dm.AggregatedSentiment(
            score=0.67,
            label=dm.SentimentLabel.SENT_NEUTRAL.value,
            mixed=False,
            sentiment_mentions=[sentiment_mention],
        )

        sentiment_targ1 = dm.AggregatedSentiment(
            score=-0.50,
            label=dm.SentimentLabel.SENT_NEGATIVE.value,
            mixed=False,
            sentiment_mentions=[sentiment_mention],
        )

        sentiment_targ2 = dm.AggregatedSentiment(
            score=0.00,
            label=dm.SentimentLabel.SENT_NEUTRAL.value,
            mixed=False,
            sentiment_mentions=[sentiment_mention],
        )

        self.TargetsSentimentPrediction = dm.TargetsSentimentPrediction(
            targeted_sentiments={"traffic": sentiment_targ1, "cython": sentiment_targ2},
            producer_id=dm.ProducerId(name="Test", version="1.0.0"),
        )

        self.SentimentPrediction = dm.SentimentPrediction(
            document_sentiment=sentiment_doc,
            targeted_sentiments=self.TargetsSentimentPrediction,
            producer_id=dm.ProducerId(name="Test", version="1.0.0"),
        )

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.SentimentPrediction))

    def test_from_proto_and_back_TargetsSentimentPrediction(self):
        new = dm.TargetsSentimentPrediction.from_proto(
            self.TargetsSentimentPrediction.to_proto()
        )
        self.assertEqual(
            len(new.targeted_sentiments),
            len(self.TargetsSentimentPrediction.targeted_sentiments),
        )
        self.assertAlmostEqual(
            new.targeted_sentiments["traffic"].score,
            self.TargetsSentimentPrediction.targeted_sentiments["traffic"].score,
        )

        self.assertAlmostEqual(
            new.targeted_sentiments["cython"].score,
            self.TargetsSentimentPrediction.targeted_sentiments["cython"].score,
        )

        self.assertEqual(
            new.targeted_sentiments["traffic"].label,
            self.TargetsSentimentPrediction.targeted_sentiments["traffic"].label,
        )
        self.assertEqual(
            new.producer_id.name, self.TargetsSentimentPrediction.producer_id.name
        )

    def test_from_proto_and_back_SentimentPrediction(self):
        proto = self.SentimentPrediction.to_proto()
        new = dm.SentimentPrediction.from_proto(proto)
        self.assertAlmostEqual(
            new.document_sentiment.score,
            self.SentimentPrediction.document_sentiment.score,
        )
        self.assertEqual(
            len(new.targeted_sentiments.targeted_sentiments),
            len(self.SentimentPrediction.targeted_sentiments.targeted_sentiments),
        )

        self.assertAlmostEqual(
            new.targeted_sentiments.targeted_sentiments["traffic"].score,
            self.SentimentPrediction.targeted_sentiments.targeted_sentiments[
                "traffic"
            ].score,
        )

        self.assertAlmostEqual(
            new.targeted_sentiments.targeted_sentiments["cython"].score,
            self.SentimentPrediction.targeted_sentiments.targeted_sentiments[
                "cython"
            ].score,
        )

        self.assertEqual(
            new.targeted_sentiments.targeted_sentiments["traffic"].label,
            self.SentimentPrediction.targeted_sentiments.targeted_sentiments[
                "traffic"
            ].label,
        )
        self.assertEqual(
            new.producer_id.name, self.SentimentPrediction.producer_id.name
        )

    def test_from_json_and_back_SentimentPrediction(self):
        new = dm.SentimentPrediction.from_json(self.SentimentPrediction.to_json())
        self.assertAlmostEqual(
            new.document_sentiment.score,
            self.SentimentPrediction.document_sentiment.score,
        )
        self.assertEqual(
            len(new.targeted_sentiments.targeted_sentiments),
            len(self.SentimentPrediction.targeted_sentiments.targeted_sentiments),
        )

        self.assertAlmostEqual(
            new.targeted_sentiments.targeted_sentiments["traffic"].score,
            self.SentimentPrediction.targeted_sentiments.targeted_sentiments[
                "traffic"
            ].score,
        )

        self.assertAlmostEqual(
            new.targeted_sentiments.targeted_sentiments["cython"].score,
            self.SentimentPrediction.targeted_sentiments.targeted_sentiments[
                "cython"
            ].score,
        )

        self.assertEqual(
            new.targeted_sentiments.targeted_sentiments["traffic"].label,
            self.SentimentPrediction.targeted_sentiments.targeted_sentiments[
                "traffic"
            ].label,
        )
        self.assertEqual(
            new.producer_id.name, self.SentimentPrediction.producer_id.name
        )
