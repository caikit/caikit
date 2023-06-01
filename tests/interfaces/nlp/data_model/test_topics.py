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
from caikit.interfaces.common.data_model import ProducerId
from caikit.interfaces.nlp import data_model as dm

# Unit Test Infrastructure
from tests.base import TestCaseBase


def compare_topics(test_instance, original_topic, reconstructed_topic):
    """Helper to check equality of a newly built against the original topic we were using.
    NOTE: In the future, it probably makes sense to move this functionality to allow for
    equality checks into the data model class itself. Explode if we find misalignment.

    Args:
        test_instance: TestCaseBase
            Instance on which we are running our tests.
        original_topic: watson_nlp.data_model.Topic
            Topic to compare against original_topic
        reconstructed_topic: watson_nlp.data_model.Topic
            Topic to compare against reconstructed_topic
    """
    test_instance.assertEqual(original_topic.score, reconstructed_topic.score)
    test_instance.assertEqual(original_topic.size, reconstructed_topic.size)
    # Compare iterable lengths
    test_instance.assertEqual(
        len(original_topic.ngrams), len(reconstructed_topic.ngrams)
    )
    test_instance.assertEqual(
        len(original_topic.snippets), len(reconstructed_topic.snippets)
    )
    test_instance.assertEqual(
        len(original_topic.sentences), len(reconstructed_topic.sentences)
    )
    # Compare values within each of our iterable objects
    # 1. n-grams
    for orig_ngram, recon_ngram in zip(
        original_topic.ngrams, reconstructed_topic.ngrams
    ):
        test_instance.assertEqual(orig_ngram.texts, recon_ngram.texts)
        test_instance.assertAlmostEqual(orig_ngram.relevance, recon_ngram.relevance)
    # 2. snippets
    for orig_snippet, recon_snippet in zip(
        original_topic.snippets, reconstructed_topic.snippets
    ):
        test_instance.assertEqual(orig_snippet.text, recon_snippet.text)
        test_instance.assertAlmostEqual(orig_snippet.distance, recon_snippet.distance)
    # 3. sentences
    for orig_sent, recon_sent in zip(
        original_topic.sentences, reconstructed_topic.sentences
    ):
        test_instance.assertEqual(orig_sent.text, recon_sent.text)
        test_instance.assertAlmostEqual(orig_sent.distance, recon_sent.distance)


class TestTopic(TestCaseBase):
    def setUp(cls):
        cls.topic = dm.Topic(
            name="Foobar",
            score=0.5,
            size=1,
            ngrams=[dm.NGram(texts=["foo", "bar", "baz"], relevance=0.0)],
            snippets=[dm.TopicPhrase(text="foo", distance=0.7)],
            sentences=[dm.TopicPhrase(text="foo", distance=0.7)],
            producer_id=ProducerId("TopicTest", "1.2.5"),
        )

    def test_fields(self):
        """Ensure that we are able to validate the fields of our Topic DM object."""
        self.assertTrue(self.validate_fields(self.topic))

    def test_from_proto_and_back(self):
        """Ensure that we can convert to proto and back and have the same data."""
        reconstructed_topic = dm.Topic.from_proto(self.topic.to_proto())
        compare_topics(self, self.topic, reconstructed_topic)

    def test_from_json_and_back(self):
        """Ensure that we can convert to json and back and have the same data."""
        reconstructed_topic = dm.Topic.from_json(self.topic.to_json())
        compare_topics(self, self.topic, reconstructed_topic)

    def test_init_with_bad_type(self):
        # Things we would use to build a happy Topic
        happy_args = {
            "name": "Foobar",
            "score": 0.5,
            "size": 1,
            "ngrams": [dm.NGram(texts=["foo", "bar", "baz"], relevance=0.0)],
            "snippets": [dm.TopicPhrase(text="foo", distance=0.7)],
            "sentences": [dm.TopicPhrase(text="foo", distance=0.7)],
            "producer_id": ProducerId("TopicTest", "1.2.5"),
        }
        # Individual overrides for each arg
        bad_arg_overrides = [
            {"name": 13},
            {"score": "sad type"},
            {"size": "sad type"},
            {"ngrams": "sad type"},
            {"sentences": "sad type"},
            {"snippets": "sad type"},
            # Checks for sad inner types on our iterable arguments
            {"ngrams": ["sad inner type"]},
            {"sentences": ["sad inner type"]},
            {"snippets": ["sad inner type"]},
        ]
        for override in bad_arg_overrides:
            # Ensure that any time we override a happy arg with a sad type, we are sad
            with self.assertRaises(TypeError):
                sad_args = {**happy_args, **override}
                dm.Topic(**sad_args)


class TestTopicPhrase(TestCaseBase):
    def setUp(cls):
        cls.topic_phrase = dm.TopicPhrase(text="foo", distance=0.7)

    def test_fields(self):
        """Ensure that we are able to validate the fields of our TopicPhrase DM object."""
        self.assertTrue(self.validate_fields(self.topic_phrase))

    def test_from_proto_and_back(self):
        """Ensure that we can convert to proto and back and have the same data."""
        reconstructed_tphrase = dm.TopicPhrase.from_proto(self.topic_phrase.to_proto())
        self.assertEqual(self.topic_phrase.text, reconstructed_tphrase.text)
        self.assertAlmostEqual(
            self.topic_phrase.distance, reconstructed_tphrase.distance
        )

    def test_from_json_and_back(self):
        """Ensure that we can convert to json and back and have the same data."""
        reconstructed_tphrase = dm.TopicPhrase.from_json(self.topic_phrase.to_json())
        self.assertEqual(self.topic_phrase.text, reconstructed_tphrase.text)
        self.assertAlmostEqual(
            self.topic_phrase.distance, reconstructed_tphrase.distance
        )

    def test_init_with_bad_type(self):
        """Ensure that we throw type error if a sad type(s) is provided."""
        happy_args = {"text": "foo", "distance": 1}
        bad_arg_overrides = [
            {"text": 13},
            {"distance": "bar"},
        ]
        for override in bad_arg_overrides:
            # Ensure that any time we override a happy arg with a sad type, we are sad
            with self.assertRaises(TypeError):
                sad_args = {**happy_args, **override}
                dm.TopicPhrase(**sad_args)


class TestTopicsPrediction(TestCaseBase):
    def setUp(cls):
        sample_topic = dm.Topic(
            name="Foobar",
            score=0.5,
            size=1,
            ngrams=[dm.NGram(texts=["foo", "bar", "baz"], relevance=0.0)],
            snippets=[dm.TopicPhrase(text="foo", distance=0.7)],
            sentences=[dm.TopicPhrase(text="foo", distance=0.7)],
            producer_id=ProducerId("TopicTest", "1.2.5"),
        )
        cls.topics_prediction = dm.TopicsPrediction([sample_topic])

    def test_fields(self):
        """Ensure that we are able to validate the fields of our TopicsPrediction DM object."""
        self.assertTrue(self.validate_fields(self.topics_prediction))

    def test_from_proto_and_back(self):
        """Ensure that we can convert to proto and back and have the same data."""
        reconstructed_tpreds = dm.TopicsPrediction.from_proto(
            self.topics_prediction.to_proto()
        )
        # Make sure we have the same number of topics in our reconstructed output
        self.assertEqual(
            len(self.topics_prediction.topics), len(reconstructed_tpreds.topics)
        )
        # Then compare individual topics
        for orig_topic, recon_topic in zip(
            self.topics_prediction.topics, reconstructed_tpreds.topics
        ):
            compare_topics(self, orig_topic, recon_topic)

    def test_from_json_and_back(self):
        """Ensure that we can convert to json and back and have the same data."""
        reconstructed_tpreds = dm.TopicsPrediction.from_json(
            self.topics_prediction.to_json()
        )
        # Make sure we have the same number of topics in our reconstructed output
        self.assertEqual(
            len(self.topics_prediction.topics), len(reconstructed_tpreds.topics)
        )
        # Then compare individual topics
        for orig_topic, recon_topic in zip(
            self.topics_prediction.topics, reconstructed_tpreds.topics
        ):
            compare_topics(self, orig_topic, recon_topic)

    def test_init_with_bad_type(self):
        """Ensure that we throw type error if a sad type(s) is provided."""
        with self.assertRaises(TypeError):
            dm.TopicsPrediction("A bad type")
