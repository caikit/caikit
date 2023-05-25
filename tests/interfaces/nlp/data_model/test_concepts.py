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


# Standard
import re

# Local
from caikit.interfaces.nlp import data_model as dm

# Unit Test Infrastructure
from tests.base import TestCaseBase


class TestConcept(TestCaseBase):
    def setUp(self):
        self.concept1 = dm.Concept("Barack_Obama")
        self.concept2 = dm.Concept(
            "Linux",
            relevance=0.79,
            dbpedia_resource="https://dbpedia.org/resource/Linux",
        )

    def test_empty(self):
        empty_concept = dm.Concept("")
        self.assertEqual(empty_concept.text, "")

    def test_defaults(self):
        self.assertAlmostEqual(self.concept1.relevance, 0.0)
        self.assertEqual(self.concept1.dbpedia_resource, "")

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.concept1))
        self.assertTrue(self.validate_fields(self.concept2))

    def test_from_proto_and_back(self):
        new = dm.Concept.from_proto(self.concept1.to_proto())
        self.assertEqual(new.text, self.concept1.text)
        self.assertAlmostEqual(new.relevance, self.concept1.relevance)
        self.assertEqual(new.dbpedia_resource, self.concept1.dbpedia_resource)

        new = dm.Concept.from_proto(self.concept2.to_proto())
        self.assertEqual(new.text, self.concept2.text)
        self.assertAlmostEqual(new.relevance, self.concept2.relevance)
        self.assertEqual(new.dbpedia_resource, self.concept2.dbpedia_resource)

    def test_from_json_and_back(self):
        new = dm.Concept.from_json(self.concept1.to_json())
        self.assertEqual(new.text, self.concept1.text)
        self.assertAlmostEqual(new.relevance, self.concept1.relevance)
        self.assertEqual(new.dbpedia_resource, self.concept1.dbpedia_resource)

        new = dm.Concept.from_json(self.concept2.to_json())
        self.assertEqual(new.text, self.concept2.text)
        self.assertAlmostEqual(new.relevance, self.concept2.relevance)
        self.assertEqual(new.dbpedia_resource, self.concept2.dbpedia_resource)

    def test_compare(self):
        self.assertLess(self.concept1, self.concept2)

    def test_dbpedia_resource_format(self):
        dbpedia_format_regex = dm.Concept.dbpedia_format_regex

        self.assertTrue(
            re.match(dbpedia_format_regex, "https://en.dbpedia.org/resource/IBM")
        )
        self.assertTrue(
            re.match(dbpedia_format_regex, "https://en.dbpedia.org/resource/a/b")
        )
        self.assertTrue(
            re.match(dbpedia_format_regex, "https://dbpedia.org/resource/IBM")
        )
        self.assertTrue(
            re.match(dbpedia_format_regex, "http://en.dbpedia.org/resource/IBM")
        )
        self.assertTrue(
            re.match(dbpedia_format_regex, "http://dbpedia.org/resource/IBM")
        )
        self.assertTrue(
            re.match(dbpedia_format_regex, "https://de.dbpedia.org/resource/IBM")
        )
        self.assertTrue(
            re.match(dbpedia_format_regex, "https://zh-cn.dbpedia.org/resource/IBM")
        )

        self.assertFalse(re.match(dbpedia_format_regex, ""))
        self.assertFalse(re.match(dbpedia_format_regex, "invalid"))
        self.assertFalse(
            re.match(dbpedia_format_regex, "htt://dbpedia.org/resource/IBM")
        )
        self.assertFalse(re.match(dbpedia_format_regex, "dbpedia.org/resource/IBM"))
        self.assertFalse(re.match(dbpedia_format_regex, "dbpedia.org/resource/"))
        self.assertFalse(re.match(dbpedia_format_regex, "en.dbpedia.org/resource/IBM"))
        self.assertFalse(
            re.match(dbpedia_format_regex, "http://en.dbpedia.org/resource/")
        )


class TestConceptPrediction(TestCaseBase):
    def setUp(self):
        self.concepts_prediction = dm.ConceptsPrediction(
            concepts=[
                dm.Concept("Linux", 0.99, "https://dbpedia.org/resource/Linux"),
                dm.Concept("IBM", 0.9, "http://en.dbpedia.org/resource/IBM"),
                dm.Concept("AIX", 0.8),
                dm.Concept("Z/OS", 0.5),
            ],
            producer_id=dm.ProducerId("BiLSTM Concepts", "0.1"),
        )

    def test_empty_concepts(self):
        concepts_empty = dm.ConceptsPrediction(concepts=[])
        self.assertEqual(len(concepts_empty.concepts), 0)
        self.assertEqual(concepts_empty.producer_id, None)

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.concepts_prediction))

    def test_from_proto_and_back(self):
        new = dm.ConceptsPrediction.from_proto(self.concepts_prediction.to_proto())

        self.assertEqual(len(new.concepts), 4)
        self.assertEqual(len(new.concepts), len(self.concepts_prediction.concepts))
        self.assertEqual(
            new.producer_id.name, self.concepts_prediction.producer_id.name
        )
        for concept1, concept2 in zip(new.concepts, self.concepts_prediction.concepts):
            self.assertEqual(concept1.text, concept2.text)
            self.assertAlmostEqual(concept1.relevance, concept2.relevance)
            self.assertEqual(concept1.dbpedia_resource, concept2.dbpedia_resource)

    def test_from_json_and_back(self):
        new = dm.ConceptsPrediction.from_json(self.concepts_prediction.to_json())

        self.assertEqual(len(new.concepts), 4)
        self.assertEqual(len(new.concepts), len(self.concepts_prediction.concepts))
        self.assertEqual(
            new.producer_id.name, self.concepts_prediction.producer_id.name
        )
        for concept1, concept2 in zip(new.concepts, self.concepts_prediction.concepts):
            self.assertEqual(concept1.text, concept2.text)
            self.assertAlmostEqual(concept1.relevance, concept2.relevance)
            self.assertEqual(concept1.dbpedia_resource, concept2.dbpedia_resource)
