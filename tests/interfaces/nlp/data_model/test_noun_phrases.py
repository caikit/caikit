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


class TestNounPhrase(TestCaseBase):
    def setUp(self):
        self.noun_phrase = dm.NounPhrase(dm.Span(0, 11, text="Hello World"))
        self.noun_phrase_minimal = dm.NounPhrase((0, 20))

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.noun_phrase))
        self.assertTrue(self.validate_fields(self.noun_phrase_minimal))

    def test_from_proto_and_back(self):
        new = dm.NounPhrase.from_proto(self.noun_phrase.to_proto())
        self.assertEqual(new.span.begin, self.noun_phrase.span.begin)
        self.assertEqual(new.span.end, self.noun_phrase.span.end)

        new = dm.NounPhrase.from_proto(self.noun_phrase_minimal.to_proto())
        self.assertEqual(new.span.begin, self.noun_phrase_minimal.span.begin)
        self.assertEqual(new.span.end, self.noun_phrase_minimal.span.end)

    def test_from_json_and_back(self):
        new = dm.NounPhrase.from_json(self.noun_phrase.to_json())
        self.assertEqual(new.span.begin, self.noun_phrase.span.begin)
        self.assertEqual(new.span.end, self.noun_phrase.span.end)

        new = dm.NounPhrase.from_json(self.noun_phrase_minimal.to_json())
        self.assertEqual(new.span.begin, self.noun_phrase_minimal.span.begin)
        self.assertEqual(new.span.end, self.noun_phrase_minimal.span.end)
