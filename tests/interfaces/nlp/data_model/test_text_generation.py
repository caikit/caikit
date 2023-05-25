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


class GeneratedResult(TestCaseBase):
    def setUp(self):
        self.generation_prediction = dm.GeneratedResult(
            text="It is 20 degrees today", stop_reason=1, generated_token_count=100
        )

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.generation_prediction))

    def test_from_proto_and_back(self):
        new = dm.GeneratedResult.from_proto(self.generation_prediction.to_proto())
        self.assertEqual(new.text, self.generation_prediction.text)
        self.assertEqual(new.stop_reason, self.generation_prediction.stop_reason)
        self.assertEqual(
            new.generated_token_count, self.generation_prediction.generated_token_count
        )

    def test_from_json_and_back(self):
        new = dm.GeneratedResult.from_json(self.generation_prediction.to_json())
        self.assertEqual(new.text, self.generation_prediction.text)
        self.assertEqual(new.stop_reason, self.generation_prediction.stop_reason)
        self.assertEqual(
            new.generated_token_count, self.generation_prediction.generated_token_count
        )
