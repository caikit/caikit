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


class TestLangDetect(TestCaseBase):
    def setUp(self):
        self.lang_detect = dm.LangDetectPrediction(lang_code=dm.LangCode.LANG_EN.value)

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.lang_detect))

    def test_from_proto_and_back(self):
        new = dm.LangDetectPrediction.from_proto(self.lang_detect.to_proto())
        self.assertEqual(new.lang_code, self.lang_detect.lang_code)

    def test_from_json_and_back(self):
        new = dm.LangDetectPrediction.from_json(self.lang_detect.to_json())
        self.assertEqual(new.lang_code, self.lang_detect.lang_code)

    def test_to_string(self):
        self.assertEqual(self.lang_detect.to_string(), "LANG_EN")

    def test_to_iso_format(self):
        self.assertEqual(self.lang_detect.to_iso_format(), "EN")
