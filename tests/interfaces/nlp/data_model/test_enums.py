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

# Third Party

# Local
from caikit.interfaces.nlp import data_model

# Unit Test Infrastructure
from tests.base import TestCaseBase


class TestEnums(TestCaseBase):
    def test_forward_lookups(self):
        self.assertEqual(data_model.PartOfSpeech(0).value, 0)
        self.assertEqual(data_model.PartOfSpeech.POS_UNSET.value, 0)

        self.assertEqual(data_model.DependencyRelation(0).value, 0)
        self.assertEqual(data_model.DependencyRelation.DEP_OTHER.value, 0)

    def test_reverse_lookups(self):
        self.assertEqual(data_model.PartOfSpeech(0).name, "POS_UNSET")
        self.assertEqual(
            data_model.PartOfSpeech(data_model.PartOfSpeech.POS_UNSET.value).name,
            "POS_UNSET",
        )

        self.assertEqual(data_model.DependencyRelation(0).name, "DEP_OTHER")
        self.assertEqual(
            data_model.DependencyRelation(data_model.DependencyRelation(0).value).name,
            "DEP_OTHER",
        )
