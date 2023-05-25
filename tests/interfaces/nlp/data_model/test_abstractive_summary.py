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


class TestAbstractiveSummary(TestCaseBase):
    def setUp(self):
        self.abstractive_summary = dm.AbstractiveSummary(
            summary=[
                "The customer received a replacement macbook that is defective and was told that apple care will take two weeks.",
                "The agent informs the customer that an incident ticket might be faster, but replacement could be rejected.",
            ]
        )

    def test_fields(self):
        self.assertTrue(self.validate_fields(self.abstractive_summary))

    def test_from_proto_and_back(self):
        new = dm.AbstractiveSummary.from_proto(self.abstractive_summary.to_proto())
        self.assertEqual(len(new.summary), len(self.abstractive_summary.summary))
        self.assertEqual(new.summary[0], self.abstractive_summary.summary[0])
        self.assertEqual(new.summary[1], self.abstractive_summary.summary[1])
        self.assertEqual(new.producer_id, self.abstractive_summary.producer_id)

    def test_from_json_and_back(self):
        new = dm.AbstractiveSummary.from_json(self.abstractive_summary.to_json())
        self.assertEqual(len(new.summary), len(self.abstractive_summary.summary))
        self.assertEqual(new.summary[0], self.abstractive_summary.summary[0])
        self.assertEqual(new.summary[1], self.abstractive_summary.summary[1])
        self.assertEqual(new.producer_id, self.abstractive_summary.producer_id)
