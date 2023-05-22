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
from collections import Counter
from multiprocessing.pool import ThreadPool
from unittest.mock import patch
import json
import unittest

# Local
from caikit.runtime.metrics.rpc_meter import RPCMeter


class TestRPCMeter(unittest.TestCase):
    """This test suite tests the RPCMeter class which is using for RPC metering"""

    def setUp(self):
        """This method runs before each test begins to run"""
        self.rpc_meter = RPCMeter()
        # Note: These are not real classes in sample_lib
        self.another_widget_type = "<class 'sample_lib.modules.sample_task.sample_implementation.AnotherWidget'>"
        self.another_fidget_type = "<class 'sample_lib.modules.sample_task.sample_implementation.AnotherFidget'>"

    def test_update_metrics(self):
        pool = ThreadPool(1000)
        inputs = [self.another_widget_type for i in range(500)]
        inputs.extend([self.another_fidget_type for i in range(500)])
        pool.map(self.rpc_meter.update_metrics, inputs)

        self.assertEqual(len(self.rpc_meter.predict_rpc_counter), 1000)
        counter = Counter(self.rpc_meter.predict_rpc_counter)
        self.assertListEqual(
            list(counter.keys()), [self.another_widget_type, self.another_fidget_type]
        )
        self.assertListEqual(list(counter.values()), [500, 500])

    @patch("threading.Thread")
    def test_write_metrics(self, mock_thread):
        mock_thread.return_value = None
        with patch.object(
            self.rpc_meter, "predict_rpc_counter", [self.another_widget_type]
        ):
            self.rpc_meter._write_metrics()

        with patch.object(
            self.rpc_meter,
            "predict_rpc_counter",
            [self.another_fidget_type, self.another_fidget_type],
        ):
            self.rpc_meter._write_metrics()

        with patch.object(
            self.rpc_meter,
            "predict_rpc_counter",
            [self.another_widget_type, self.another_fidget_type],
        ):
            self.rpc_meter._write_metrics()

        data = []
        with open(self.rpc_meter.file_path) as f:
            data = [json.loads(line) for line in f]

        expected_keys = [
            "timestamp",
            "batch_size",
            "model_type_counters",
            "container_id",
        ]

        self.assertEqual(len(data), 3)
        self.assertListEqual(
            list(data[0].keys()),
            expected_keys,
        )
        self.assertEqual(data[0]["batch_size"], 1)
        self.assertEqual(len(data[0]["model_type_counters"]), 1)
        self.assertDictEqual(
            data[0]["model_type_counters"],
            {self.another_widget_type: 1},
        )

        self.assertEqual(data[1]["batch_size"], 2)
        self.assertEqual(len(data[1]["model_type_counters"]), 1)
        self.assertDictEqual(
            data[1]["model_type_counters"],
            {self.another_fidget_type: 2},
        )

        self.assertEqual(data[2]["batch_size"], 2)
        self.assertEqual(len(data[2]["model_type_counters"]), 2)
        self.assertDictEqual(
            data[2]["model_type_counters"],
            {self.another_widget_type: 1, self.another_fidget_type: 1},
        )

    def tearDown(self):
        self.rpc_meter.end_writer_thread()
