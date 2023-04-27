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
from typing import Dict
import threading
import unittest
import uuid

# Third Party
from prometheus_client import core, parser, start_http_server
import requests

# First Party
import alog

# Local
from caikit import get_config
from caikit.runtime.metrics.throughput import Throughput

log = alog.use_channel("TEST-THROUGHPT")
# add 1 to get a new unused metrics port for this test. (Can't seem to shutdown the prometheus server)
METRICS_PORT = int(get_config().runtime.metrics.port) + 1


def scrape_metrics() -> Dict[str, core.Metric]:
    r = requests.get("http://localhost:{}".format(METRICS_PORT))
    metrics = {}
    for metric in parser.text_string_to_metric_families(r.text):
        metrics[metric.name] = metric
    return metrics


def random_id():
    return "test_metric_" + str(uuid.uuid4()).replace("-", "_")


class TestThroughput(unittest.TestCase):
    """This test suite tests the throughput metric class"""

    LABEL = "test_label"

    @classmethod
    def setUpClass(cls):
        """This method runs before all the tests begin to run"""
        # Spin up prometheus metrics reporting, and hope this doesn't fail if it happens more than once
        start_http_server(int(METRICS_PORT))

    def test_it_does_not_barf_if_no_input_size_given(self):
        metric_name = random_id()
        metric = Throughput(metric_name, "", [self.LABEL])

        metric.labels(test_label="foo").observe(1.234)
        # assert nothing reported
        metrics = self.get_metrics(metric_name, self.LABEL, "foo")
        self.assertEqual(metrics["count"], 0)
        self.assertEqual(metrics["sum"], 0)

    def test_it_does_not_barf_on_zero_time_elapsed(self):
        metric_name = random_id()
        metric = Throughput(metric_name, "", [self.LABEL])

        metric.input_size(12345).labels(test_label="foo").observe(0)
        # assert no divide by zero error, and nothing reported
        metrics = self.get_metrics(metric_name, self.LABEL, "foo")
        self.assertEqual(metrics["count"], 0)
        self.assertEqual(metrics["sum"], 0)

    def test_it_reports_throughput_based_on_input_size(self):
        metric_name = random_id()
        metric = Throughput(metric_name, "", [self.LABEL])

        input_size = 12345
        time_seconds = 2
        expected_throughput = input_size / time_seconds

        metric.input_size(input_size).labels(test_label="foo").observe(time_seconds)
        # assert correct throughput calculated and reported
        metrics = self.get_metrics(metric_name, self.LABEL, "foo")
        self.assertEqual(metrics["count"], 1)
        self.assertEqual(metrics["sum"], expected_throughput)

    def test_it_does_not_blow_up_reporting_multiple_metrics(self):
        metric_name = random_id()
        metric = Throughput(metric_name, "", [self.LABEL])

        input_size = 12345
        time_seconds = 2
        expected_throughput = input_size / time_seconds

        metric.input_size(input_size).labels(test_label="foo").observe(time_seconds)
        metric.input_size(input_size).labels(test_label="foo").observe(time_seconds)
        metric.input_size(input_size).labels(test_label="foo").observe(time_seconds)

        # assert correct throughput calculated and reported
        metrics = self.get_metrics(metric_name, self.LABEL, "foo")
        self.assertEqual(metrics["count"], 3)
        self.assertEqual(metrics["sum"] / metrics["count"], expected_throughput)

    def test_input_sizes_are_not_kept_around_to_pollute_subsequent_metrics(self):
        metric_name = random_id()
        metric = Throughput(metric_name, "", [self.LABEL])

        input_size = 12345
        time_seconds = 2
        # This call should use the input size
        metric.input_size(input_size).labels(test_label="foo").observe(time_seconds)
        # And this should do nothing, since no input was given
        metric.labels(test_label="foo").observe(time_seconds)
        # So the count should be 1
        metrics = self.get_metrics(metric_name, self.LABEL, "foo")
        self.assertEqual(metrics["count"], 1)

    def test_input_sizes_are_thread_local(self):
        metric_name = random_id()
        metric = Throughput(metric_name, "", [self.LABEL])

        input_size = 12345
        time_seconds = 2
        expected_throughput = input_size / time_seconds

        def tput(factor):
            metric.input_size(input_size * factor).labels(test_label="foo").observe(
                time_seconds * factor
            )

        # Start a bunch of threads, each of them reporting the same throughput but at different scales.
        # The end result should be the same
        threads = []
        for i in range(1, 1000):
            thread = threading.Thread(target=tput, args=(i,))
            threads.append(thread)
            thread.start()
        [t.join() for t in threads]

        metrics = self.get_metrics(metric_name, self.LABEL, "foo")
        self.assertEqual(metrics["sum"] / metrics["count"], expected_throughput)

    @staticmethod
    def get_metrics(metric, label_name, label_value):
        """This function grabs samples for a metric with label_name matching label_value.
        It does some funky business to pull out things like count and sum in an easy to use format.
        It looks at all metrics that have fired from all tests in this class, so tests can pollute each other
        if they use the same metric names"""
        mets = scrape_metrics()[metric]
        easy_metric_dict = {}
        for s in mets.samples:
            if s.labels[label_name] == label_value:
                # For sample names like '<metric>_count', convert to 'count'
                key = s.name.replace(metric + "_", "")
                easy_metric_dict[key] = s.value
        return easy_metric_dict


if __name__ == "__main__":
    unittest.main()
