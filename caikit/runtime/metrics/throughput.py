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
import threading

# Third Party
from prometheus_client import Summary


class Throughput(Summary):
    """
    This class extends a Summary metric to calculate throughput as `input_size / seconds`.
    Assumes input time to `observe()` is in milliseconds.

    Usage:
        # If you timed something yourself:
        throughput_metric.input_size(my_input_size).labels(my_labels).observe(elapsed_seconds)

        # As a context manager:
        with throughput_metric.input_size(my_input_size).labels(my_labels).time():
            work.do()

    Quirks:
        Doesn't work without labels
    """

    _thread_input_size_map = {}

    def input_size(self, input_size):
        """
        Set the size of the input to use for throughput calculations

        Args:
            input_size: The amount of stuff that will be processed.
                        Typically text size measured in kilo-code-points

        Returns:
            self, for chaining a call to `.time()` or `.observe()`
        """
        self._thread_input_size_map[threading.get_ident()] = input_size
        return self

    def observe(self, amount):
        """
        Overloads Summary.observe(). Report the elapsed processing time to calculate and collect
        a throughput metric

        Args:
            amount: elapsed time in seconds
        """
        # If time was less than 1 nanosecond, don't record anything.
        # We shouldn't have clocks with this resolution, and we don't want divide-by-zero errors
        # or infinity throughput
        if abs(amount) < 0.000001:
            return

        if threading.get_ident() in self._thread_input_size_map:
            super().observe(
                self._thread_input_size_map.pop(threading.get_ident()) / amount
            )
