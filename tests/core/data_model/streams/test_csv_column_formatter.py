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
from caikit.core.data_model import DataStream
from caikit.core.data_model.streams.csv_column_formatter import CSVColumnFormatter

# Unit Test Infrastructure
from tests.base import TestCaseBase


class TestCSVColumnFormatter(TestCaseBase):
    def test_does_nothing_if_no_expected_types_are_lists(self):
        formatter = CSVColumnFormatter({"a": int, "b": int})
        list_stream = DataStream.from_iterable([[1, 2], [3, 4]])
        self.assertEqual(list_stream, formatter.format(list_stream))

    def test_it_is_okay_with_empty_streams(self):
        formatter = CSVColumnFormatter({"a": int, "b": list})
        empty_stream = DataStream.from_iterable([])
        stream = formatter.format(empty_stream)
        self.assertEqual([], list(stream))

    def test_it_listifies_basic_types_on_list_streams(self):
        formatter = CSVColumnFormatter({"a": int, "b": list})
        basic_stream = [[1, 2], [3, 4.0], [5, "6.0"], [7, True], [8, [9]]]
        expected_stream = [[1, [2]], [3, [4.0]], [5, ["6.0"]], [7, [True]], [8, [9]]]
        stream = DataStream.from_iterable(basic_stream)
        self.assertEqual(expected_stream, list(formatter.format(stream)))

    def test_it_does_not_listify_dicts_or_tuples(self):
        formatter = CSVColumnFormatter({"a": list})
        # Dicts and Tuples as elements of a list stream are not converted
        stream = [[{"foo": "bar"}], [("foo", "bar")]]
        stream = DataStream.from_iterable(stream)
        self.assertEqual(list(stream), list(formatter.format(stream)))

    def test_it_collapses_extra_elements_into_a_list(self):
        formatter = CSVColumnFormatter({"text": str, "labels": list})
        stream = [["foo", "bar"], ["foo", "bar", "baz", "buz"]]
        expected_stream = [["foo", ["bar"]], ["foo", ["bar", "baz", "buz"]]]
        stream = DataStream.from_iterable(stream)
        self.assertEqual(expected_stream, list(formatter.format(stream)))

    def test_it_does_not_process_items_that_are_not_lists(self):
        formatter = CSVColumnFormatter({"text": str, "labels": list})
        stream = [["foo", "bar"], "baz", 5, ("foo", "bar"), {"foo": "bar"}]
        expected_stream = [["foo", ["bar"]], "baz", 5, ("foo", "bar"), {"foo": "bar"}]
        stream = DataStream.from_iterable(stream)
        self.assertEqual(expected_stream, list(formatter.format(stream)))

    def test_it_does_not_barf_if_lists_have_fewer_elements_than_expected(self):
        formatter = CSVColumnFormatter({"text": str, "labels": list})
        stream = [["foo", "bar"], ["baz"], [5]]
        expected_stream = [["foo", ["bar"]], ["baz"], [5]]
        stream = DataStream.from_iterable(stream)
        self.assertEqual(expected_stream, list(formatter.format(stream)))
