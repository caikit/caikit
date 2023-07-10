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


"""DataStream converter that attempts to consolidate some columns from a CSV.
Useful for both collapsing multiple columns of a CSV stream into one list, or creating a list out of
a single column from a CSV file.
"""
# Standard
from typing import Dict

# First Party
import alog

# Local
from ...toolkit.errors import error_handler
from .. import DataStream

log = alog.use_channel("CSVCLMNCNSLDTR")
error = error_handler.get(log)


class CSVColumnFormatter:
    """Consolidates column(s) of data items in a csv stream.
    Only operates on streams of lists.

    For example, for an input stream that looks like:
        [
            ["foo", "bar"],
            ["bazz", "buzz", "bozz"]
        ]
    If the expected_columns are {'text': str, 'labels': list}, the result will be:
        [
            ["foo", ["bar"]],
            ["bazz", ["buzz, "bozz"]]
        ]

    >>> list_of_lists = [["foo", "bar"], ["bazz", "buzz", "bozz"]]
    >>> list_stream = DataStream.from_iterable(list_of_lists)
    >>> converter = CSVColumnFormatter(expected_columns={'text': str, 'labels': list})
    >>> list_stream = converter.format(list_stream)
    >>> # [ ["foo", ["bar"]], ["bazz", ["buzz, "bozz"]] ]
    """

    def __init__(self, expected_columns: Dict[str, type]):
        """Initialize CSVColumnConsolidator

        Args:
            expected_columns (Dict(str, type)): (Ordered) dictionary mapping the
                csv column names to expected types
        """
        error.type_check("<COR56775937E>", dict, expected_columns=expected_columns)
        error.value_check(
            "<COR98237989E>", expected_columns, "`expected_columns` should be nonempty"
        )
        self._expected_columns: Dict[str, type] = expected_columns

    def format(self, stream: DataStream) -> DataStream:
        """Attempt to format some CSV columns. Only operates on streams of lists.

        Pulls column elements into lists if the column type is `list`
        Slurps remaining columns into the last column if the last column type is `list`

        See classdoc for examples

        Args:
            stream (DataStream): stream intended to be converted

        Returns:
            DataStream:
                stream with internal elements of each item converted
        """

        # Element listification only works for lists.
        # If neither is asked for, do nothing.
        if list not in self._expected_columns.values():
            return stream

        def _convert(data_item):
            if isinstance(data_item, list):
                # Don't mutate a list that we're iterating on here
                # Subsequent re-entries into the stream would mutate the list further
                # and really mess things up
                data_item_copy = list(data_item[:])

                last_type = None
                for i, (element, type_) in enumerate(
                    zip(data_item, self._expected_columns.values())
                ):
                    last_type = type_
                    if type_ == list:
                        data_item_copy[i] = CSVColumnFormatter._attempt_to_listify(
                            element
                        )

                if len(data_item) > len(self._expected_columns):
                    # More data in the data item left...
                    if last_type == list:
                        # Last element was a list, so slurp the rest of the row in
                        length = len(self._expected_columns)

                        data_item_copy[length - 1].extend(data_item[length:])
                        data_item_copy = data_item_copy[0:length]

                return data_item_copy

            # Do nothing if thing wasn't a list
            return data_item

        return stream.map(_convert)

    @staticmethod
    def _attempt_to_listify(data):
        """Mostly we want to return [data] if this thing isn't already a list.
        But we don't want to do things like convert a dictionary into a list of keys.

        So we'll just check for some common types, and listify those.
        """
        if isinstance(data, (bool, float, int, str)):
            return [data]
        return data
