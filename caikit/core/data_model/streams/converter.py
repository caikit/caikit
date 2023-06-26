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


"""DataStream converter that converts a given Datastream to either a DataStream of lists or
dictionaries.
These type conversions are encapsulated here right now for consistency in the workflows served in
production cloud offerings, but we could see moving these to be directly on the DataStream class.
"""

# Standard
from typing import Callable, Dict, List

# First Party
import alog

# Local
from ...toolkit.errors import error_handler
from .. import DataStream

log = alog.use_channel("DATSTRMCNVRTR")
error = error_handler.get(log)


class DataStreamConverter:
    """Converts DataStreams to a target type

    This uses a target type and a list of expected keys to convert each data item in the stream.

    For example, for an input stream that looks like:
        [
            { a: 1, b: 2, c: 3 }
            { a: 4, b: 5, c: 6 }
            { a: 7, b: 8, c: 9 }
        ]
    If target_type is list with key_list of ['a','c'], the result will be:
        [
            [1, 3]
            [4, 6]
            [7, 9]
        ]

    Or, for an input stream that looks like:
        [
            [1, 3]
            [4, 6]
            [7, 9]
        ]
    If target_type is dict with key_list of ['foo', 'bar'], the result will be:
        [
            { foo: 1, bar: 3 }
            { foo: 4, bar: 6 }
            { foo: 7, bar: 9 }
        ]

    >>> list_of_dicts = [{"foo": 1, "bar": 2}, {"foo": 3, "bar": 4}]
    >>> dict_stream = DataStream.from_iterable(list_of_dicts)
    >>> converter = DataStreamConverter(target_type=list, key_list=['foo', 'bar'])
    >>> list_stream = converter.convert(dict_stream)
    """

    def __init__(self, target_type: type, key_list: List[str]):
        """Initialize DataStreamConverter

        Args:
            target_type (type): Target DataStream type, either dict or list
            key_list (List(str)): List of keys that determines how data will be
                formatted into a DataStream
        """
        error.type_check("<COR56775827E>", type, target_type=target_type)
        error.type_check_all("<COR16523028E>", str, key_list=key_list)
        # Tuple support can be added here in the future if anybody needs it
        self._conversion_function_map: Dict[type, Callable] = {
            dict: self._convert_stream_to_dicts,
            list: self._convert_stream_to_lists,
        }

        # only support streams of dict or list for now
        if target_type not in self._conversion_function_map:
            error(
                "<COR77775827E>",
                ValueError(
                    "Conversion of stream data items to type {} not supported".format(
                        target_type
                    )
                ),
            )

        self._target_type: type = target_type
        self._key_list: List[str] = key_list
        error.value_check(
            "<COR98237539E>", self._key_list, "`key_list` should be nonempty"
        )

    def convert(self, stream: DataStream) -> DataStream:
        """Attempt to convert a given datastream to a datastream of the target type
        See classdoc for examples

        Args:
            stream (DataStream): stream intended to be converted

        Returns:
            Converted datastream based on the target type
        """
        # check the target type
        return self._conversion_function_map[self._target_type](stream)

    def _convert_stream_to_dicts(self, stream: DataStream) -> DataStream:
        """Attempt to convert a stream to dictionaries

        Args:
            stream (DataStream): Stream to convert to a stream of dictionaries
        Returns:
            A stream which will lazily convert data items to dictionaries
        """

        def convert(data_item):
            if isinstance(data_item, dict):
                log.debug(
                    "Trying to convert stream to dict, but data item was already a dict"
                )
                return data_item

            return {key: data_item[index] for index, key in enumerate(self._key_list)}

        return stream.map(convert)

    def _convert_stream_to_lists(self, stream: DataStream) -> DataStream:
        """Attempt to convert a stream to lists

        Args:
            stream (DataStream): Stream to convert to a stream of lists
        Returns:
            A stream which will lazily convert data items to lists
        """

        def convert(data_item):
            if isinstance(data_item, list):
                log.debug(
                    "Trying to convert stream to list, but data item was already a list"
                )
                return data_item

            if isinstance(data_item, tuple):
                log.debug("Converting tuple to list")
                return list(data_item)

            return [data_item[key] for key in self._key_list]

        return stream.map(convert)
