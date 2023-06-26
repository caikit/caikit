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


"""DataStream resolver that resolves a given file or Datastream to a DataStream
"""
# Standard
from typing import Dict

# First Party
import alog

# Local
from ...toolkit.errors import error_handler
from .. import DataStream
from .converter import DataStreamConverter
from .csv_column_formatter import CSVColumnFormatter
from .validator import DataStreamValidator

log = alog.use_channel("DATSTRMRSLVR")
error = error_handler.get(log)


class DataStreamResolver:
    """Resolves files or DataStreams into DataStreams"""

    def __init__(self, target_stream_type: type, expected_keys: Dict[str, type]):
        """Initialize DataStreamResolver

        Args:
            target_stream_type (type): The target type for the data items in the
                resolved stream. dict and list are supported.
            expected_keys (dict(str, type)): Dictionary of keys -> types that
                determines how data will be formatted in the data stream.

                If you want a stream of dictionaries, we'll try to put these as keys in each one.
                If you want a stream of lists, we'll try to locate these values and place them in
                each list in order.

                In either case, we'll validate that each piece of data is of the type specified.

                See the DataStreamConverter docs for more info.
        """
        error.type_check("<COR24496441E>", type, target_stream_type=target_stream_type)
        error.type_check("<COR24496442E>", dict, expected_keys=expected_keys)

        self._converter: DataStreamConverter = DataStreamConverter(
            target_stream_type, list(expected_keys.keys())
        )
        self._validator: DataStreamValidator = DataStreamValidator(expected_keys)
        self._csv_formatter: CSVColumnFormatter = CSVColumnFormatter(expected_keys)

    def as_data_stream(self, file_or_data_stream) -> DataStream:
        """Marshals whatever you give it into a data stream, or dies trying.
        ...Or leaves the error for you to find later when reading the stream.

        Args:
            file_or_data_stream (str or DataStream): Either a string path to a
                file, or a DataStream
        Returns:
            DataStream: The data as a converted and properly formatted data stream
        """
        error.type_check(
            "<COR92088414E>", str, DataStream, file_or_data_stream=file_or_data_stream
        )

        if isinstance(file_or_data_stream, str):
            # File here
            loaded_stream = DataStream.from_file(file_or_data_stream)
            listified_stream = self._csv_formatter.format(loaded_stream)
            validated_stream = self._validator.validate(listified_stream)
            return self._converter.convert(validated_stream)

        # Datastream here
        listified_stream = self._csv_formatter.format(file_or_data_stream)
        validated_stream = self._validator.validate(listified_stream)
        return self._converter.convert(validated_stream)
