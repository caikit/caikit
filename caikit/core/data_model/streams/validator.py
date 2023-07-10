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


"""DataStream validator that validates that a datastream includes the expected
parts - for dictionaries, that all expected dictionary keys exist, and for
iterables, that the iterable length is the expected length.
The types of the internal elements of each item are also validated.
"""
# Standard
from typing import Dict

# First Party
import alog

# Local
from ...toolkit.errors import DataValidationError, error_handler
from .. import DataStream

log = alog.use_channel("DATSTRMVLDTR")
error = error_handler.get(log)


class DataStreamValidator:
    """Validates that DataStreams contain the expected data items based on type"""

    def __init__(self, expected_keys: Dict[str, type]):
        """Initialize DataStreamValidator

        Args:
            expected_keys (Dict(str)): (Ordered) dictionary of key names ->
                types that determines how data will be validated
        """
        self._expected_keys: Dict[str, type] = expected_keys
        error.value_check(
            "<COR53878661E>", self._expected_keys, "`expected_keys` should be nonempty"
        )

    def validate(self, stream: DataStream) -> DataStream:
        """Attempt to validate the data items in a data stream (lazily)

        Validation checks:
        1. That the data in the stream is either a dictionary or list or tuple
        2. In the case of dictionaries, all expected keys exist and the values are the types
        given in
            `self._expected_keys`
        3. In the case of iterables, that the length of each iterable matches the expected number of
            items and the type of the nth item matches the nth type in `self._expected_keys`

        Args:
            stream (DataStream): stream intended to be converted

        Returns:
            The same data stream, which will now throw DataValidationErrors when accessed if the
            data is not valid.
        """
        data_item_number = -1

        def validate(data_item):
            # Enclose the counter here for later mutation when this map function is applied
            nonlocal data_item_number
            data_item_number += 1
            self._validate_data(data_item, data_item_number)
            return data_item

        return stream.map(validate)

    def _validate_data(self, data_item: object, data_item_number: int) -> None:
        """Validate a single data item from a data stream

        Args:
            data_item: A data object yielded by the stream
            data_item_number: The index of the object in the stream
        """
        if isinstance(data_item, dict):
            # From dictionary: error if key doesn't exist
            missing_keys = set(self._expected_keys.keys()) - set(data_item.keys())
            if len(missing_keys) > 0:
                message = "Data item is missing required key(s): {} ".format(
                    missing_keys
                )
                raise DataValidationError(message, data_item_number)
            # Error if any key is wrong type
            for key, type_ in self._expected_keys.items():
                if not isinstance(data_item[key], type_):
                    message = (
                        "Expected {} in data item to be of type {}, but got {}".format(
                            key, type_, type(data_item[key])
                        )
                    )
                    raise DataValidationError(message, data_item_number)

        elif isinstance(data_item, (list, tuple)):
            # From iterable: error on too many items, too few items
            if len(self._expected_keys) != len(data_item):
                message = "Expected data item to have {} elements but contained {} elements".format(
                    len(self._expected_keys), len(data_item)
                )
                raise DataValidationError(message, data_item_number)
            # Error if any element has wrong type
            for type_, element, index in zip(
                self._expected_keys.values(), data_item, range(len(data_item))
            ):
                if not isinstance(element, type_):
                    # pylint: disable=too-many-format-args
                    message = "Expected element {} in data item to be of type {}, "
                    "but got {}".format(index, type_, type(element))
                    raise DataValidationError(message, data_item_number)

        else:
            message = "This data item of type {} cannot be converted :(".format(
                type(data_item)
            )
            log.error("<COR76665827E>", message)

            raise DataValidationError(message, data_item_number)
