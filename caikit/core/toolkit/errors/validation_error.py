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


class DataValidationError(Exception):
    """This error is used for data validation problems during training"""

    def __init__(self, reason, item_number=None):
        if item_number:
            message = "Training data validation failed on item {}. {}".format(
                item_number, reason
            )
        else:
            message = "Training data validation failed: {}".format(reason)
        super().__init__(message)
        self._reason = reason
        self._item_number = item_number

    @property
    def reason(self) -> str:
        """The reason given for this data validation error"""
        return self._reason

    @property
    def item_number(self) -> int:
        """The index of the training data item that failed validation. Probably zero indexed"""
        return self._item_number
