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
import os

# Local
from . import json_test_data, text_test_data


def test_data_iter(directory):
    if os.path.isdir(directory):
        return text_test_data.TextDirIterator(directory)

    if directory.endswith(".json"):
        return json_test_data.JsonDataIterator(directory)

    raise ValueError(
        "Incorrect test directory. Must be either a directory or a .json file "
        "with `text` attribute"
    )
