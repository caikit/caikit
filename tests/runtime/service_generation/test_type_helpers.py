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
from typing import List, Optional, Union

# Local
from caikit.core.data_model import DataStream
from caikit.runtime.service_generation.type_helpers import (
    get_data_stream_type,
    has_data_stream,
    is_model_type,
)
import sample_lib


def test_data_stream_helpers():
    assert has_data_stream(DataStream[int])
    assert has_data_stream(Optional[DataStream[int]])
    assert has_data_stream(Union[List[DataStream[int]], str])

    assert get_data_stream_type(Optional[DataStream[int]]) == DataStream[int]
    assert get_data_stream_type(Union[List[DataStream[str]], int]) == DataStream[str]


def test_model_type_helpers():
    assert is_model_type(sample_lib.modules.sample_task.SampleModule)
    assert is_model_type(Union[str, sample_lib.modules.sample_task.SampleModule])
