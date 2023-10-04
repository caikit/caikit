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
import pickle

# Third Party
import grpc

# Local
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException


def test_caikit_runtime_exception_initialization():
    """Test that we can construct a CaikitRuntimeException with the right args"""

    caikit_runtime_exception = CaikitRuntimeException(
        status_code=grpc.StatusCode.INTERNAL,
        message="This is a test message",
        metadata={"key1": "val1"},
    )

    assert isinstance(caikit_runtime_exception, Exception)
    assert caikit_runtime_exception.id
    assert caikit_runtime_exception.message == "This is a test message"
    assert caikit_runtime_exception.metadata == {
        "key1": "val1",
        "error_id": caikit_runtime_exception.id,
    }


def test_caikit_runtime_exception_is_pickleable():
    """This is to ensure we can send caikit_runtime_exception through sub-processes"""

    caikit_runtime_exception = CaikitRuntimeException(
        status_code=grpc.StatusCode.INTERNAL, message="This is a test message"
    )

    caikit_runtime_exception_pickle = pickle.dumps(caikit_runtime_exception)
    caikit_runtime_exception_loaded = pickle.loads(caikit_runtime_exception_pickle)

    assert isinstance(caikit_runtime_exception_loaded, CaikitRuntimeException)
    assert caikit_runtime_exception_loaded.message == caikit_runtime_exception.message
