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

"""
Tests for backend type logic
"""

# Standard

# Third Party
import pytest

# Local
from caikit.core import backend_types
from caikit.core.module_backends.base import BackendBase
from tests.core.helpers import MockBackend, reset_backend_types

## Tests #######################################################################


def test_type_attr_forwarding():
    """Test that the LOCAL type behaves as expected"""
    assert backend_types.LOCAL == "LOCAL"


def test_register_backend_type_with_config(reset_backend_types):
    """Test that a backend type can be registered with a callable"""

    backend_types.register_backend_type(MockBackend)
    assert backend_types.MOCK == "MOCK"


def test_non_registered_backend_type_raises():
    """Test that non registered backend type raises error"""
    with pytest.raises(ValueError):
        backend_types.DUMMY


def test_register_backend_type_uppercase_required():
    """Make sure that the type name is required to be uppercase"""

    class DummyBackend(BackendBase):
        backend_type = "foo"

        def start(self):
            pass

        def register_config(self, config=...):
            pass

    with pytest.raises(ValueError):
        backend_types.register_backend_type(DummyBackend)


def test_register_backend_type_callable_required():
    """Make sure that the configuration function must be a callable"""
    with pytest.raises(TypeError):
        backend_types.register_backend_type("FOO")


def test_incorrect_backend_type_raises():
    """Test that if backend is not subclassed from BackendBase then registration
    raises error"""

    class MockBackend:
        pass

    with pytest.raises(TypeError):
        backend_types.register_backend_type(MockBackend)
