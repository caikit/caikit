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
"""Tests for default functionality in the BackendBase
"""

# Third Party
import pytest

# Local
from caikit.core.module_backends.base import BackendBase
from tests.core.helpers import MockBackend


def test_backend_base_is_abstract():
    """Make sure the class is abstract and can't be instantiated with missing
    implementations
    """

    class IntermediateBase(BackendBase):
        def register_config(self, config):
            pass

    with pytest.raises(TypeError):
        IntermediateBase()


def test_handle_runtime_context():
    """Make sure the handle_runtime_context implementation does nothing by
    default, but can be overridden
    """

    class Derived(BackendBase):
        backend_type = "TEST_DERIVED"

        def register_config(self, config):
            pass

        def start(self):
            pass

        def stop(self):
            pass

    # No-op default implementation
    model_id = "foo"
    ctx = "dummy context"
    be1 = Derived()
    be1.handle_runtime_context(model_id, ctx)

    # Derived with real implementation
    be2 = MockBackend()
    be2.handle_runtime_context(model_id, ctx)
    assert be2.runtime_contexts[model_id] is ctx
