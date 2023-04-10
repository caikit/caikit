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
Tests for the backend configuration framework
"""

# Standard
from unittest.mock import Mock

# Local
from caikit.core.blocks import base, block
from caikit.core.module_backend_config import (
    _CONFIGURED_BACKENDS,
    configure,
    configured_backends,
    get_backend,
    start_backends,
)
from caikit.core.module_backends import backend_types
from tests.core.helpers import *

# Setup #########################################################################

foo_cfg = {"mock": 1}

## Tests #######################################################################


def test_configure_with_module(reset_globals):
    """Test that configuring with a configured bakend type that has a
    configuration module obj works
    """
    backend_types.register_backend_type(MockBackend)
    configure(backend_priority=[backend_types.MOCK], backends={"mock": foo_cfg})
    assert "MOCK" in configured_backends()
    assert _CONFIGURED_BACKENDS[backend_types.MOCK].backend_type == backend_types.MOCK
    assert foo_cfg == _CONFIGURED_BACKENDS[backend_types.MOCK].config


def test_non_supported_backend_raises():
    """Test that backend provided as priority to configure raises
    if not registered"""
    with pytest.raises(ValueError):
        configure(backend_priority=[Mock()])


def test_disabling_local_backend(reset_globals):
    """Test that disabling local backend does not add it to priority automatically"""
    backend_types.register_backend_type(MockBackend)
    configure(backend_priority=[backend_types.MOCK], disable_local_backend=True)
    assert "LOCAL" not in configured_backends()


def test_duplicate_config_raises(reset_globals):
    """Test that duplicate configuration of a backend raises"""
    backend_types.register_backend_type(MockBackend)
    configure(backend_priority=[backend_types.MOCK])
    with pytest.raises(AssertionError):
        configure(backend_priority=[backend_types.MOCK])


def test_one_configured_backend_can_start(reset_globals):
    """Test that the configured backend can be started"""
    backend_types.register_backend_type(MockBackend)
    configure(backend_priority=[backend_types.MOCK], backends={"mock": foo_cfg})
    start_backends()
    # This is configured to be True in helpers
    assert _CONFIGURED_BACKENDS[backend_types.MOCK].backend_type == backend_types.MOCK
    assert _CONFIGURED_BACKENDS[backend_types.MOCK].is_started


def test_multiple_module_same_backend_configures(reset_globals):
    """Test to check if multiple modules for same backend
    can override backend configurations"""
    # Register backend type
    backend_types.register_backend_type(MockBackend)

    @block(id="foo", name="dummy base", version="0.0.1")
    class DummyFoo(base.BlockBase):
        pass

    # Create dummy classes
    @block(
        base_module=DummyFoo,
        backend_type=backend_types.MOCK,
        backend_config_override={"bar1": 1},
    )
    class DummyBar:
        pass

    @block(id="foo2", name="dummy base", version="0.0.1")
    class DummyFoo2(base.BlockBase):
        pass

    # Create dummy classes
    @block(
        base_module=DummyFoo2,
        backend_type=backend_types.MOCK,
        backend_config_override={"bar2": 2},
    )
    class DummyBar:
        pass

    # Initiate configuration

    configure(backend_priority=[backend_types.MOCK])
    assert "MOCK" in configured_backends()
    assert _CONFIGURED_BACKENDS[backend_types.MOCK].backend_type == backend_types.MOCK
    assert "bar1" in _CONFIGURED_BACKENDS[backend_types.MOCK].config
    assert "bar2" in _CONFIGURED_BACKENDS[backend_types.MOCK].config
    assert _CONFIGURED_BACKENDS[backend_types.MOCK].config["bar1"] == 1


def test_get_backend_starts_backend(reset_globals):
    """Test that fetching a handle to a backend with get_backend ensures that it
    is started
    """
    backend_types.register_backend_type(MockBackend)
    configure(backend_priority=[backend_types.MOCK], disable_local_backend=True)
    assert not _CONFIGURED_BACKENDS[backend_types.MOCK].is_started
    backend = get_backend(backend_types.MOCK)
    assert backend.is_started
    assert _CONFIGURED_BACKENDS[backend_types.MOCK].is_started
