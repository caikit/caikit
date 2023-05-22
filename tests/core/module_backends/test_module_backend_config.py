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

# Local
from caikit.core.module_backends.module_backend_config import (
    configure,
    configured_load_backends,
    configured_train_backends,
    start_backends,
)
from caikit.core.modules import base, module
from sample_lib.data_model import SampleTask
from tests.conftest import temp_config
from tests.core.helpers import *

# Setup #########################################################################

foo_cfg = {"mock": 1}


def _get_backend(backend_type, backend_list):
    """Get the single configured backend of the given type. An error is raised
    if the number of matches != 1
    """
    matches = [
        backend for backend in backend_list if backend.backend_type == backend_type
    ]
    assert len(matches) == 1
    return matches[0]


def get_load_backend(backend_type):
    return _get_backend(backend_type, configured_load_backends())


def get_train_backend(backend_type):
    return _get_backend(backend_type, configured_train_backends())


## Tests #######################################################################


def test_configure_with_module(reset_globals):
    """Test that configuring with a configured backend type that has a
    configuration module obj works
    """
    with temp_config(
        {
            "module_backends": {
                "load_priority": [
                    {
                        "type": backend_types.MOCK,
                        "config": foo_cfg,
                    }
                ],
                "train_priority": [
                    {
                        "type": backend_types.MOCK,
                        "config": foo_cfg,
                    }
                ],
            }
        }
    ):
        configure()

        # Test load backend config
        mock_load_backend = get_load_backend(backend_types.MOCK)
        assert mock_load_backend.backend_type == backend_types.MOCK
        assert foo_cfg == mock_load_backend.config

        # Test train backend config
        mock_train_backend = get_train_backend(backend_types.MOCK)
        assert mock_train_backend.backend_type == backend_types.MOCK
        assert foo_cfg == mock_train_backend.config


def test_configure_load_only(reset_globals):
    """Test that train and load backends can be configured independently"""
    with temp_config(
        {
            "module_backends": {
                "load_priority": [
                    {
                        "type": backend_types.MOCK,
                        "config": foo_cfg,
                    }
                ],
                "train_priority": [],
            }
        }
    ):
        configure()

        # Test load backend config
        mock_load_backend = get_load_backend(backend_types.MOCK)
        assert mock_load_backend.backend_type == backend_types.MOCK
        assert foo_cfg == mock_load_backend.config

        # Test train backend config
        with pytest.raises(AssertionError):
            get_train_backend(backend_types.MOCK)


def test_non_supported_backend_raises():
    """Test that backend provided as priority to configure raises
    if not registered"""
    with temp_config(
        {
            "module_backends": {
                "load_priority": [{"type": "unsupported"}],
            }
        }
    ):
        with pytest.raises(ValueError):
            configure()


def test_duplicate_config_raises(reset_globals):
    """Test that duplicate configuration of a backend raises"""
    with temp_config(
        {
            "module_backends": {
                "load_priority": [
                    {"type": backend_types.MOCK},
                ],
            }
        }
    ):
        # The mock backend is already registered for tests in the core
        # see tests/core/helpers.py
        configure()
        with pytest.raises(ValueError):
            configure()


def test_duplicate_explicit_instances_allowed(reset_globals):
    """Test that duplicate entries are allowed"""
    with temp_config(
        {
            "module_backends": {
                "load_priority": [
                    {"type": backend_types.MOCK},
                    {"type": backend_types.MOCK},
                ],
            }
        }
    ):
        configure()


def test_one_configured_backend_can_start(reset_globals):
    """Test that the configured backend can be started"""
    with temp_config(
        {
            "module_backends": {
                "load_priority": [
                    {
                        "type": backend_types.MOCK,
                        "config": foo_cfg,
                    }
                ],
            }
        }
    ):
        configure()
        start_backends()

        # This is configured to be True in helpers
        mock_load_backend = get_load_backend(backend_types.MOCK)
        assert mock_load_backend.backend_type == backend_types.MOCK
        assert mock_load_backend.is_started


def test_multiple_module_same_backend_configures(reset_globals):
    """Test to check if multiple modules for same backend
    can override backend configurations"""
    # Register backend type

    @module(id="foo", name="dummy base", version="0.0.1", task=SampleTask)
    class DummyFoo(base.ModuleBase):
        pass

    # Create dummy classes
    @module(
        base_module=DummyFoo,
        backend_type=backend_types.MOCK,
        backend_config_override={"bar1": 1},
    )
    class DummyBar(base.ModuleBase):
        pass

    @module(id="foo2", name="dummy base", version="0.0.1", task=SampleTask)
    class DummyFoo2(base.ModuleBase):
        pass

    # Create dummy classes
    @module(
        base_module=DummyFoo2,
        backend_type=backend_types.MOCK,
        backend_config_override={"bar2": 2},
    )
    class DummyBar(base.ModuleBase):
        pass

    # Initiate configuration

    with temp_config(
        {
            "module_backends": {
                "load_priority": [{"type": backend_types.MOCK}],
            }
        }
    ):
        configure()
        mock_load_backend = get_load_backend(backend_types.MOCK)
        assert mock_load_backend.backend_type == backend_types.MOCK
        assert "bar1" in mock_load_backend.config
        assert "bar2" in mock_load_backend.config
        assert mock_load_backend.config["bar1"] == 1
