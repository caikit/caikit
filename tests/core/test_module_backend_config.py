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
from caikit.core.blocks import base, block
from caikit.core.module_backend_config import (
    configure,
    configured_load_backends,
    configured_train_backends,
    get_load_backend,
    get_train_backend,
    start_backends,
)
from caikit.core.module_backends import backend_types
from sample_lib.data_model import SampleTask
from tests.conftest import temp_config
from tests.core.helpers import *

# Setup #########################################################################

foo_cfg = {"mock": 1}

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
        with pytest.raises(ValueError):
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


def test_disabling_local_backend(reset_globals):
    """Test that disabling local backend does not add it to priority automatically"""
    with temp_config(
        {
            "module_backends": {
                "disable_local": True,
                "load_priority": [{"type": backend_types.MOCK}],
                "train_priority": [{"type": backend_types.MOCK}],
            }
        }
    ):
        configure()
        assert get_load_backend(backend_types.MOCK)
        assert get_train_backend(backend_types.MOCK)
        with pytest.raises(ValueError):
            get_load_backend(backend_types.LOCAL)
        with pytest.raises(ValueError):
            get_train_backend(backend_types.LOCAL)


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


def test_duplicate_implied_names_raise(reset_globals):
    """Test that duplicate entries with the same name implied from type raises"""
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
        with pytest.raises(ValueError):
            configure()


def test_duplicate_explicit_names_raise(reset_globals):
    """Test that duplicate entries with the same name given explicitly from
    raises
    """
    with temp_config(
        {
            "module_backends": {
                "load_priority": [
                    {"type": backend_types.MOCK, "name": "foo"},
                    {"type": backend_types.MOCK, "name": "foo"},
                ],
            }
        }
    ):
        with pytest.raises(ValueError):
            configure()


def test_duplicate_type_name_disambig(reset_globals):
    """Test that multiple instances of the same type can be configured with
    different names
    """
    with temp_config(
        {
            "module_backends": {
                "load_priority": [
                    {"type": backend_types.MOCK},
                    {"type": backend_types.MOCK, "name": "foo"},
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

    @block(id="foo", name="dummy base", version="0.0.1", task=SampleTask)
    class DummyFoo(base.BlockBase):
        pass

    # Create dummy classes
    @block(
        base_module=DummyFoo,
        backend_type=backend_types.MOCK,
        backend_config_override={"bar1": 1},
    )
    class DummyBar(base.BlockBase):
        pass

    @block(id="foo2", name="dummy base", version="0.0.1", task=SampleTask)
    class DummyFoo2(base.BlockBase):
        pass

    # Create dummy classes
    @block(
        base_module=DummyFoo2,
        backend_type=backend_types.MOCK,
        backend_config_override={"bar2": 2},
    )
    class DummyBar(base.BlockBase):
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


def test_get_backend_starts_backend(reset_globals):
    """Test that fetching a handle to a backend with get_backend ensures that it
    is started
    """
    with temp_config(
        {"module_backends": {"train_priority": [{"type": backend_types.MOCK}]}}
    ):
        configure()
        mock_train_backend = get_train_backend(backend_types.MOCK)
        assert mock_train_backend.is_started
