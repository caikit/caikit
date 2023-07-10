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
import io
import os
import tempfile

# First Party
import aconfig

# Local
from caikit.core import ModuleConfig, ModuleLoader
from caikit.core.modules.decorator import SUPPORTED_LOAD_BACKENDS_VAR_NAME

# pylint: disable=import-error
from sample_lib.data_model.sample import SampleInputType

# Unit Test Infrastructure
from tests.conftest import temp_config

# NOTE: We do need to import `reset_backend_types` and `reset_module_distribution_registry` for `reset_globals` to work
from tests.core.helpers import *
import caikit.core

## Helpers #####################################################################


@pytest.fixture
def base_module_instance():
    return caikit.core.ModuleBase()


@pytest.fixture
def model_path(fixtures_dir):
    yield os.path.join(fixtures_dir, "dummy_module")


@pytest.fixture
def model_config(model_path):
    yield ModuleConfig.load(model_path)


DUMMY_MODULE_ID = "foo"


def configure_alternate_backend_impl():
    """Function to register a new backend type and register a module implementation
    of existing caikit.core module"""

    @caikit.core.modules.module(id=DUMMY_MODULE_ID, name="dummy base", version="0.0.1")
    class DummyFoo(caikit.core.ModuleBase):
        pass

    # Register backend type
    backend_types.register_backend_type(MockBackend)

    @caikit.core.modules.module(base_module=DummyFoo, backend_type=backend_types.MOCK)
    class DummyBar(caikit.core.ModuleBase):
        def test_fetching_backend(self):
            return [
                backend
                for backend in configured_backends()
                if backend.backend_type == backend_types.MOCK
            ][0]

    return DummyFoo, DummyBar


## ModuleBase ##################################################################


def test_load_evaluation_dataset(fixtures_dir):
    assert isinstance(
        ModuleBase._load_evaluation_dataset(
            os.path.join(fixtures_dir, "dummy_dataset.json")
        ),
        list,
    )


def test_init_available():
    """Make sure that there is an __init__ that takes no args or kwargs"""
    model = caikit.core.ModuleBase()
    assert isinstance(model, caikit.core.ModuleBase)


def test_underscore_load_not_implemented():
    with pytest.raises(NotImplementedError):
        caikit.core.ModuleBase._load(None)


def test_load_delegates_to_underscore_load(good_model_path):
    @caikit.core.modules.module(id="blah", name="dummy base", version="0.0.1")
    class DummyFoo(caikit.core.ModuleBase):
        @classmethod
        def _load(cls, module_loader, **kwargs):
            assert isinstance(module_loader, ModuleLoader)
            return "foo"

    assert DummyFoo.load(good_model_path) == "foo"


def test_run_not_implemented(base_module_instance):
    with pytest.raises(NotImplementedError):
        base_module_instance.run()


def test_save_not_implemented(base_module_instance):
    with pytest.raises(NotImplementedError):
        base_module_instance.save("dummy_path")


def test_train_not_implemented():
    with pytest.raises(NotImplementedError):
        caikit.core.ModuleBase.train()


def test_run_batch_keyword_only(model_path):
    """Make sure that calling run_batch without any positional args is safe"""
    dummy_model = caikit.core.load(model_path)
    dummy_model.run(sample_input=SampleInputType(name="Gabe"))
    dummy_model.run_batch(sample_input=[SampleInputType(name="Gabe")])


## ModuleConfig ################################################################


def test_isinstance(model_config):
    assert isinstance(model_config, ModuleConfig)
    assert isinstance(model_config, aconfig.Config)


def test_init_and_members():
    config = ModuleConfig(
        {
            "module_id": "123",
            "module_class": "caikit.core.modules.dummy.Dummy",
            "name": "Dummy Block",
            "string": "hello",
            "integer": 1,
            "float": 0.5,
            "nested": {
                "string": "world",
                "integer": 2,
                "float": -0.123,
            },
        }
    )

    assert config.module_id == "123"
    assert config.module_class == "caikit.core.modules.dummy.Dummy"
    assert isinstance(config.string, str)
    assert config.string == "hello"
    assert isinstance(config.integer, int)
    assert config.integer == 1
    assert isinstance(config.float, float)
    assert config.float == 0.5
    assert isinstance(config.nested, dict)
    assert config.nested.string == "world"
    assert config.nested.integer == 2
    assert config.nested.float == -0.123


def test_backwards_compatibility_for_block():
    config = ModuleConfig({"block_id": "123"})
    assert config.module_id == "123"


def test_backwards_compatibility_for_workflow():
    config = ModuleConfig(
        {
            "workflow_id": "456",
        }
    )
    assert config.module_id == "456"


def test_backwards_compatibility_for_resource():
    config = ModuleConfig(
        {
            "resource_id": "789",
        }
    )
    assert config.module_id == "789"


def test_reserved_keys():
    for reserved_key in ("model_path",):
        with pytest.raises(KeyError):
            ModuleConfig(
                {
                    reserved_key: "x",
                    "module_class": "caikit.core.modules.dummy.Dummy",
                    "name": "Dummy Module",
                }
            )


def test_no_config_yaml():
    with pytest.raises(FileNotFoundError):
        with tempfile.TemporaryDirectory() as tempd:
            ModuleConfig.load(tempd)


def test_model_path_is_file():
    with tempfile.TemporaryDirectory() as tempd:
        tempf = os.path.join(tempd, "junk")

        with open(tempf, mode="w", encoding="utf-8") as fh:
            fh.write("\n")

        with pytest.raises(FileNotFoundError):
            ModuleConfig.load(tempf)


def test_save_and_load(model_config):
    with tempfile.TemporaryDirectory() as tempd:
        model_config.save(tempd)
        ModuleConfig.load(tempd)


## Model Conversion Operations #################################################
# Tests for interacting with file-like objects


def test_model_file_like_conversion(model_path):
    """Test that we can convert a model into a file-like object."""
    dummy_model = caikit.core.load(model_path)
    file_like = dummy_model.as_file_like_object()
    assert isinstance(file_like, io.BytesIO)


def test_model_byte_conversion(model_path):
    """Test that we can convert a model into bytes."""
    dummy_model = caikit.core.load(model_path)
    bytes_like = dummy_model.as_bytes()
    assert isinstance(bytes_like, bytes)


def test_load_file_like_conversion_and_back(model_path):
    """Test that we can load a model, export to a file-like object, and reload the model."""
    dummy_model = caikit.core.load(model_path)
    file_like = dummy_model.as_file_like_object()
    reloaded_model = caikit.core.load(file_like)
    assert isinstance(reloaded_model, caikit.core.ModuleBase)


def test_load_bytes_conversion_and_back(model_path):
    """Test that we can load a model, export to a bytes object, and reload the model."""
    dummy_model = caikit.core.load(model_path)
    bytes_like = dummy_model.as_bytes()
    reloaded_model = caikit.core.load(bytes_like)
    assert isinstance(reloaded_model, caikit.core.ModuleBase)


## Non-Local Backends ##########################################################


def test_can_get_module_id(reset_globals):
    """Test we can get registered module from registry"""
    _, DummyBar = configure_alternate_backend_impl()
    assert (
        module_backend_registry()
        .get(DUMMY_MODULE_ID)
        .get(MockBackend.backend_type)
        .impl_class
        == DummyBar
    )


def test_module_id_registration_for_backend(reset_globals):
    """Test module id is properly registered for backend implementation"""
    DummyFoo, DummyBar = configure_alternate_backend_impl()
    assert DummyBar.MODULE_ID == DummyFoo.MODULE_ID


def test_module_registry_for_backend_type(reset_globals):
    """Test backend is properly registered in the module registry"""
    _ = configure_alternate_backend_impl()
    assert DUMMY_MODULE_ID in module_backend_registry()
    assert MockBackend.backend_type in module_backend_registry()[DUMMY_MODULE_ID]


def test_duplicate_registration_raises(reset_globals):
    """Test duplicate registration of same backend for same module raises"""
    DummyFoo, _ = configure_alternate_backend_impl()
    with pytest.raises(AssertionError):

        @caikit.core.modules.module(
            base_module=DummyFoo, backend_type=backend_types.MOCK
        )
        class DummyBat(caikit.core.ModuleBase):
            pass


def test_backend_impl_inheritance_error(reset_globals):
    """Test creating a backend implementation of a module
    which inherits from base module raises"""
    DummyFoo, DummyBar = configure_alternate_backend_impl()
    with pytest.raises(TypeError):

        @caikit.core.modules.module(backend_type=backend_types.MOCK)
        class DummyBat(DummyFoo):
            pass


def test_class_attributes(reset_globals):
    """Test that the wrapped class has the correct attributes set"""
    DummyFoo, DummyBar = configure_alternate_backend_impl()

    # Make sure all of the right class attrs are set
    assert DummyBar.BACKEND_TYPE == backend_types.MOCK
    assert DummyBar.MODULE_ID == DummyFoo.MODULE_ID
    assert DummyBar.MODULE_NAME
    assert DummyBar.MODULE_VERSION
    assert DummyBar.MODULE_CLASS
    assert DummyBar.PRODUCER_ID

    # Configure and make sure it can be fetched by the class
    with temp_config(
        {
            "model_management": {
                "initializers": {
                    "default": {
                        "type": "LOCAL",
                        "config": {
                            "backend_priority": [
                                {"type": backend_types.MOCK},
                            ]
                        },
                    },
                }
            }
        }
    ):
        assert DummyBar.BACKEND_TYPE == backend_types.MOCK

        # Normally non-local backend modules are loaded through caikit.load, so
        # we need to make sure the local loader has been properly configured to
        # enable local instance instantiation.
        assert MODEL_MANAGER._get_initializer("default")

        # Make sure an instance can fetch via get_backend()
        inst = DummyBar()
        assert inst.test_fetching_backend().backend_type == backend_types.MOCK


def test_default_load_supported_backend(reset_globals):
    """Test the defined backend gets added as the supported backend by default"""
    _, DummyBar = configure_alternate_backend_impl()
    assert hasattr(DummyBar, SUPPORTED_LOAD_BACKENDS_VAR_NAME)
    assert len(DummyBar.SUPPORTED_LOAD_BACKENDS) == 1
    assert DummyBar.SUPPORTED_LOAD_BACKENDS[0] == backend_types.MOCK


def test_override_load_supported_backend(reset_globals):
    """Test if the class can successfully define its own backends
    that it supports load from"""

    @caikit.core.modules.module(id=DUMMY_MODULE_ID, name="dummy base", version="0.0.1")
    class DummyFoo(caikit.core.ModuleBase):
        pass

    class BazBackend(BackendBase):
        backend_type = "BAZ"

    class FooBackend(BackendBase):
        backend_type = "FOO"

    backend_types.register_backend_type(BazBackend)
    backend_types.register_backend_type(FooBackend)

    supported_backends = [backend_types.BAZ, backend_types.FOO]

    @caikit.core.modules.module(base_module=DummyFoo, backend_type=backend_types.BAZ)
    class DummyBaz(caikit.core.ModuleBase):
        SUPPORTED_LOAD_BACKENDS = supported_backends

    assert hasattr(DummyBaz, SUPPORTED_LOAD_BACKENDS_VAR_NAME)
    assert DummyBaz.SUPPORTED_LOAD_BACKENDS == supported_backends


def test_base_module_in_decorator(reset_globals):
    """Test multiple backend implementation of a module doesn't override base module
    from module registry"""

    class BazBackend(BackendBase):
        backend_type = "BAZ"

    class FooBackend(BackendBase):
        backend_type = "FOO"

    backend_types.register_backend_type(BazBackend)
    backend_types.register_backend_type(FooBackend)

    @caikit.core.modules.module(id=DUMMY_MODULE_ID, name="dummy base", version="0.0.1")
    class DummyLocal(caikit.core.ModuleBase):
        pass

    @caikit.core.modules.module(backend_type=backend_types.BAZ, base_module=DummyLocal)
    class DummyBaz(caikit.core.ModuleBase):
        pass

    @caikit.core.modules.module(
        backend_type=backend_types.FOO, base_module=DummyLocal.MODULE_ID
    )
    class DummyFoo(caikit.core.ModuleBase):
        pass

    assert DummyLocal in list(module_registry().values())
