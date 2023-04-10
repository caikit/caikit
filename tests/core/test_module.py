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
import abc
import io
import os
import tempfile
import uuid

# Third Party
# pylint: disable=import-error
from sample_lib.blocks.sample_task import SampleBlock
from sample_lib.data_model.sample import SampleInputType
import pytest

# First Party
import aconfig

# Local
from caikit.core import ModuleConfig, module
from caikit.core.module_backend_config import get_backend
from caikit.core.module_backends import backend_types

# Unit Test Infrastructure
from tests.base import TestCaseBase

# NOTE: We do need to import `reset_backend_types` and `reset_module_distribution_registry` for `reset_globals` to work
from tests.core.helpers import *
import caikit.core


class TestModuleBase(TestCaseBase):
    def setUp(self):
        self.base_module_instance = caikit.core.ModuleBase()

    def test_load_evaluation_dataset(self):
        self.assertIsInstance(
            module.ModuleBase.load_evaluation_dataset(
                os.path.join(self.fixtures_dir, "dummy_dataset.json")
            ),
            list,
        )

    def test_init_available(self):
        model = caikit.core.ModuleBase([0, 1, 2], kw1=0, kw2=1, kw3=2)
        self.assertIsInstance(model, caikit.core.ModuleBase)

    def test_load_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            caikit.core.ModuleBase.load()

    def test_run_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.base_module_instance.run()

    def test_save_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.base_module_instance.save("dummy_path")

    def test_train_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            caikit.core.BlockBase.train()


class TestModuleConfig(TestCaseBase):
    def setUp(self):
        self.model_path = os.path.join(self.fixtures_dir, "dummy_block")

        self.model_config = ModuleConfig.load(self.model_path)

    def test_isinstance(self):
        self.assertIsInstance(self.model_config, ModuleConfig)
        self.assertIsInstance(self.model_config, aconfig.Config)

    def test_init_and_members(self):
        config = ModuleConfig(
            {
                "block_id": "123",
                "block_class": "caikit.core.blocks.dummy.Dummy",
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

        self.assertEqual(config.block_id, "123")
        self.assertEqual(config.block_class, "caikit.core.blocks.dummy.Dummy")
        self.assertIsInstance(config.string, str)
        self.assertEqual(config.string, "hello")
        self.assertIsInstance(config.integer, int)
        self.assertEqual(config.integer, 1)
        self.assertIsInstance(config.float, float)
        self.assertAlmostEqual(config.float, 0.5)
        self.assertIsInstance(config.nested, dict)
        self.assertEqual(config.nested.string, "world")
        self.assertEqual(config.nested.integer, 2)
        self.assertAlmostEqual(config.nested.float, -0.123)

    def test_block_config_has_module_id(self):
        self.assertIsNotNone(self.model_config.module_id)
        self.assertEqual(self.model_config.module_id, self.model_config.block_id)

    def test_workflow_config_has_module_id(self):
        config = ModuleConfig(
            {
                "workflow_id": "123",
                "workflow_class": "caikit.core.workflows.dummy.Dummy",
                "name": "Dummy Workflow",
            }
        )

        self.assertIsNotNone(config.module_id)
        self.assertEqual(config.module_id, config.workflow_id)

    def test_resource_config_has_module_id(self):
        config = ModuleConfig(
            {
                "resource_id": "r123",
                "resource_class": "caikit.core.resources.dummy.Dummy",
                "name": "Dummy Resource",
            }
        )

        self.assertIsNotNone(config.module_id)
        self.assertEqual(config.module_id, config.resource_id)

    def test_reserved_keys(self):
        for reserved_key in ("module_id", "model_path"):
            with self.assertRaises(KeyError):
                ModuleConfig(
                    {
                        reserved_key: "x",
                        "block_class": "caikit.core.blocks.dummy.Dummy",
                        "name": "Dummy Workflow",
                    }
                )

    def test_no_config_yaml(self):
        with self.assertRaises(FileNotFoundError):
            with tempfile.TemporaryDirectory() as tempd:
                ModuleConfig.load(tempd)

    def test_model_path_is_file(self):
        with tempfile.TemporaryDirectory() as tempd:
            tempf = os.path.join(tempd, "junk")

            with open(tempf, mode="w", encoding="utf-8") as fh:
                fh.write("\n")

            with self.assertRaises(FileNotFoundError):
                ModuleConfig.load(tempf)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tempd:
            self.model_config.save(tempd)
            ModuleConfig.load(tempd)


class TestModelConversionOperations(TestCaseBase):
    def setUp(self):
        self.dummy_model = os.path.join(self.fixtures_dir, "dummy_block")

    ###############################################################################################
    #                        Tests for interacting with file-like objects                         #
    ###############################################################################################
    def test_model_file_like_conversion(self):
        """Test that we can convert a model into a file-like object."""
        dummy_model = caikit.core.load(self.dummy_model)
        file_like = dummy_model.as_file_like_object()
        self.assertIsInstance(file_like, io.BytesIO)

    def test_model_byte_conversion(self):
        """Test that we can convert a model into bytes."""
        dummy_model = caikit.core.load(self.dummy_model)
        bytes_like = dummy_model.as_bytes()
        self.assertIsInstance(bytes_like, bytes)

    def test_load_file_like_conversion_and_back(self):
        """Test that we can load a model, export to a file-like object, and reload the model."""
        dummy_model = caikit.core.load(self.dummy_model)
        file_like = dummy_model.as_file_like_object()
        reloaded_model = caikit.core.load(file_like)
        self.assertIsInstance(reloaded_model, caikit.core.BlockBase)

    def test_load_bytes_conversion_and_back(self):
        """Test that we can load a model, export to a bytes object, and reload the model."""
        dummy_model = caikit.core.load(self.dummy_model)
        bytes_like = dummy_model.as_bytes()
        reloaded_model = caikit.core.load(bytes_like)
        self.assertIsInstance(reloaded_model, caikit.core.BlockBase)


class TestModuleTypeDecorator(TestCaseBase):
    @classmethod
    def tearDownClass(cls):
        if "TESTMOD" in module._MODULE_TYPES:
            module._MODULE_TYPES.remove("TESTMOD")
        if hasattr(caikit.core, "TESTMOD_REGISTRY"):
            delattr(caikit.core, "TESTMOD_REGISTRY")

    @module.module_type("testmod")
    class TestModBase(module.ModuleBase):
        """Derived module type"""

    ######################### Tests #########################

    def test_module_type_new_type(self):
        """Make sure that the newly declared module type can be used just like a
        first-class module type such as block
        """
        assert hasattr(caikit.core, "TESTMOD_REGISTRY")
        assert "TESTMOD" in module._MODULE_TYPES
        assert caikit.core.TESTMOD_REGISTRY == {}

        # Add a new derived testmod
        mod_id = str(uuid.uuid4())

        @self.TestModBase.testmod(id=mod_id, name="Sample tesmod", version="1.2.3")
        class SampleTestmod(self.TestModBase):
            """A sample test mod"""

        # Make sure that the test mod was added to the registry
        assert caikit.core.TESTMOD_REGISTRY == {mod_id: SampleTestmod}

        # Make sure module type is set as attribute of sample module
        assert hasattr(SampleTestmod, "MODULE_TYPE")
        assert SampleTestmod.MODULE_TYPE == "TESTMOD"

    def test_module_type_missing_base_class(self):
        """Make sure that if a derived class misses the inheritance from the
        right base class, an exception is raised
        """
        mod_id = str(uuid.uuid4())
        with pytest.raises(TypeError):
            # pylint: disable=unused-variable
            @self.TestModBase.testmod(id=mod_id, name="Sample tesmod", version="1.2.3")
            class SampleBadTestmod:
                """A sample test mod that is missing the base class"""

    def test_module_type_wrong_base_class(self):
        """Make sure that if a derived class inherits from the wrong module type
        an exception is raised
        """
        mod_id = str(uuid.uuid4())
        with pytest.raises(TypeError):
            # pylint: disable=unused-variable
            @self.TestModBase.testmod(id=mod_id, name="Sample tesmod", version="1.2.3")
            class SampleBadTestmod(caikit.core.BlockBase):
                """A sample test mod that is missing the base class"""

    def test_module_no_reused_ids(self):
        """Make sure that module ids can't be reused, even across module types"""
        with pytest.raises(RuntimeError):

            @self.TestModBase.testmod(
                id=SampleBlock.MODULE_ID,
                name="Sample tesmod",
                version="1.2.3",
            )
            # pylint: disable=unused-variable
            class SampleTestmod(self.TestModBase):
                """A sample test mod"""

    def test_intermediate_metabase(self):
        """Make sure that an abc.ABC can be declared that derives from ModuleBase"""

        class Intermediate(caikit.core.blocks.base.BlockBase, abc.ABC):
            """Sample intermediate base class"""

            @abc.abstractmethod
            def foobar(self, arg):
                """Got to define foobar!"""

        @caikit.core.block("asdf-qwer-zxcv", "FooBar", "0.0.1")
        class Derived(Intermediate):
            def foobar(self, arg):
                return arg + 1

        d = Derived()
        assert d.foobar(1) == 2


class TestModuleDefaultFunctionality(TestCaseBase):
    """Tests for default functionality (e.g. run_batch)"""

    def setUp(self):
        self.dummy_model = os.path.join(self.fixtures_dir, "dummy_block")

    def test_run_batch_keyword_only(self):
        """Make sure that calling run_batch without any positional args is safe"""
        dummy_model = caikit.core.load(self.dummy_model)
        dummy_model.run(sample_input=SampleInputType(name="Gabe"))
        dummy_model.run_batch(sample_input=[SampleInputType(name="Gabe")])


## Pytest tests ######################################

DUMMY_MODULE_ID = "foo"


def configure_alternate_backend_impl():
    """Function to register a new backend type and register a module implementation
    of existing caikit.core module"""

    @caikit.core.blocks.block(id=DUMMY_MODULE_ID, name="dummy base", version="0.0.1")
    class DummyFoo(caikit.core.blocks.base.BlockBase):
        pass

    # Register backend type
    backend_types.register_backend_type(MockBackend)

    @caikit.core.blocks.block(base_module=DummyFoo, backend_type=backend_types.MOCK)
    class DummyBar:
        def test_fetching_backend(self):
            return get_backend(backend_types.MOCK)

    return DummyFoo, DummyBar


### Tests #######################################################################


def test_can_get_module_id(reset_globals):
    """Test we can get registered module from registry"""
    _, DummyBar = configure_alternate_backend_impl()
    assert (
        caikit.core.MODULE_BACKEND_REGISTRY.get(DUMMY_MODULE_ID)
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
    assert DUMMY_MODULE_ID in MODULE_BACKEND_REGISTRY
    assert MockBackend.backend_type in MODULE_BACKEND_REGISTRY[DUMMY_MODULE_ID]


def test_duplicate_registration_raises(reset_globals):
    """Test duplicate registration of same backend for same module raises"""
    DummyFoo, _ = configure_alternate_backend_impl()
    with pytest.raises(AssertionError):

        @caikit.core.blocks.block(base_module=DummyFoo, backend_type=backend_types.MOCK)
        class DummyBat:
            pass


def test_backend_impl_inheritance_error(reset_globals):
    """Test creating a backend implementation of a module
    which inherits from base module raises"""
    DummyFoo, DummyBar = configure_alternate_backend_impl()
    with pytest.raises(TypeError):

        @caikit.core.blocks.block(backend_type=backend_types.MOCK)
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
    assert DummyBar.BLOCK_ID == DummyFoo.BLOCK_ID
    assert DummyBar.BLOCK_NAME
    assert DummyBar.BLOCK_VERSION
    assert DummyBar.BLOCK_CLASS
    assert DummyBar.PRODUCER_ID

    # Fetch without config raises
    with pytest.raises(ValueError):
        assert get_backend(DummyBar.BACKEND_TYPE)

    # Configure and make sure it can be fetched by the class
    caikit.core.module_backend_config.configure(backend_priority=[backend_types.MOCK])
    assert DummyBar.BACKEND_TYPE == backend_types.MOCK

    # Make sure an instance can fetch via get_backend()
    inst = DummyBar()
    assert inst.test_fetching_backend().backend_type == backend_types.MOCK


def test_default_load_supported_backend(reset_globals):
    """Test the defined backend gets added as the supported backend by default"""
    _, DummyBar = configure_alternate_backend_impl()
    assert hasattr(DummyBar, module.SUPPORTED_LOAD_BACKENDS_VAR_NAME)
    assert len(DummyBar.SUPPORTED_LOAD_BACKENDS) == 1
    assert DummyBar.SUPPORTED_LOAD_BACKENDS[0] == backend_types.MOCK


def test_override_load_supported_backend(reset_globals):
    """Test if the class can successfully define its own backends
    that it supports load from"""

    @caikit.core.blocks.block(id=DUMMY_MODULE_ID, name="dummy base", version="0.0.1")
    class DummyFoo(caikit.core.blocks.base.BlockBase):
        pass

    class BazBackend(BackendBase):
        backend_type = "BAZ"

    class FooBackend(BackendBase):
        backend_type = "FOO"

    backend_types.register_backend_type(BazBackend)
    backend_types.register_backend_type(FooBackend)

    supported_backends = [backend_types.BAZ, backend_types.FOO]

    @caikit.core.blocks.block(base_module=DummyFoo, backend_type=backend_types.BAZ)
    class DummyBaz:
        SUPPORTED_LOAD_BACKENDS = supported_backends

    assert hasattr(DummyBaz, module.SUPPORTED_LOAD_BACKENDS_VAR_NAME)
    assert DummyBaz.SUPPORTED_LOAD_BACKENDS == supported_backends


def test_base_module_in_decorator(reset_globals):
    """Test multiple backend implementation of a module doesn't override base module
    from MODULE_REGISTRY"""

    class BazBackend(BackendBase):
        backend_type = "BAZ"

    class FooBackend(BackendBase):
        backend_type = "FOO"

    backend_types.register_backend_type(BazBackend)
    backend_types.register_backend_type(FooBackend)

    @caikit.core.blocks.block(id=DUMMY_MODULE_ID, name="dummy base", version="0.0.1")
    class DummyLocal(caikit.core.blocks.base.BlockBase):
        pass

    @caikit.core.blocks.block(backend_type=backend_types.BAZ, base_module=DummyLocal)
    class DummyBaz(caikit.core.ModuleBase):
        pass

    @caikit.core.blocks.block(
        backend_type=backend_types.FOO, base_module=DummyLocal.MODULE_ID
    )
    class DummyFoo(caikit.core.ModuleBase):
        pass

    assert DummyLocal in list(caikit.core.MODULE_REGISTRY.values())
