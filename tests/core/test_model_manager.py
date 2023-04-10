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
"""Tests for the model manager, which corrals catalogs to resolve models from names,
and download and load them.
"""

# Standard
from contextlib import contextmanager
import os
import tempfile

# Third Party
import pytest

# Local
from caikit.core.config import lib_config
from caikit.core.module_backend_config import configure, configured_backends
from caikit.core.module_backends import LocalBackend, backend_types

# Unit Test Infrastructure
from tests.base import TestCaseBase

# NOTE: We do need to import `reset_backend_types` and `reset_module_distribution_registry` for `reset_globals` to work
from tests.core.helpers import *
import caikit.core


class TestModelManager(TestCaseBase):
    """This test class tries to mock out the catalogs and direct download calls to reduce network
    overhead.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Test fixtures that can be directly loaded
        cls.model_path = os.path.join(cls.fixtures_dir, "dummy_block")
        # This model has no unique hash set in its config; we use it as a nonsingleton model too
        cls.non_singleton_model_path = cls.model_path
        cls.singleton_model_path = os.path.join(
            cls.fixtures_dir, "dummy_block_singleton"
        )

        cls.resource_path = os.path.join(cls.fixtures_dir, "dummy_resource")

        # Binary buffers of zip archives, for mocking downloads
        cls.block_zip_path = os.path.join(cls.fixtures_dir, "dummy_block.zip")
        with open(cls.block_zip_path, "rb") as f:
            cls.block_archive_buffer = f.read()

        zipfile = os.path.join(cls.fixtures_dir, "dummy_workflow.zip")
        with open(zipfile, "rb") as f:
            cls.workflow_archive_buffer = f.read()

        zipfile = os.path.join(cls.fixtures_dir, "dummy_resource.zip")
        with open(zipfile, "rb") as f:
            cls.resource_archive_buffer = f.read()

    @pytest.fixture
    def global_load_path(self):
        """Set load_path prior to importing caikit.core."""
        # Set test load path
        test_load_path = os.path.join(self.fixtures_dir, "studio_models")

        # Save the original load path so that it can be undone
        std_load_path = lib_config.load_path

        # Overwrite the load path with the test fixtures path
        lib_config.load_path = test_load_path

        # Yield execution to the test itself
        yield

        # Undo the load path patching
        lib_config.load_path = std_load_path

    def test_load_can_return_a_block(self):
        model = caikit.core.load(self.model_path)
        self.assertIsInstance(model, caikit.core.BlockBase)

    def test_load_can_load_a_block_as_a_singleton(self):
        # load a model with no hash config
        no_hashed_model = caikit.core.load(
            self.non_singleton_model_path, load_singleton=True
        )
        self.assertIsInstance(no_hashed_model, caikit.core.BlockBase)

        model2 = caikit.core.load(self.singleton_model_path, load_singleton=True)
        model3 = caikit.core.load(self.singleton_model_path, load_singleton=True)

        # Pointer should be equal
        self.assertEqual(id(model2), id(model3))

        # Pointer should not be equal
        self.assertNotEqual(id(no_hashed_model), id(model3))

    def test_load_can_load_a_block_with_singleton_disabled(self):
        # load a model with no hash config
        no_hashed_model = caikit.core.load(
            self.non_singleton_model_path, load_singleton=True
        )
        self.assertIsInstance(no_hashed_model, caikit.core.BlockBase)

        model2 = caikit.core.load(self.singleton_model_path, load_singleton=False)
        model3 = caikit.core.load(self.singleton_model_path, load_singleton=False)

        # Pointer should not be equal
        self.assertNotEqual(id(model2), id(model3))
        self.assertNotEqual(id(no_hashed_model), id(model3))

    def test_singleton_cache_can_be_cleared(self):
        model = caikit.core.load(self.singleton_model_path, load_singleton=True)
        self.assertGreater(
            len(caikit.core.MODEL_MANAGER.get_singleton_model_cache_info()), 0
        )

        caikit.core.MODEL_MANAGER.clear_singleton_cache()
        self.assertEqual(
            len(caikit.core.MODEL_MANAGER.get_singleton_model_cache_info()), 0
        )

        model2 = caikit.core.load(self.singleton_model_path, load_singleton=True)
        # Pointers should be different, since cache was cleared in between
        self.assertNotEqual(id(model), id(model2))

    def test_load_raises_on_bad_paths(self):
        with self.assertRaises(FileNotFoundError):
            caikit.core.load("bad/path/to/model")

    def test_extract(self):
        with tempfile.TemporaryDirectory() as tempdir:
            extract_path = caikit.core.extract(
                self.block_zip_path, tempdir, force_overwrite=True
            )
            self.assertEqual(extract_path, tempdir)
            self.assertTrue(os.path.isdir(extract_path))
            # load shouldn't barf
            caikit.core.load(extract_path)

    def test_resolve_and_load_can_load_a_model_from_a_path(self):
        """Check that we can resolve paths to module directories"""
        model = caikit.core.resolve_and_load(self.model_path)
        self.assertIsInstance(model, caikit.core.ModuleBase)

    def test_resolve_and_load_returns_back_a_model_reference(self):
        """Check that we can regurgitate things that are already Modules"""
        model = caikit.core.load(self.model_path)
        model = caikit.core.resolve_and_load(model)
        self.assertIsInstance(model, caikit.core.ModuleBase)

    def test_resolve_and_load_rejects_things_that_are_not_strings_or_modules(self):
        """Check that we throw a nice error if the user tries to resolve nonsense"""
        with self.assertRaises(TypeError):
            caikit.core.resolve_and_load(1)
        with self.assertRaises(TypeError):
            caikit.core.resolve_and_load(["test", "not", "a", "module"])
        with self.assertRaises(TypeError):
            caikit.core.resolve_and_load({"this dict is": "not a module"})

    def test_load_model_with_artifacts_from_zip_str(self):
        """Test that we can load a model archive [extracts to temp_dir/...] with artifacts."""
        model = caikit.core.load(self.block_zip_path)
        self.assertIsInstance(model, caikit.core.BlockBase)

    def test_load_model_with_artifacts_from_bytes(self):
        """Test that we can load a bytes object as a model, even if it has artifacts."""
        model_bytes = caikit.core.load(self.block_zip_path).as_bytes()
        model = caikit.core.load(model_bytes)
        self.assertIsInstance(model, caikit.core.BlockBase)

    def test_load_model_with_artifacts_from_file_like(self):
        """Test that we can load a file-like object as a model, even if it has artifacts."""
        model_bytesio = caikit.core.load(self.block_zip_path).as_file_like_object()
        model = caikit.core.load(model_bytesio)
        self.assertIsInstance(model, caikit.core.BlockBase)

    def test_load_model_with_no_nesting(self):
        """Test that we can load a zip even if it unzips directly into the extraction archive."""
        model_path = os.path.join(self.fixtures_dir, "dummy_block_no_nesting.zip")
        model = caikit.core.load(model_path)
        self.assertIsInstance(model, caikit.core.BlockBase)

    def test_load_invalid_zip_file(self):
        """Test that loading a zip archive not containing a model fails gracefully."""
        model_path = os.path.join(self.fixtures_dir, "invalid.zip")
        with self.assertRaises(FileNotFoundError):
            caikit.core.load(model_path)

    @pytest.mark.usefixtures("global_load_path")
    def test_load_path(self):
        """Test that loading a model from a path defined in the load_path config variable works."""
        model = caikit.core.load("studio_block")
        self.assertIsInstance(model, caikit.core.BlockBase)

    def test_import_block_registry(self):
        """Make sure that the BLOCK_REGISTRY can be imported from model_manager"""
        # pylint: disable = import-outside-toplevel,no-name-in-module,unused-import
        # Local
        from caikit.core.model_manager import BLOCK_REGISTRY

    def test_import_workflow_registry(self):
        """Make sure that the WORKFLOW_REGISTRY can be imported from model_manager"""
        # pylint: disable = import-outside-toplevel,no-name-in-module,unused-import
        # Local
        from caikit.core.model_manager import WORKFLOW_REGISTRY

    def test_import_resource_registry(self):
        """Make sure that the RESOURCE_REGISTRY can be imported from model_manager"""
        # pylint: disable = import-outside-toplevel,no-name-in-module,unused-import
        # Local
        from caikit.core.model_manager import RESOURCE_REGISTRY


# Pytest tests #########################################################

# Setup #########################################################################

DUMMY_MODULE_ID = "foo"

TEST_DATA_PATH = os.path.join("tests", "fixtures")
DUMMY_LOCAL_MODEL_NAME = "dummy_block_foo"
DUMMY_BACKEND_MODEL_NAME = "dummy_block_backend"
CONFIG_FILE_NAME = "config.yml"


def setup_saved_model(mock_backend_class):
    """Fixture to create and cleanup a dummy loadable model"""

    backend_types.register_backend_type(LocalBackend)

    @caikit.core.blocks.block(id=DUMMY_MODULE_ID, name="dummy base", version="0.0.1")
    class DummyFoo(caikit.core.blocks.base.BlockBase):
        @classmethod
        def load(cls, *args, **kwargs):
            return cls()

    # Register backend type
    backend_types.register_backend_type(mock_backend_class)

    @caikit.core.blocks.block(base_module=DummyFoo, backend_type=backend_types.MOCK)
    class DummyBar:
        SUPPORTED_LOAD_BACKENDS = [backend_types.MOCK, backend_types.LOCAL]

        @classmethod
        def load(self, *args, **kwargs):
            return DummyBar()

    return DummyFoo, DummyBar


@caikit.core.blocks.block(
    id="non-distributed", name="non distributed mod", version="0.0.1"
)
class NonDistributedBlock(caikit.core.blocks.base.BlockBase):
    @classmethod
    def load(cls, *args, **kwargs):
        return cls()

    def save(self, model_path):
        block_saver = caikit.core.blocks.BlockSaver(
            self,
            model_path=model_path,
            library_name="foo",
            library_version="0.42.0",
        )
        with block_saver:
            pass


@contextmanager
def temp_saved_model(model):
    """Temporarily save the model and yield its path"""
    with tempfile.TemporaryDirectory() as model_path:
        model.save(model_path)
        yield model_path


## Tests #######################################################################


def test_backend_supported_model_load_successfully(reset_globals):
    """Test backend supported model can load successfully"""

    _, DummyBar = setup_saved_model(MockBackend)
    # Configure backend
    configure(backend_priority=[backend_types.MOCK], backends={"mock": {}})

    dummy_model_path = os.path.join(TEST_DATA_PATH, DUMMY_BACKEND_MODEL_NAME)
    model = caikit.core.load(dummy_model_path)
    assert isinstance(model, DummyBar)


def test_local_model_load_successfully(reset_globals):
    """Test regular / local model can load successfully"""
    DummyFoo, DummyBar = setup_saved_model(MockBackend)

    # Configure backend
    configure(backend_priority=[backend_types.LOCAL], backends={"mock": {}})

    dummy_model_path = os.path.join(TEST_DATA_PATH, DUMMY_LOCAL_MODEL_NAME)
    model = caikit.core.load(dummy_model_path)

    assert isinstance(model, DummyFoo)
    assert not isinstance(model, DummyBar)


def test_local_model_loaded_backend_successfully(reset_globals):
    """Test if local models can be loaded in specified backend
    if available in SUPPORTED_LOAD_BACKENDS
    """
    _, DummyBar = setup_saved_model(MockBackend)
    # Configure backend
    configure(backend_priority=[backend_types.MOCK], backends={"mock": {}})

    DummyBar.SUPPORTED_LOAD_BACKENDS.append("LOCAL")
    dummy_model_path = os.path.join(TEST_DATA_PATH, DUMMY_LOCAL_MODEL_NAME)
    model = caikit.core.load(dummy_model_path)
    assert isinstance(model, DummyBar)


def test_backend_model_loaded_as_singleton(reset_globals):
    """Test that backend specific model can be loaded as a singleton"""

    _, DummyBar = setup_saved_model(MockBackend)
    # Configure backend
    configure(backend_priority=[backend_types.MOCK], backends={"mock": {}})

    dummy_model_path = os.path.join(TEST_DATA_PATH, DUMMY_BACKEND_MODEL_NAME)
    model1 = caikit.core.load(dummy_model_path, load_singleton=True)
    model2 = caikit.core.load(dummy_model_path, load_singleton=True)

    # Pointers should be same
    assert id(model1) == id(model2)


def test_singleton_cache_can_be_cleared(reset_globals):
    """Test that a singleton cache can be cleared"""
    dummy_model_path = os.path.join(TEST_DATA_PATH, DUMMY_BACKEND_MODEL_NAME)
    # Setup the mock backend
    _, _ = setup_saved_model(MockBackend)
    model = caikit.core.load(dummy_model_path, load_singleton=True)

    assert len(caikit.core.MODEL_MANAGER.get_singleton_model_cache_info()) > 0

    # Clear cache
    caikit.core.MODEL_MANAGER.clear_singleton_cache()
    assert len(caikit.core.MODEL_MANAGER.get_singleton_model_cache_info()) == 0


def test_singleton_cache_with_different_backend(reset_globals):
    """Test singleton cache doesn't stop different backend models"""

    DummyFoo, DummyBar = setup_saved_model(MockBackend)
    # Configure backend
    configure(backend_priority=[backend_types.MOCK], backends={"mock": {}})

    dummy_model_path = os.path.join(TEST_DATA_PATH, DUMMY_BACKEND_MODEL_NAME)
    model1 = caikit.core.load(dummy_model_path, load_singleton=True)

    dummy_model_path = os.path.join(TEST_DATA_PATH, DUMMY_LOCAL_MODEL_NAME)
    model2 = caikit.core.load(dummy_model_path, load_singleton=True)

    assert len(caikit.core.MODEL_MANAGER.get_singleton_model_cache_info()) == 2


def test_get_module_class():
    """Test to verify get_module_class function can return appropriate module class"""
    # Block
    config = {"block_id": "foo", "block_class": "Foo"}
    module_config = caikit.core.module.ModuleConfig(config)
    assert caikit.core.ModelManager.get_module_class_from_config(module_config) == "Foo"

    # Workflow
    config = {"workflow_id": "foo", "workflow_class": "Foo"}
    module_config = caikit.core.module.ModuleConfig(config)
    assert caikit.core.ModelManager.get_module_class_from_config(module_config) == "Foo"

    # Resource
    config = {"resource_id": "foo", "resource_class": "Foo"}
    module_config = caikit.core.module.ModuleConfig(config)
    assert caikit.core.ModelManager.get_module_class_from_config(module_config) == "Foo"


def test_fall_back_to_local(reset_globals):
    """Make sure that if LOCAL is enabled and a given module doesn't have any
    registered backends, the default caikit.core.load is used.
    """
    configure(backend_priority=[])
    with temp_saved_model(NonDistributedBlock()) as model_path:
        model = caikit.core.load(model_path)

    assert isinstance(model, NonDistributedBlock)


def test_no_local_if_disabled(reset_globals):
    """Make sure that if LOCAL is disabled and a given module doesn't have any
    registered backends, loading fails.
    """
    _ = setup_saved_model(MockBackend)
    configure(backend_priority=[backend_types.MOCK], disable_local_backend=True)
    with temp_saved_model(NonDistributedBlock()) as model_path:
        with pytest.raises(ValueError):
            caikit.core.load(model_path)


def test_preferred_backend_enabled(reset_globals):
    """Make sure that for a model artifact saved with a local backend can be
    loaded with an enabled non-local backend if given as a preferred_backend.
    """
    _, DummyBar = setup_saved_model(MockBackend)
    configure(backend_priority=[backend_types.MOCK], backends={"mock": {}})

    dummy_model_path = os.path.join(TEST_DATA_PATH, DUMMY_LOCAL_MODEL_NAME)
    model = caikit.core.load(dummy_model_path)
    assert isinstance(model, DummyBar)


def test_preferred_backend_disabled(reset_globals):
    """Make sure that for a model artifact saved with a local backend loads as
    local even with a preferred_backend when the preferred backend is disabled.
    """
    DummyFoo, DummyBar = setup_saved_model(MockBackend)
    configure(backend_priority=[backend_types.LOCAL], backends={})

    dummy_model_path = os.path.join(TEST_DATA_PATH, DUMMY_LOCAL_MODEL_NAME)
    model = caikit.core.load(dummy_model_path)
    assert isinstance(model, DummyFoo)
    assert not isinstance(model, DummyBar)


def test_non_local_supported_backed(reset_globals):
    """Make sure model artifact saved with as non-local backend loads as
    non-local backend if supported by SUPPORTED_LOAD_BACKENDS
    """
    DummyFoo, _ = setup_saved_model(MockBackend)

    class MockBackend2(BackendBase):
        backend_type = "MOCK2"

        def start(self):
            pass

        def register_config(self, config):
            pass

        def stop(self):
            pass

    backend_types.register_backend_type(MockBackend2)

    @caikit.core.blocks.block(base_module=DummyFoo, backend_type=backend_types.MOCK2)
    class DummyBaz:
        SUPPORTED_LOAD_BACKENDS = [backend_types.MOCK, backend_types.MOCK2]

        @classmethod
        def load(self, *args, **kwargs):
            return DummyBaz()

    configure(backend_priority=[backend_types.MOCK2], backends={})

    dummy_model_path = os.path.join(TEST_DATA_PATH, DUMMY_BACKEND_MODEL_NAME)
    model = caikit.core.load(dummy_model_path)
    assert isinstance(model, DummyBaz)
