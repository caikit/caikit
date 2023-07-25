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

# Local
from caikit.core import LocalBackend
from caikit.core.data_model import DataStream, TrainingStatus
from caikit.core.model_management import ModelFinderBase, model_finder_factory

# Unit Test Infrastructure
from sample_lib.modules.sample_task import SampleModule
from tests.base import TestCaseBase
from tests.conftest import backend_priority, temp_config

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
        cls.model_path = os.path.join(cls.fixtures_dir, "dummy_module")
        # This model has no unique hash set in its config; we use it as a nonsingleton model too
        cls.non_singleton_model_path = cls.model_path
        cls.singleton_model_path = os.path.join(
            cls.fixtures_dir, "dummy_module_singleton"
        )

        cls.resource_path = os.path.join(cls.fixtures_dir, "dummy_resource")

        # Binary buffers of zip archives, for mocking downloads
        cls.module_zip_path = os.path.join(cls.fixtures_dir, "dummy_module.zip")
        with open(cls.module_zip_path, "rb") as f:
            cls.module_archive_buffer = f.read()

    @pytest.fixture
    def global_load_path(self):
        """Set load_path prior to importing caikit.core."""
        # Set test load path
        test_load_path = os.path.join(self.fixtures_dir, "models")

        with temp_config({"load_path": test_load_path}):
            yield

    def test_load_can_return_a_module(self):
        model = caikit.core.load(self.model_path)
        self.assertIsInstance(model, caikit.core.ModuleBase)

    def test_load_can_load_a_module_as_a_singleton(self):
        model1 = caikit.core.load(self.singleton_model_path, load_singleton=True)
        model2 = caikit.core.load(self.singleton_model_path, load_singleton=True)
        assert model1 is model2

    def test_load_can_load_a_module_with_singleton_disabled(self):
        model1 = caikit.core.load(self.singleton_model_path, load_singleton=True)
        model2 = caikit.core.load(self.singleton_model_path, load_singleton=False)
        assert model1 is not model2

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
                self.module_zip_path, tempdir, force_overwrite=True
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
        model = caikit.core.load(self.module_zip_path)
        self.assertIsInstance(model, caikit.core.ModuleBase)

    def test_load_model_with_artifacts_from_bytes(self):
        """Test that we can load a bytes object as a model, even if it has artifacts."""
        model_bytes = caikit.core.load(self.module_zip_path).as_bytes()
        model = caikit.core.load(model_bytes)
        self.assertIsInstance(model, caikit.core.ModuleBase)

    def test_load_model_with_artifacts_from_file_like(self):
        """Test that we can load a file-like object as a model, even if it has artifacts."""
        model_bytesio = caikit.core.load(self.module_zip_path).as_file_like_object()
        model = caikit.core.load(model_bytesio)
        self.assertIsInstance(model, caikit.core.ModuleBase)

    def test_load_model_with_no_nesting(self):
        """Test that we can load a zip even if it unzips directly into the extraction archive."""
        model_path = os.path.join(self.fixtures_dir, "dummy_module_no_nesting.zip")
        model = caikit.core.load(model_path)
        self.assertIsInstance(model, caikit.core.ModuleBase)

    def test_load_invalid_zip_file(self):
        """Test that loading a zip archive not containing a model fails gracefully."""
        model_path = os.path.join(self.fixtures_dir, "invalid.zip")
        with self.assertRaises(FileNotFoundError):
            caikit.core.load(model_path)

    @pytest.mark.usefixtures("global_load_path")
    def test_load_path(self):
        """Test that loading a model from a path defined in the load_path config variable works."""
        model = caikit.core.load("foo")
        self.assertIsInstance(model, caikit.core.ModuleBase)

    def test_import_module_registry(self):
        """Make sure that the module registry can be imported from model_manager"""
        # pylint: disable = import-outside-toplevel,no-name-in-module,unused-import
        from caikit.core.model_manager import module_registry  # isort: skip

    def test_load_model_with_no_module_id(self):
        """Test that we can load a module with the legacy block_id instead of module_id"""
        model_path = os.path.join(self.fixtures_dir, DUMMY_NO_ID_MODEL_NAME)
        model = caikit.core.load(model_path)
        self.assertIsInstance(model, caikit.core.ModuleBase)


# Pytest tests #########################################################

# Setup #########################################################################

DUMMY_MODULE_ID = "foo"

TEST_DATA_PATH = os.path.join("tests", "fixtures")
DUMMY_LOCAL_MODEL_NAME = "dummy_module_foo"
DUMMY_BACKEND_MODEL_NAME = "dummy_module_backend"
DUMMY_NO_ID_MODEL_NAME = "dummy_module_no_id"
CONFIG_FILE_NAME = "config.yml"


def setup_saved_model(mock_backend_class):
    """Fixture to create and cleanup a dummy loadable model"""

    backend_types.register_backend_type(LocalBackend)

    @caikit.core.modules.module(id=DUMMY_MODULE_ID, name="dummy base", version="0.0.1")
    class DummyFoo(caikit.core.ModuleBase):
        @classmethod
        def load(cls, *args, **kwargs):
            return cls()

    # Register backend type
    backend_types.register_backend_type(mock_backend_class)

    @caikit.core.modules.module(base_module=DummyFoo, backend_type=backend_types.MOCK)
    class DummyBar(caikit.core.ModuleBase):
        SUPPORTED_LOAD_BACKENDS = [backend_types.MOCK, backend_types.LOCAL]

        @classmethod
        def load(self, *args, **kwargs):
            return DummyBar()

    return DummyFoo, DummyBar


@caikit.core.modules.module(
    id="non-distributed", name="non distributed mod", version="0.0.1"
)
class NonDistributedModule(caikit.core.ModuleBase):
    @classmethod
    def load(cls, *args, **kwargs):
        return cls()

    def save(self, model_path):
        module_saver = caikit.core.modules.ModuleSaver(
            self,
            model_path=model_path,
        )
        with module_saver:
            pass


@contextmanager
def temp_saved_model(model):
    """Temporarily save the model and yield its path"""
    with tempfile.TemporaryDirectory() as model_path:
        model.save(model_path)
        yield model_path


class NoYamlFinder(ModelFinderBase):
    name = "NOYAML"

    def __init__(self, *_, **__):
        pass

    def find_model(self, model_path):
        """This load function doesn't read from model_path, so it definitely does not read the config.yml file"""
        return ModuleConfig({"module_id": SampleModule.MODULE_ID})


class NoYamlInitializer(ModelInitializerBase):
    name = "NOYAML"

    def __init__(self, *_, **__):
        pass

    def init(self, model_config):
        """This load function doesn't read from model_path, so it definitely does not read the config.yml file"""
        assert model_config.module_id == SampleModule.MODULE_ID
        return SampleModule()


model_finder_factory.register(NoYamlFinder)
model_initializer_factory.register(NoYamlInitializer)


@contextmanager
def setup_test_trainer(config=None):
    factory_blob = {"type": TestTrainer.name}
    if config is not None:
        factory_blob["config"] = config
    with temp_config({"model_management": {"trainers": {"default": factory_blob}}}):
        yield


## Tests #######################################################################


def test_backend_supported_model_load_successfully(reset_globals):
    """Test backend supported model can load successfully"""
    _, DummyBar = setup_saved_model(MockBackend)
    with backend_priority({"type": backend_types.MOCK}):
        dummy_model_path = os.path.join(TEST_DATA_PATH, DUMMY_BACKEND_MODEL_NAME)
        model = caikit.core.load(dummy_model_path)
        assert isinstance(model, DummyBar)


def test_local_model_load_successfully(reset_globals):
    """Test regular / local model can load successfully"""
    DummyFoo, DummyBar = setup_saved_model(MockBackend)
    with backend_priority({"type": backend_types.LOCAL}):
        dummy_model_path = os.path.join(TEST_DATA_PATH, DUMMY_LOCAL_MODEL_NAME)
        model = caikit.core.load(dummy_model_path)
        assert isinstance(model, DummyFoo)
        assert not isinstance(model, DummyBar)


def test_local_model_loaded_backend_successfully(reset_globals):
    """Test if local models can be loaded in specified backend
    if available in SUPPORTED_LOAD_BACKENDS
    """
    _, DummyBar = setup_saved_model(MockBackend)
    with backend_priority({"type": backend_types.MOCK}):
        DummyBar.SUPPORTED_LOAD_BACKENDS.append("LOCAL")
        dummy_model_path = os.path.join(TEST_DATA_PATH, DUMMY_LOCAL_MODEL_NAME)
        model = caikit.core.load(dummy_model_path)
        assert isinstance(model, DummyBar)


def test_backend_model_loaded_as_singleton(reset_globals):
    """Test that backend specific model can be loaded as a singleton"""

    _, DummyBar = setup_saved_model(MockBackend)
    with backend_priority({"type": backend_types.MOCK}):
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
    _ = caikit.core.load(dummy_model_path, load_singleton=True)

    assert len(caikit.core.MODEL_MANAGER.get_singleton_model_cache_info()) > 0

    # Clear cache
    caikit.core.MODEL_MANAGER.clear_singleton_cache()
    assert len(caikit.core.MODEL_MANAGER.get_singleton_model_cache_info()) == 0


def test_singleton_cache_with_different_backend(reset_globals):
    """Test singleton cache doesn't stop different backend models"""

    _, _ = setup_saved_model(MockBackend)
    with backend_priority({"type": backend_types.MOCK}):
        dummy_model_path = os.path.join(TEST_DATA_PATH, DUMMY_BACKEND_MODEL_NAME)
        _ = caikit.core.load(dummy_model_path, load_singleton=True)

        dummy_model_path = os.path.join(TEST_DATA_PATH, DUMMY_LOCAL_MODEL_NAME)
        _ = caikit.core.load(dummy_model_path, load_singleton=True)

        assert len(caikit.core.MODEL_MANAGER.get_singleton_model_cache_info()) == 2


def test_fall_back_to_local(reset_globals):
    """Make sure that if LOCAL is enabled and a given module doesn't have any
    registered backends, the default caikit.core.load is used.
    """
    with temp_saved_model(NonDistributedModule()) as model_path:
        model = caikit.core.load(model_path)

    assert isinstance(model, NonDistributedModule)


def test_load_fails_on_no_supported_backend(reset_globals):
    """Make sure if a given module doesn't have any registered backends,
    loading fails.
    """
    _ = setup_saved_model(MockBackend)
    with backend_priority({"type": backend_types.MOCK}):
        with temp_saved_model(NonDistributedModule()) as model_path:
            with pytest.raises(ValueError):
                caikit.core.load(model_path)


def test_preferred_backend_enabled(reset_globals):
    """Make sure that for a model artifact saved with a local backend can be
    loaded with an enabled non-local backend if given as a preferred_backend.
    """
    _, DummyBar = setup_saved_model(MockBackend)
    with backend_priority({"type": backend_types.MOCK}):
        dummy_model_path = os.path.join(TEST_DATA_PATH, DUMMY_LOCAL_MODEL_NAME)
        model = caikit.core.load(dummy_model_path)
        assert isinstance(model, DummyBar)


def test_module_backend_instance_is_passed_to_load_classmethod(reset_globals):
    """When a module's load function supports the `load_backend` kwarg, the concrete
    instance of the module backend is passed in at load time.
    """
    # re-register the local backend because this test clears out the global registries
    backend_types.register_backend_type(LocalBackend)

    @caikit.core.modules.module(
        id=DUMMY_MODULE_ID, name="backend loader", version="0.0.1"
    )
    class DummyBackendLoadingModule(caikit.core.ModuleBase):
        @classmethod
        def load(cls, model_path: str, load_backend: BackendBase):
            assert load_backend is not None
            return cls()

    dummy_model_path = os.path.join(TEST_DATA_PATH, DUMMY_LOCAL_MODEL_NAME)
    caikit.core.load(dummy_model_path)


def test_preferred_backend_disabled(reset_globals):
    """Make sure that for a model artifact saved with a local backend loads as
    local even with a preferred_backend when the preferred backend is disabled.
    """
    DummyFoo, DummyBar = setup_saved_model(MockBackend)
    with backend_priority({"type": backend_types.LOCAL}):
        dummy_model_path = os.path.join(TEST_DATA_PATH, DUMMY_LOCAL_MODEL_NAME)
        model = caikit.core.load(dummy_model_path)
        assert isinstance(model, DummyFoo)
        assert not isinstance(model, DummyBar)


def test_non_local_supported_backend(reset_globals):
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

    @caikit.core.modules.module(base_module=DummyFoo, backend_type=backend_types.MOCK2)
    class DummyBaz(caikit.core.ModuleBase):
        SUPPORTED_LOAD_BACKENDS = [backend_types.MOCK, backend_types.MOCK2]

        @classmethod
        def load(self, *args, **kwargs):
            return DummyBaz()

    with backend_priority({"type": backend_types.MOCK2}):
        dummy_model_path = os.path.join(TEST_DATA_PATH, DUMMY_BACKEND_MODEL_NAME)
        model = caikit.core.load(dummy_model_path)
        assert isinstance(model, DummyBaz)


def test_load_must_return_model():
    """Make sure that the return type of load is checked to be an instance of
    ModuleBase, and will raise TypeError if it is not.
    """

    @caikit.core.module("00110203-baad-beef-0809-0a0b0c0d0e0f", "FunkyModule", "0.0.1")
    class _FunkyModel(SampleModule):
        @classmethod
        def load(cls, model_path):
            return (super().load(model_path), "something else")

    model = _FunkyModel()
    with tempfile.TemporaryDirectory() as tempdir:
        # NOTE: the module will get detected as tests since _FunkyModel is defined here
        model.save(tempdir)
        with pytest.raises(TypeError):
            caikit.core.load(tempdir)


def test_load_with_two_shared_initializers_of_the_same_type(
    good_model_path, reset_globals
):
    """Multiple instances of one shared loader type can be configured"""

    with temp_config(
        {
            "model_management": {
                "finders": {"default": {"type": "LOCAL"}},
                "initializers": {
                    "model-one": {
                        "type": TestInitializer.name,
                        "config": {"model_type": "model one"},
                    },
                    "model-two": {
                        "type": TestInitializer.name,
                        "config": {"model_type": "model two"},
                    },
                    "default": {"type": "LOCAL"},
                },
            }
        }
    ):
        model_one_loader = MODEL_MANAGER._get_initializer("model-one")
        model_two_loader = MODEL_MANAGER._get_initializer("model-two")

        # plain model load should use first loader
        model = caikit.core.load(good_model_path, initializer="model-one")
        assert model_one_loader
        assert model in model_one_loader.loaded_models
        assert model not in model_two_loader.loaded_models

        # model load that fails in the first loader will use the second
        model = caikit.core.load(good_model_path, initializer="model-two")
        assert model not in model_one_loader.loaded_models
        assert model in model_two_loader.loaded_models


def test_load_does_not_read_config_yml_if_loader_does_not_require_it(
    reset_globals, tmp_path
):
    tmpdir = str(tmp_path)
    with open(os.path.join(tmpdir, "config.yml"), "w") as f:
        f.write("{this is not yaml} !!@#$%^")

    with temp_config(
        {
            "model_management": {
                "finders": {
                    "default": {"type": NoYamlFinder.name},
                },
                "initializers": {
                    "default": {"type": NoYamlInitializer.name},
                },
            }
        }
    ):
        model = caikit.core.load(tmpdir)
        assert isinstance(model, SampleModule)


def test_load_loader_finder_by_name(reset_globals):
    """Make sure that non-default loaders and finders can be referenced by name
    when calling load
    """
    with temp_config(
        {
            "model_management": {
                "finders": {
                    "default": {"type": "LOCAL"},
                    "other": {"type": NoYamlFinder.name},
                },
                "initializers": {
                    "default": {"type": "LOCAL"},
                    "other": {"type": NoYamlInitializer.name},
                },
            }
        }
    ):
        model = caikit.core.load("unused", finder="other", initializer="other")
        assert isinstance(model, SampleModule)


def test_load_loader_finder_by_value(reset_globals):
    """Make sure that non-default loaders and finders can be passed by value"""
    with temp_config(
        {
            "model_management": {
                "finders": {
                    "default": {"type": "LOCAL"},
                },
                "initializers": {
                    "default": {"type": "LOCAL"},
                },
            }
        }
    ):
        model = caikit.core.load(
            "unused", finder=NoYamlFinder(), initializer=NoYamlInitializer()
        )
        assert isinstance(model, SampleModule)


def test_initialize_all_components(reset_globals):
    """Make sure that calling initialize_components will proactively create all
    component instances from config
    """
    with temp_config(
        {
            "model_management": {
                "trainers": {
                    "default": {"type": "LOCAL"},
                    "foobar": {"type": "LOCAL"},
                },
                "finders": {
                    "default": {"type": "LOCAL"},
                    "foobar": {"type": "LOCAL"},
                },
                "initializers": {
                    "default": {"type": "LOCAL"},
                    "foobar": {"type": "LOCAL"},
                },
            }
        }
    ):
        assert not MODEL_MANAGER._trainers
        assert not MODEL_MANAGER._finders
        assert not MODEL_MANAGER._initializers
        MODEL_MANAGER.initialize_components()
        assert set(MODEL_MANAGER._trainers.keys()) == {"default", "foobar"}
        assert set(MODEL_MANAGER._finders.keys()) == {"default", "foobar"}
        assert set(MODEL_MANAGER._initializers.keys()) == {"default", "foobar"}


def test_train_by_module_class(reset_globals):
    """Make sure training can be accessed through the central train function
    with the class directly
    """
    with setup_test_trainer():
        train_future = caikit.train(SampleModule, DataStream.from_iterable([]))
        assert isinstance(train_future, ModelTrainerBase.ModelFutureBase)
        loaded_model = train_future.load()
        assert isinstance(loaded_model, SampleModule)
        found_future = caikit.get_model_future(train_future.id)
        assert found_future is train_future


def test_train_by_module_id(reset_globals):
    """Make sure training can be accessed through the central train function
    with the module_id for the module to train
    """
    with setup_test_trainer():
        train_future = caikit.train(
            SampleModule.MODULE_ID, DataStream.from_iterable([])
        )
        assert isinstance(train_future, ModelTrainerBase.ModelFutureBase)
        loaded_model = train_future.load()
        assert isinstance(loaded_model, SampleModule)
        found_future = caikit.get_model_future(train_future.id)
        assert found_future is train_future


def test_train_with_wait(reset_globals):
    """Make sure calling train with wait correctly waits for completion"""
    with setup_test_trainer():
        # Call without waiting and make sure it's not COMPLETED
        unfinished_train_future = caikit.train(
            SampleModule, DataStream.from_iterable([])
        )
        assert unfinished_train_future.get_info().status == TrainingStatus.RUNNING
        # Call with wait and make sure it is COMPLETED
        finished_train_future = caikit.train(
            SampleModule, DataStream.from_iterable([]), wait=True
        )
        assert finished_train_future.get_info().status == TrainingStatus.COMPLETED


def test_train_with_save_path(reset_globals):
    """Make sure calling train with a save_path correctly saves out the model"""
    with setup_test_trainer():
        with tempfile.TemporaryDirectory() as workdir:
            save_path = os.path.join(workdir, "saved_model")
            train_future = caikit.train(
                SampleModule, DataStream.from_iterable([]), save_path=save_path
            )
            assert train_future.get_info().status == TrainingStatus.RUNNING
            assert not os.path.exists(save_path)
            assert train_future.save_path == save_path
            train_future.wait()
            assert train_future.get_info().status == TrainingStatus.COMPLETED
            assert os.path.exists(save_path)
