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
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from tempfile import TemporaryDirectory
from typing import Optional
from unittest.mock import MagicMock, patch
import os
import shutil
import threading
import time

# Third Party
import grpc
import pytest

# First Party
from aconfig.aconfig import Config
import aconfig

# Local
from caikit import get_config
from caikit.core.model_management import ModelFinderBase
from caikit.core.model_management.local_model_initializer import LocalModelInitializer
from caikit.core.model_manager import ModelManager as CoreModelManager
from caikit.core.modules import ModuleBase, ModuleConfig
from caikit.runtime.model_management.loaded_model import LoadedModel
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from caikit.runtime.utils.import_util import get_dynamic_module
from sample_lib.data_model import SampleInputType
from tests.conftest import TempFailWrapper, random_test_id, temp_config
from tests.core.helpers import TestFinder
from tests.fixtures import Fixtures
from tests.runtime.conftest import deploy_good_model_files
import caikit.runtime.model_management.model_loader_base

get_dynamic_module("caikit.core")
ANY_MODEL_TYPE = "test-any-model-type"
ANY_MODEL_PATH = "test-any-model-path"
MODEL_MANAGER = ModelManager.get_instance()


@pytest.fixture(autouse=True)
def tear_down():
    yield
    MODEL_MANAGER.unload_all_models()


@contextmanager
def temp_local_models_dir(workdir, model_manager=MODEL_MANAGER):
    prev_local_models_dir = model_manager._local_models_dir
    model_manager._local_models_dir = workdir
    yield
    model_manager._local_models_dir = prev_local_models_dir


@contextmanager
def non_singleton_model_managers(num_mgrs=1, *args, **kwargs):
    with temp_config(*args, **kwargs):
        with patch(
            "caikit.runtime.model_management.core_model_loader.MODEL_MANAGER",
            new_callable=CoreModelManager,
        ):
            instances = []
            try:
                for _ in range(num_mgrs):
                    ModelManager._ModelManager__instance = None
                    instances.append(ModelManager.get_instance())
                yield instances
            finally:
                for inst in instances:
                    inst.shut_down()
                ModelManager._ModelManager__instance = MODEL_MANAGER


class SlowLoader:
    """Helper class to simulate slow loading"""

    def __init__(self, load_result="STUB"):
        self._load_result = load_result
        self._load_start_event = threading.Event()
        self._load_end_event = threading.Event()

    def load(self, *_, **__):
        self._load_start_event.wait()
        self._load_end_event.set()
        if isinstance(self._load_result, Exception):
            raise self._load_result
        return self._load_result

    def unblock_load(self):
        self._load_start_event.set()

    def done_loading(self):
        return self._load_end_event.is_set()


# ****************************** Integration Tests ****************************** #
# These tests do not patch in mocks, the manager will use real instances of its dependencies


def test_load_model_ok_response():
    model_id = "happy_load_test"
    model_size = MODEL_MANAGER.load_model(
        model_id=model_id,
        local_model_path=Fixtures.get_good_model_path(),
        model_type=Fixtures.get_good_model_type(),
    ).size()
    assert model_size > 0


@pytest.mark.parametrize("model_id", ["", b"bytes-id", 123])
def test_retrieve_invalid_model_ids(model_id):
    with pytest.raises(CaikitRuntimeException) as context:
        MODEL_MANAGER.retrieve_model(model_id)
    assert context.value.status_code == grpc.StatusCode.INVALID_ARGUMENT


def test_load_model_no_size_update():
    model_id = random_test_id()
    model_size = MODEL_MANAGER.load_model(
        model_id=model_id,
        local_model_path=Fixtures.get_good_model_path(),
        model_type=Fixtures.get_good_model_type(),
    ).size()
    assert model_size > 0
    loaded_model = MODEL_MANAGER.loaded_models[model_id]
    assert loaded_model.size() == model_size
    loaded_model.set_size(model_size * 2)
    assert loaded_model.size() == model_size


def test_load_local_models():
    with TemporaryDirectory() as tempdir:
        shutil.copytree(Fixtures.get_good_model_path(), os.path.join(tempdir, "model1"))
        shutil.copy(
            Fixtures.get_good_model_archive_path(),
            os.path.join(tempdir, "model2.zip"),
        )

        with temp_local_models_dir(tempdir):
            MODEL_MANAGER.sync_local_models(wait=True)
        assert len(MODEL_MANAGER.loaded_models) == 2
        assert "model1" in MODEL_MANAGER.loaded_models.keys()
        assert "model2.zip" in MODEL_MANAGER.loaded_models.keys()
        assert "model-does-not-exist.zip" not in MODEL_MANAGER.loaded_models.keys()


@pytest.mark.parametrize("wait", [True, False])
def test_model_manager_loads_local_models_on_init(wait):
    with TemporaryDirectory() as tempdir:
        shutil.copytree(Fixtures.get_good_model_path(), os.path.join(tempdir, "model1"))
        shutil.copy(
            Fixtures.get_good_model_archive_path(),
            os.path.join(tempdir, "model2.zip"),
        )
        ModelManager._ModelManager__instance = None
        with temp_config(
            {
                "runtime": {
                    "local_models_dir": tempdir,
                    "wait_for_initial_model_loads": wait,
                },
            },
            merge_strategy="merge",
        ):
            MODEL_MANAGER = ModelManager()

            assert len(MODEL_MANAGER.loaded_models) == 2
            assert "model1" in MODEL_MANAGER.loaded_models.keys()
            assert "model2.zip" in MODEL_MANAGER.loaded_models.keys()
            assert "model-does-not-exist.zip" not in MODEL_MANAGER.loaded_models.keys()

            # Make sure that the loaded model can be retrieved and run
            for model_name in ["model1", "model2.zip"]:
                model = MODEL_MANAGER.retrieve_model(model_name)
                model.run(SampleInputType("hello"))


def test_load_model_error_response():
    """Test load model's model does not exist when the loader throws"""
    with pytest.raises(CaikitRuntimeException) as context:
        MODEL_MANAGER.load_model(
            model_id=random_test_id(),
            local_model_path=Fixtures().get_invalid_model_archive_path(),
            model_type="categories_esa",
            wait=True,
        )

    assert context.value.status_code == grpc.StatusCode.NOT_FOUND
    assert len(MODEL_MANAGER.loaded_models) == 0


def test_load_model_map_insertion():
    """Test if loaded model is correctly added to map storing model data"""
    model = random_test_id()
    MODEL_MANAGER.load_model(
        model_id=model,
        local_model_path=Fixtures.get_good_model_path(),
        model_type=Fixtures.get_good_model_type(),
    )
    assert MODEL_MANAGER.loaded_models[model].size() > 0


def test_load_model_count():
    """Test if multiple loaded models are added to map storing model data"""
    MODEL_MANAGER.load_model(
        model_id=random_test_id(),
        local_model_path=Fixtures.get_good_model_path(),
        model_type=Fixtures.get_good_model_type(),
    )
    MODEL_MANAGER.load_model(
        model_id=random_test_id(),
        local_model_path=Fixtures.get_good_model_path(),
        model_type=Fixtures.get_good_model_type(),
    )
    assert len(MODEL_MANAGER.loaded_models) == 2


def test_unload_model_ok_response():
    """Test to make sure that given a loaded model ID, the model manager is able to correctly
    unload a model, giving a nonzero model size back."""
    model_id = "happy_unload_test"
    MODEL_MANAGER.load_model(
        model_id=model_id,
        local_model_path=Fixtures.get_good_model_path(),
        model_type=Fixtures.get_good_model_type(),
    )
    model_size = MODEL_MANAGER.unload_model(model_id=model_id)
    assert model_size >= 0


def test_unload_model_count():
    """Test if unloaded models are deleted from loaded models map"""
    id_1 = "test"
    id_2 = "test2"
    # Load models from COS
    MODEL_MANAGER.load_model(
        model_id=id_1,
        local_model_path=Fixtures.get_good_model_path(),
        model_type=Fixtures.get_good_model_type(),
    )
    MODEL_MANAGER.load_model(
        model_id=id_2,
        local_model_path=Fixtures.get_good_model_path(),
        model_type=Fixtures.get_good_model_type(),
    )
    # Unload one of the models, and make sure it was properly removed
    MODEL_MANAGER.unload_model(model_id=id_2)
    assert len(MODEL_MANAGER.loaded_models) == 1
    assert id_1 in MODEL_MANAGER.loaded_models.keys()


def test_unload_all_models_count():
    """Test if unload all models deletes every model from loaded models map"""
    id_1 = "test"
    id_2 = "test2"
    MODEL_MANAGER.load_model(
        model_id=id_1,
        local_model_path=Fixtures.get_good_model_path(),
        model_type=Fixtures.get_good_model_type(),
    )
    MODEL_MANAGER.load_model(
        model_id=id_2,
        local_model_path=Fixtures.get_good_model_path(),
        model_type=Fixtures.get_good_model_type(),
    )
    MODEL_MANAGER.unload_all_models()
    assert len(MODEL_MANAGER.loaded_models) == 0


def test_unload_model_not_loaded_response():
    """Test unload model for model not loaded does NOT throw an error"""
    MODEL_MANAGER.unload_model(model_id=random_test_id())


def test_retrieve_model_returns_loaded_model():
    """Test that a loaded model can be retrieved"""
    model_id = random_test_id()
    MODEL_MANAGER.load_model(
        model_id,
        local_model_path=Fixtures.get_good_model_path(),
        model_type=Fixtures.get_good_model_type(),  # eventually we'd like to be determining the type from the model itself...
    )
    model = MODEL_MANAGER.retrieve_model(model_id)
    assert isinstance(model, ModuleBase)
    assert len(MODEL_MANAGER.loaded_models) == 1


def test_retrieve_model_raises_error_for_not_found_model():
    """Test that gRPC NOT_FOUND exception raised when non-existent model retrieved"""
    with pytest.raises(CaikitRuntimeException) as context:
        MODEL_MANAGER.retrieve_model("not-found")
    assert context.value.status_code == grpc.StatusCode.NOT_FOUND


def test_model_size_ok_response():
    """Test if loaded model correctly returns model size"""
    model = random_test_id()
    MODEL_MANAGER.load_model(
        model_id=model,
        local_model_path=Fixtures.get_good_model_path(),
        model_type=Fixtures.get_good_model_type(),
    )
    assert MODEL_MANAGER.get_model_size(model) > 0


def test_model_size_error_not_found_response():
    """Test model size's model does not exist error response"""
    with pytest.raises(CaikitRuntimeException) as context:
        MODEL_MANAGER.get_model_size("no_exist_model")
    assert context.value.status_code == grpc.StatusCode.NOT_FOUND


def test_model_size_error_none_not_found_response():
    """Test model size's model is None error response"""
    with pytest.raises(CaikitRuntimeException) as context:
        MODEL_MANAGER.get_model_size(None)
    assert context.value.status_code == grpc.StatusCode.NOT_FOUND


def test_estimate_model_size_ok_response_on_loaded_model():
    """Test if loaded model correctly returns model size"""
    MODEL_MANAGER.load_model(
        model_id=random_test_id(),
        local_model_path=Fixtures.get_good_model_path(),
        model_type=Fixtures.get_good_model_type(),
    )
    assert (
        MODEL_MANAGER.estimate_model_size(
            "test", Fixtures.get_good_model_path(), Fixtures.get_good_model_type()
        )
        > 0
    )


def test_estimate_model_size_ok_response_on_nonloaded_model():
    """Test if a model that's not loaded, correctly returns predicted model size"""
    assert (
        MODEL_MANAGER.estimate_model_size(
            "test", Fixtures.get_good_model_path(), Fixtures.get_good_model_type()
        )
        > 0
    )


def test_estimate_model_size_by_type():
    """Test that a model's size is estimated differently based on its type"""
    config = get_config().inference_plugin.model_mesh
    assert Fixtures.get_good_model_type() in config.model_size_multipliers

    typed_model_size = MODEL_MANAGER.estimate_model_size(
        "test", Fixtures.get_good_model_path(), Fixtures.get_good_model_type()
    )
    untyped_model_size = MODEL_MANAGER.estimate_model_size(
        "test", Fixtures.get_good_model_path(), "test-not-a-model-type"
    )

    assert typed_model_size > 0
    assert untyped_model_size > 0
    assert typed_model_size != untyped_model_size


def test_estimate_model_size_error_not_found_response():
    """Test if error in predict model size on unknown model path"""
    with pytest.raises(CaikitRuntimeException) as context:
        MODEL_MANAGER.estimate_model_size(
            model_id=random_test_id(),
            local_model_path="no_exist.zip",
            model_type="categories_esa",
        )
    assert context.value.status_code == grpc.StatusCode.NOT_FOUND


def test_model_manager_replicas_with_disk_caching(good_model_path):
    """Test that multiple model manager instances can co-exist and share a set
    of models when local_models_cache is used.

    This test simulates running concurrent replicas of caikit.runtime that are
    logically independent from one another.
    """
    # NOTE: This test requires that the ModelManager class not be a singleton.
    #   To accomplish this, the singleton instance is temporarily removed.
    with TemporaryDirectory() as cache_dir:
        with non_singleton_model_managers(
            2,
            {
                "runtime": {
                    "local_models_dir": cache_dir,
                    "lazy_load_local_models": True,
                    "lazy_load_poll_period_seconds": 0,
                },
            },
            "merge",
        ) as managers:
            # Make two standalone manager instances
            manager_one, manager_two = managers
            assert manager_two is not manager_one
            assert manager_one._local_models_dir == cache_dir
            assert manager_two._local_models_dir == cache_dir

            # Trying to retrieve a model that is not loaded in either and is
            # not saved in the cache dir should yield NOT_FOUND
            model_id = random_test_id()
            with pytest.raises(CaikitRuntimeException) as context:
                manager_one.retrieve_model(model_id)
            assert context.value.status_code == grpc.StatusCode.NOT_FOUND
            with pytest.raises(CaikitRuntimeException) as context:
                manager_two.retrieve_model(model_id)
            assert context.value.status_code == grpc.StatusCode.NOT_FOUND

            # Place a saved model into the cache dir manually
            model_one_path = os.path.join(cache_dir, model_id)
            shutil.copytree(good_model_path, model_one_path)

            # Retrieve the model by ID and ensure it's available now
            model_one = manager_one.retrieve_model(model_id)
            assert isinstance(model_one, ModuleBase)
            assert (
                manager_one.loaded_models.get(model_id)
                and manager_one.loaded_models[model_id].model() is model_one
            )

            # Make sure the same model is available on the second manager
            assert manager_two.loaded_models.get(model_id) is None
            assert manager_two.retrieve_model(model_id)
            assert manager_two.loaded_models.get(model_id)

            # Remove the model from the local dir and trigger both managers to
            # sync their local model dirs to make sure it gets removed
            shutil.rmtree(model_one_path)
            manager_one.sync_local_models(wait=True)
            manager_two.sync_local_models(wait=True)
            with pytest.raises(CaikitRuntimeException) as context:
                manager_one.retrieve_model(model_id)
            with pytest.raises(CaikitRuntimeException) as context:
                manager_two.retrieve_model(model_id)


def test_model_manager_disk_caching_periodic_sync(good_model_path):
    """Make sure that when using disk caching, the manager periodically syncs
    its loaded models based on their presence in the cache
    """
    purge_period = 0.002
    with TemporaryDirectory() as cache_dir:
        with non_singleton_model_managers(
            2,
            {
                "runtime": {
                    "local_models_dir": cache_dir,
                    "lazy_load_local_models": True,
                    "lazy_load_poll_period_seconds": purge_period,
                },
            },
            "merge",
        ) as managers:
            manager_one, manager_two = managers

            # Load the model on the first manager and get it loaded into the
            # the first by fetching it
            model_id = random_test_id()
            model_cache_path = os.path.join(cache_dir, model_id)
            assert not os.path.exists(model_cache_path)
            shutil.copytree(good_model_path, model_cache_path)
            assert manager_one.retrieve_model(model_id)
            assert model_id in manager_one.loaded_models

            # Wait for the sync period and make sure it's available in the
            # second manager
            start = time.time()
            mgr_two_loaded = False
            while (time.time() - start) < (purge_period * 1000):
                time.sleep(purge_period)
                if model_id in manager_two.loaded_models:
                    mgr_two_loaded = True
                    break
            assert mgr_two_loaded

            # Remove the model from disk and wait to ensure that the model gets
            # unloaded
            shutil.rmtree(model_cache_path)
            start = time.time()
            mgr_one_unloaded = False
            mgr_two_unloaded = False
            while (time.time() - start) < (purge_period * 1000):
                if model_id not in manager_one.loaded_models:
                    mgr_one_unloaded = True
                if model_id not in manager_two.loaded_models:
                    mgr_two_unloaded = True
                if mgr_one_unloaded and mgr_two_unloaded:
                    break
            assert mgr_one_unloaded and mgr_two_unloaded


def test_periodic_sync_without_loading(good_model_path):
    """Test that periodic synchronization of local_models_dir can proceed
    without loading new models found there (unload only with lazy loading)
    """
    purge_period = 0.001
    with TemporaryDirectory() as cache_dir:
        # Copy the good model to the cache dir before starting the manager
        model_id = random_test_id()
        model_cache_path = os.path.join(cache_dir, model_id)
        shutil.copytree(good_model_path, model_cache_path)

        # Start the manager without loading new local models
        with non_singleton_model_managers(
            1,
            {
                "runtime": {
                    "load_new_local_models": False,
                    "local_models_dir": cache_dir,
                    "lazy_load_local_models": True,
                    "lazy_load_poll_period_seconds": purge_period,
                    # NOTE: There won't be any initial model loads, but this
                    #   ensures that if there were, they would happen
                    #   synchronously during __init__
                    "wait_for_initial_model_loads": True,
                },
            },
            "merge",
        ) as managers:
            manager = managers[0]

            # The model doesn't load at boot
            assert model_id not in manager.loaded_models

            # Wait for the purge period to run and make sure it's still not
            # loaded
            manager._lazy_sync_timer.join()
            assert model_id not in manager.loaded_models

            # Explicitly retrieve the model and make sure it _does_ lazy load
            model = manager.retrieve_model(model_id)
            assert model
            assert model_id in manager.loaded_models

            # Remove the file from local_models_dir and make sure it gets purged
            shutil.rmtree(model_cache_path)
            manager._lazy_sync_timer.join()
            assert model_id not in manager.loaded_models


def test_lazy_load_of_large_model(good_model_path):
    """Test that a large model that is actively being written to disk is not incorrectly loaded
    too soon by the lazy loading poll
    """
    with TemporaryDirectory() as cache_dir:
        with non_singleton_model_managers(
            1,
            {
                "runtime": {
                    "local_models_dir": cache_dir,
                    "lazy_load_local_models": True,
                    "lazy_load_poll_period_seconds": 0,
                },
            },
            "merge",
        ) as managers:
            manager = managers[0]

            # Start with a valid model
            model_name = os.path.basename(good_model_path)
            model_cache_path = os.path.join(cache_dir, model_name)
            assert not os.path.exists(model_cache_path)
            shutil.copytree(good_model_path, model_cache_path)

            # Then kick off a thread that will start writing a large file inside this model dir.
            # This simulates uploading a large model artifact
            def write_big_file(path: str, stop_event: threading.Event):
                big_file = os.path.join(path, "big_model_artifact.txt")
                with open(big_file, "w") as bf:
                    while not stop_event.is_set():
                        bf.write("This is a big file\n" * 1000)

            stop_write_event = threading.Event()
            writer_thread = threading.Thread(
                target=write_big_file, args=(model_cache_path, stop_write_event)
            )
            writer_thread.start()

            try:
                # Trigger the periodic sync and make sure the model is NOT loaded
                assert model_name not in manager.loaded_models
                manager.sync_local_models(wait=True)
                assert model_name not in manager.loaded_models

                # Stop the model writing thread (Finish the model upload)
                stop_write_event.set()
                writer_thread.join()

                # Re-trigger the sync and make sure the model is loaded this time
                manager.sync_local_models(wait=True)
                assert model_name in manager.loaded_models

            finally:
                stop_write_event.set()
                writer_thread.join()


def test_nested_local_model_load_unload(good_model_path):
    """Test that a model can be loaded in a subdirectory of the local_models_dir
    and that the periodic sync does not unload the model.
    """
    with TemporaryDirectory() as cache_dir:
        with non_singleton_model_managers(
            1,
            {
                "runtime": {
                    "local_models_dir": cache_dir,
                    "lazy_load_local_models": True,
                    "lazy_load_poll_period_seconds": 0,
                },
            },
            "merge",
        ) as managers:
            manager = managers[0]

            # Copy the model into a nested model directory
            model_name = os.path.join("parent", os.path.basename(good_model_path))
            model_cache_path = os.path.join(cache_dir, model_name)
            assert not os.path.exists(model_cache_path)
            shutil.copytree(good_model_path, model_cache_path)

            # Trigger the periodic sync and make sure the model is NOT loaded
            assert model_name not in manager.loaded_models
            manager.sync_local_models(wait=True)
            assert model_name not in manager.loaded_models

            # Explicitly ask to load the nested model name to trigger the lazy
            # load
            model = manager.retrieve_model(model_name)
            assert model
            assert model_name in manager.loaded_models

            # Re-trigger the sync and make sure the model does not get unloaded
            manager.sync_local_models(wait=True)
            assert model_name in manager.loaded_models


def test_model_unload_race(good_model_path):
    """Test that if a model gets unloaded _while_ it's actively being loaded
    (before retrieve_model completes, but after load_model completes), no
    exception is raised.
    """
    with TemporaryDirectory() as cache_dir:
        with non_singleton_model_managers(
            1,
            {
                "runtime": {
                    "local_models_dir": cache_dir,
                    "lazy_load_local_models": True,
                    "lazy_load_poll_period_seconds": 0,
                },
            },
            "merge",
        ) as managers:
            manager = managers[0]

            # Copy the model to the local_models_dir
            model_id = random_test_id()
            model_cache_path = os.path.join(cache_dir, model_id)
            shutil.copytree(good_model_path, model_cache_path)

            # Patch the manager's load_model to immediately unload the model
            orig_load_model = manager.load_model

            def load_and_unload_model(self, model_id: str, *args, **kwargs):
                res = orig_load_model(model_id, *args, **kwargs)
                manager.unload_model(model_id)
                return res

            with patch.object(manager.__class__, "load_model", load_and_unload_model):

                # Retrieve the model and make sure there's no error
                assert manager.retrieve_model(model_id)
                assert model_id not in manager.loaded_models


def test_load_local_model_deleted_dir():
    """Make sure losing the local_models_dir out from under a running manager
    doesn't kill the whole thing
    """
    with TemporaryDirectory() as tempdir:
        cache_dir = os.path.join(tempdir, "cache_dir")
        os.makedirs(cache_dir)
        with non_singleton_model_managers(
            1,
            {
                "runtime": {
                    "local_models_dir": cache_dir,
                    "lazy_load_local_models": True,
                    "lazy_load_poll_period_seconds": 0.001,
                },
            },
            "merge",
        ) as managers:
            manager = managers[0]

            # Make sure the timer started
            assert manager._lazy_sync_timer is not None

            # Delete the cache dir and force a sync
            shutil.rmtree(cache_dir)
            while True:
                try:
                    os.listdir(cache_dir)
                except FileNotFoundError:
                    break
            manager.sync_local_models(wait=True)

            # Make sure the timer is removed
            assert manager._lazy_sync_timer is None


def test_load_local_model_deleted_dir():
    """Make sure bad models in local_models_dir at boot don't cause exceptions"""
    with TemporaryDirectory() as cache_dir:
        model_id = random_test_id()
        model_cache_path = os.path.join(cache_dir, model_id)
        os.makedirs(model_cache_path)
        with non_singleton_model_managers(
            1,
            {
                "runtime": {
                    "local_models_dir": cache_dir,
                    "lazy_load_local_models": False,
                },
            },
            "merge",
        ) as managers:
            manager = managers[0]
            assert not manager.loaded_models


def test_lazy_load_ephemeral_model():
    """Make sure an ephemeral model (not on disk) can be lazy loaded if the
    right finder configuration is present to load it without hitting disk.
    """
    with TemporaryDirectory() as cache_dir:
        with non_singleton_model_managers(
            1,
            {
                "model_management": {"finders": {"default": {"type": TestFinder.name}}},
                "runtime": {
                    "local_models_dir": cache_dir,
                    "lazy_load_local_models": True,
                },
            },
            "merge",
        ) as managers:
            manager = managers[0]
            model_id = random_test_id()
            model = manager.retrieve_model(model_id)
            assert model
            assert cache_dir not in manager.loaded_models[model_id].path()

            # Make sure the model does not get unloaded on sync
            manager._local_models_dir_sync(wait=True)
            assert model_id in manager.loaded_models


def test_deploy_undeploy_model(deploy_good_model_files):
    """Test that a model can be deployed by copying to the local models dir"""
    with TemporaryDirectory() as cache_dir:
        with non_singleton_model_managers(
            1,
            {
                "runtime": {
                    "local_models_dir": cache_dir,
                    "lazy_load_local_models": True,
                    "lazy_load_poll_period_seconds": 0,
                },
            },
            "merge",
        ) as managers:
            manager = managers[0]
            model_name = "my-model"

            # Make sure model is not currently loaded
            with pytest.raises(CaikitRuntimeException) as excinfo:
                manager.retrieve_model(model_name)
            assert excinfo.value.status_code == grpc.StatusCode.NOT_FOUND

            # Do the deploy (pass wait through to load)
            loaded_model = manager.deploy_model(
                model_name, deploy_good_model_files, wait=True
            )
            assert loaded_model
            assert loaded_model.loaded

            # Make sure model can be retrieved and exists in the local models dir
            assert manager.retrieve_model(model_name)
            assert os.path.isdir(os.path.join(cache_dir, model_name))

            # Make sure model cannot be deployed over
            with pytest.raises(CaikitRuntimeException) as excinfo:
                manager.deploy_model(model_name, deploy_good_model_files)
            assert excinfo.value.status_code == grpc.StatusCode.ALREADY_EXISTS

            # Undeploy the model
            manager.undeploy_model(model_name)

            # Make sure the model is not loaded anymore and was removed from
            # local models dir
            with pytest.raises(CaikitRuntimeException) as excinfo:
                manager.retrieve_model(model_name)
            assert excinfo.value.status_code == grpc.StatusCode.NOT_FOUND
            assert not os.path.exists(os.path.join(cache_dir, model_name))


@pytest.mark.parametrize(
    ["invalid_fname", "expected_reason"],
    [
        ("", "Got whitespace-only model file name"),
        ("\t\n  ", "Got whitespace-only model file name"),
        ("/foo/bar.txt", "Cannot use absolute paths for model files"),
    ],
)
def test_deploy_invalid_files(invalid_fname, expected_reason):
    """Test that various flavors of invalid model names are not supported"""
    with TemporaryDirectory() as cache_dir:
        with non_singleton_model_managers(
            1,
            {
                "runtime": {
                    "local_models_dir": cache_dir,
                    "lazy_load_local_models": True,
                    "lazy_load_poll_period_seconds": 0,
                },
            },
            "merge",
        ) as managers:
            manager = managers[0]
            with pytest.raises(
                CaikitRuntimeException, match=expected_reason
            ) as excinfo:
                manager.deploy_model("bad-model", {invalid_fname: b"asdf"})
            assert excinfo.value.status_code == grpc.StatusCode.INVALID_ARGUMENT


def test_deploy_with_nested_files(deploy_good_model_files):
    """Make sure models with nested directories can be deployed"""
    with TemporaryDirectory() as cache_dir:
        with non_singleton_model_managers(
            1,
            {
                "runtime": {
                    "local_models_dir": cache_dir,
                    "lazy_load_local_models": True,
                    "lazy_load_poll_period_seconds": 0,
                },
            },
            "merge",
        ) as managers:
            manager = managers[0]
            model_name = "my-model"

            # Read the model files and deploy
            nested_dir = os.path.join("nested", "twice")
            nested_fname = "foo.txt"
            deploy_good_model_files[os.path.join(nested_dir, nested_fname)] = b"foo"
            loaded_model = manager.deploy_model(
                model_name, deploy_good_model_files, wait=True
            )
            assert loaded_model

            # Make sure the nested file structure was set up correctly
            local_nested_dir = os.path.join(cache_dir, model_name, nested_dir)
            assert os.path.isdir(local_nested_dir)
            assert os.path.exists(os.path.join(local_nested_dir, nested_fname))


def test_deploy_invalid_permissions(deploy_good_model_files):
    """Make sure that an error is raised if attempting to deploy when writing to
    local_models_dir is denied
    """
    with TemporaryDirectory() as cache_dir:
        local_models_dir = os.path.join(cache_dir, "local_models")
        os.makedirs(local_models_dir)
        os.chmod(local_models_dir, 0o600)
        try:
            with non_singleton_model_managers(
                1,
                {
                    "runtime": {
                        "local_models_dir": local_models_dir,
                        "lazy_load_local_models": True,
                        "lazy_load_poll_period_seconds": 0,
                    },
                },
                "merge",
            ) as managers:
                manager = managers[0]
                model_name = "my-model"

                # Make sure the deploy fails with a permission error
                with pytest.raises(CaikitRuntimeException) as excinfo:
                    manager.deploy_model(model_name, deploy_good_model_files, wait=True)
                assert excinfo.value.status_code == grpc.StatusCode.FAILED_PRECONDITION

        finally:
            os.chmod(local_models_dir, 0o777)
            shutil.rmtree(local_models_dir)


def test_undeploy_unkonwn_model():
    """Make sure that attempting to undeploy an unknown model raises NOT_FOUND"""
    with TemporaryDirectory() as cache_dir:
        with non_singleton_model_managers(
            1,
            {
                "runtime": {
                    "local_models_dir": cache_dir,
                    "lazy_load_local_models": True,
                    "lazy_load_poll_period_seconds": 0,
                },
            },
            "merge",
        ) as managers:
            manager = managers[0]
            with pytest.raises(CaikitRuntimeException) as excinfo:
                manager.undeploy_model("foobar")
            assert excinfo.value.status_code == grpc.StatusCode.NOT_FOUND


def test_undeploy_unloaded_model(deploy_good_model_files):
    """If running with replicas and a shared local_models_dir, the replica that
    gets the undeploy request may not have loaded the model into memory yet.
    This tests that the model gets properly removed from local_models_dir, even
    if not yet loaded.
    """
    with TemporaryDirectory() as cache_dir:
        with non_singleton_model_managers(
            1,
            {
                "runtime": {
                    "local_models_dir": cache_dir,
                    "lazy_load_local_models": True,
                    "lazy_load_poll_period_seconds": 0,
                },
            },
            "merge",
        ) as managers:
            manager = managers[0]

            # Copy files to the local_models_dir
            model_name = "foobar"
            model_dir = os.path.join(cache_dir, model_name)
            os.makedirs(model_dir)
            for fname, data in deploy_good_model_files.items():
                with open(os.path.join(model_dir, fname), "wb") as handle:
                    handle.write(data)

            # Make sure the undeploy completes successfully
            assert model_name not in manager.loaded_models
            assert os.path.exists(model_dir)
            manager.undeploy_model(model_name)
            assert model_name not in manager.loaded_models
            assert not os.path.exists(model_dir)


# ****************************** Unit Tests ****************************** #
# These tests patch in mocks for the manager's dependencies, to test its code in isolation


def test_load_model():
    """Test to make sure that given valid input, the model manager gives a happy response
    when we tried to load in a model (model size > 0 or 0 if the model size will be computed
    at a later time)."""
    mock_loader = MagicMock()
    mock_sizer = MagicMock()
    model_id = random_test_id()
    expected_model_size = 1234

    with patch.object(MODEL_MANAGER, "model_loader", mock_loader):
        with patch.object(MODEL_MANAGER, "model_sizer", mock_sizer):
            loaded_model = LoadedModel()
            loaded_model._model = "something"
            mock_loader.load_model.return_value = loaded_model
            mock_sizer.get_model_size.return_value = expected_model_size

            model_size = MODEL_MANAGER.load_model(
                model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE
            ).size()
            assert expected_model_size == model_size
            mock_loader.load_model.assert_called_once()
            call_args = mock_loader.load_model.call_args
            assert call_args.args == (
                model_id,
                ANY_MODEL_PATH,
                ANY_MODEL_TYPE,
            )
            assert "fail_callback" in call_args.kwargs
            mock_sizer.get_model_size.assert_called_once_with(
                model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE
            )


def test_load_model_throws_if_the_model_loader_throws():
    """Test load model's model does not exist when the loader throws"""
    mock_loader = MagicMock()
    with patch.object(MODEL_MANAGER, "model_loader", mock_loader):
        mock_loader.load_model.side_effect = CaikitRuntimeException(
            grpc.StatusCode.NOT_FOUND, "test any not found exception"
        )

        with pytest.raises(CaikitRuntimeException) as context:
            MODEL_MANAGER.load_model(random_test_id(), ANY_MODEL_PATH, ANY_MODEL_TYPE)

        assert context.value.status_code == grpc.StatusCode.NOT_FOUND
        assert len(MODEL_MANAGER.loaded_models) == 0


def test_retrieve_model_returns_the_module_from_the_model_loader():
    """Test that a loaded model can be retrieved"""
    model_id = random_test_id()
    expected_module = "this is definitely a stub module"
    mock_sizer = MagicMock()
    mock_loader = MagicMock()

    with patch.object(MODEL_MANAGER, "model_loader", mock_loader):
        with patch.object(MODEL_MANAGER, "model_sizer", mock_sizer):
            mock_sizer.get_model_size.return_value = 1
            model_future = Future()
            model_future.result = lambda *_, **__: expected_module
            model_future_factory = lambda: model_future
            mock_loader.load_model.return_value = (
                LoadedModel.Builder()
                .model_future_factory(model_future_factory)
                .id("foo")
                .type("bar")
                .build()
            )
            MODEL_MANAGER.load_model(model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE)

            model = MODEL_MANAGER.retrieve_model(model_id)
            assert expected_module == model


def test_unload_partially_loaded():
    """Make sure that when unloading a model that is still loading, the load
    completes correctly
    """
    # Set up a "slow load" that we can use to ensure that the loading has
    # completed successfully
    slow_loader = SlowLoader()
    pool = ThreadPoolExecutor()
    model_future_factory = partial(pool.submit, slow_loader.load)

    model_id = random_test_id()
    mock_sizer = MagicMock()
    mock_loader = MagicMock()
    with patch.object(MODEL_MANAGER, "model_loader", mock_loader):
        with patch.object(MODEL_MANAGER, "model_sizer", mock_sizer):
            mock_sizer.get_model_size.return_value = 1
            mock_loader.load_model.return_value = (
                LoadedModel.Builder()
                .model_future_factory(model_future_factory)
                .id("foo")
                .type("bar")
                .build()
            )
            MODEL_MANAGER.load_model(
                model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE, wait=False
            )

            # Start the unload assertion that will block on loading
            unload_future = pool.submit(MODEL_MANAGER.unload_model, model_id)

            # Unblock the load
            assert not slow_loader.done_loading()
            slow_loader.unblock_load()

            # Make sure unload completes and the model finished loading
            unload_future.result()
            assert slow_loader.done_loading()


def test_unload_unexpected_error_loaded():
    """Make sure that when unloading a model that is still loading, errors when
    waiting for the load to complete are handled
    """
    # Set up a "slow load" that we can use to ensure that the loading has
    # completed successfully
    slow_loader = SlowLoader(RuntimeError("yikes"))
    pool = ThreadPoolExecutor()
    model_future_factory = partial(pool.submit, slow_loader.load)

    model_id = random_test_id()
    mock_sizer = MagicMock()
    mock_loader = MagicMock()
    with patch.object(MODEL_MANAGER, "model_loader", mock_loader):
        with patch.object(MODEL_MANAGER, "model_sizer", mock_sizer):
            mock_sizer.get_model_size.return_value = 1
            mock_loader.load_model.return_value = (
                LoadedModel.Builder()
                .model_future_factory(model_future_factory)
                .id("foo")
                .type("bar")
                .build()
            )
            MODEL_MANAGER.load_model(
                model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE, wait=False
            )

            # Start the unload assertion that will block on loading
            unload_future = pool.submit(MODEL_MANAGER.unload_model, model_id)

            # Unblock the load
            assert not slow_loader.done_loading()
            slow_loader.unblock_load()

            # Make sure unload completes and the model finished loading
            with pytest.raises(CaikitRuntimeException) as context:
                unload_future.result()
            assert context.value.status_code == grpc.StatusCode.INTERNAL
            assert slow_loader.done_loading()


def test_reload_partially_loaded():
    """Make sure that attempting to reload a model that has already started to
    load simply returns the existing LoadedModel
    """
    # Set up a "slow load" that we can use to ensure that the loading has
    # completed successfully
    slow_loader = SlowLoader()
    pool = ThreadPoolExecutor()
    model_future_factory = partial(pool.submit, slow_loader.load)

    model_id = random_test_id()
    mock_sizer = MagicMock()
    mock_loader = MagicMock()
    special_model_size = 123321
    with patch.object(MODEL_MANAGER, "model_loader", mock_loader):
        with patch.object(MODEL_MANAGER, "model_sizer", mock_sizer):
            mock_sizer.get_model_size.return_value = special_model_size
            loaded_model = (
                LoadedModel.Builder()
                .model_future_factory(model_future_factory)
                .id("foo")
                .type("bar")
                .build()
            )
            mock_loader.load_model.return_value = loaded_model
            model_size = MODEL_MANAGER.load_model(
                model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE, wait=False
            ).size()
            assert model_size == special_model_size
            assert (
                MODEL_MANAGER.load_model(
                    model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE, wait=False
                ).size()
                == special_model_size
            )

            # Unblock the load
            assert not slow_loader.done_loading()
            slow_loader.unblock_load()
            loaded_model.wait()
            assert slow_loader.done_loading()


def test_get_model_size_returns_size_from_model_sizer():
    """Test that loading a model stores the size from the ModelSizer"""
    mock_loader = MagicMock()
    mock_sizer = MagicMock()
    expected_model_size = 1234
    model_id = random_test_id()

    with patch.object(MODEL_MANAGER, "model_loader", mock_loader):
        with patch.object(MODEL_MANAGER, "model_sizer", mock_sizer):
            loaded_model = LoadedModel()
            loaded_model._model = "something"
            mock_loader.load_model.return_value = loaded_model
            mock_sizer.get_model_size.return_value = expected_model_size

            MODEL_MANAGER.load_model(model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE)

            model_size = MODEL_MANAGER.get_model_size(model_id)
            assert expected_model_size == model_size


def test_estimate_model_size_returns_size_from_model_sizer():
    """Test that estimating a model size uses the ModelSizer"""
    mock_sizer = MagicMock()
    expected_model_size = 5678
    model_id = random_test_id()

    with patch.object(MODEL_MANAGER, "model_sizer", mock_sizer):
        mock_sizer.get_model_size.return_value = expected_model_size
        model_size = MODEL_MANAGER.estimate_model_size(
            model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE
        )
        assert expected_model_size == model_size


def test_estimate_model_size_throws_if_model_sizer_throws():
    """Test that estimating a model size uses the ModelSizer"""
    mock_sizer = MagicMock()
    model_id = random_test_id()

    with patch.object(MODEL_MANAGER, "model_sizer", mock_sizer):
        mock_sizer.get_model_size.side_effect = CaikitRuntimeException(
            grpc.StatusCode.UNAVAILABLE, "test-any-exception"
        )
        with pytest.raises(CaikitRuntimeException) as context:
            MODEL_MANAGER.estimate_model_size(model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE)
        assert context.value.status_code == grpc.StatusCode.UNAVAILABLE


def test_periodic_sync_handles_errors():
    """Test that any exception raised during syncing local models is handled
    without terminating the polling loop
    """

    class SecretException(Exception):
        pass

    with TemporaryDirectory() as cache_dir:
        with non_singleton_model_managers(
            1,
            {
                "runtime": {
                    "local_models_dir": cache_dir,
                    "lazy_load_local_models": True,
                },
            },
            "merge",
        ) as managers:
            manager = managers[0]
            with patch.object(manager, "_local_models_dir_sync") as mock_sync:
                mock_sync.side_effect = SecretException()
                assert manager._lazy_sync_timer is not None
                manager.sync_local_models(True)
                mock_sync.assert_called_once()
                assert manager._lazy_sync_timer is not None


def test_periodic_sync_handles_temporary_errors():
    """Test that models loaded with the periodic sync can retry if the initial
    load operation fails
    """
    with TemporaryDirectory() as cache_dir:
        with non_singleton_model_managers(
            1,
            {
                "runtime": {
                    "local_models_dir": cache_dir,
                    "lazy_load_local_models": True,
                    "lazy_load_retries": 1,
                },
            },
            "merge",
        ) as managers:
            manager = managers[0]
            flakey_loader = TempFailWrapper(
                manager.model_loader.load_module_instance,
                num_failures=1,
                exc=CaikitRuntimeException(grpc.StatusCode.INTERNAL, "Dang"),
            )
            with patch.object(
                manager.model_loader,
                "load_module_instance",
                flakey_loader,
            ):
                assert manager._lazy_sync_timer is not None
                model_path = Fixtures.get_good_model_path()
                model_name = os.path.basename(model_path)
                shutil.copytree(model_path, os.path.join(cache_dir, model_name))
                manager.sync_local_models(True)
                assert manager._lazy_sync_timer is not None
                model = manager.retrieve_model(model_name)
                assert model


def test_lazy_load_handles_temporary_errors():
    """Test that a lazy load without a periodic sync correctly retries failed
    loads
    """
    with TemporaryDirectory() as cache_dir:
        with non_singleton_model_managers(
            1,
            {
                "runtime": {
                    "local_models_dir": cache_dir,
                    "lazy_load_local_models": True,
                    "lazy_load_poll_period_seconds": 0,
                    "lazy_load_retries": 1,
                },
            },
            "merge",
        ) as managers:
            manager = managers[0]
            flakey_loader = TempFailWrapper(
                manager.model_loader.load_module_instance,
                num_failures=1,
                exc=CaikitRuntimeException(grpc.StatusCode.INTERNAL, "Dang"),
            )
            with patch.object(
                manager.model_loader,
                "load_module_instance",
                flakey_loader,
            ):
                assert manager._lazy_sync_timer is None
                model_path = Fixtures.get_good_model_path()
                model_name = os.path.basename(model_path)
                shutil.copytree(model_path, os.path.join(cache_dir, model_name))
                assert manager._lazy_sync_timer is None
                model = manager.retrieve_model(model_name)
                assert model


def test_lazy_load_true_local_models_dir_valid():
    """When lazy_load_local_models is True and local_models_dir exists.
    Check that the local_models_dir is pointing to the correct location
    """

    with TemporaryDirectory() as cache_dir:

        ModelManager._ModelManager__instance = None
        with temp_config(
            {
                "runtime": {
                    "local_models_dir": cache_dir,
                    "lazy_load_local_models": True,
                }
            },
            merge_strategy="merge",
        ):
            MODEL_MANAGER = ModelManager()
            assert len(MODEL_MANAGER.loaded_models) == 0
            assert MODEL_MANAGER._local_models_dir == cache_dir


def test_lazy_load_true_local_models_dir_invalid():
    """When lazy_load_local_models is True and local_models_dir does not exist.
    Raise ValueError with an appropriate message
    """

    with TemporaryDirectory() as cache_dir:

        with pytest.raises(
            ValueError,
            match=(
                "runtime.local_models_dir must be a valid path"
                " if set with runtime.lazy_load_local_models. "
                "Provided path: invalid"
            ),
        ):

            ModelManager._ModelManager__instance = None
            with temp_config(
                {
                    "runtime": {
                        "local_models_dir": "invalid",
                        "lazy_load_local_models": True,
                    }
                },
                merge_strategy="merge",
            ):
                MODEL_MANAGER = ModelManager()


def test_lazy_load_true_local_models_dir_none():
    """When lazy_load_local_models is True and local_models_dir is not set in the config.
    Raise ValueError with an appropriate message
    """

    with TemporaryDirectory() as cache_dir:

        with pytest.raises(
            ValueError,
            match=(
                "runtime.local_models_dir must be set"
                " if using runtime.lazy_load_local_models. "
            ),
        ):

            ModelManager._ModelManager__instance = None
            with temp_config(
                {
                    "runtime": {
                        "local_models_dir": None,
                        "lazy_load_local_models": True,
                    }
                },
                merge_strategy="merge",
            ):
                MODEL_MANAGER = ModelManager()


def test_lazy_load_false_local_models_dir_valid():
    """When lazy_load_local_models is False and local_models_dir exists.
    Check that the local_models_dir is pointing to the correct location
    """

    with TemporaryDirectory() as cache_dir:

        ModelManager._ModelManager__instance = None
        with temp_config(
            {
                "runtime": {
                    "local_models_dir": cache_dir,
                    "lazy_load_local_models": False,
                }
            },
            merge_strategy="merge",
        ):
            MODEL_MANAGER = ModelManager()
            assert len(MODEL_MANAGER.loaded_models) == 0
            assert MODEL_MANAGER._local_models_dir == cache_dir


def test_lazy_load_false_local_models_dir_invalid():
    """When lazy_load_local_models is False and local_models_dir does not exist.
    Check that the local_models_dir is False / Empty
    """

    with TemporaryDirectory() as cache_dir:

        ModelManager._ModelManager__instance = None
        with temp_config(
            {
                "runtime": {
                    "local_models_dir": "",
                    "lazy_load_local_models": False,
                }
            },
            merge_strategy="merge",
        ):
            MODEL_MANAGER = ModelManager()
            assert len(MODEL_MANAGER.loaded_models) == 0
            assert not MODEL_MANAGER._local_models_dir


class NoModelFinder(ModelFinderBase):
    name = "NOMODEL"

    def __init__(self, config: Config, instance_name: str):
        super().__init__(config, instance_name)

    def find_model(self, model_path: str, **kwargs) -> ModuleConfig:
        raise FileNotFoundError(f"Unable to find model {model_path}")


def test_load_model_custom_finder():
    """Test to ensure loading model works with custom finder"""
    bad_finder = NoModelFinder(aconfig.Config({}), "bad_instance")

    model_id = random_test_id()
    with pytest.raises(CaikitRuntimeException) as exp:
        MODEL_MANAGER.load_model(
            model_id=model_id,
            local_model_path=Fixtures.get_good_model_path(),
            model_type=Fixtures.get_good_model_type(),
            finder=bad_finder,
        )
    assert exp.value.status_code == grpc.StatusCode.NOT_FOUND


class CustomParamInitializer(LocalModelInitializer):
    name = "CUSTOMPARAM"

    def init(self, model_config: ModuleConfig, **kwargs) -> ModuleBase:
        module = super().init(model_config, **kwargs)
        module.custom_param = True
        return module


def test_load_model_custom_initializer():
    """Test to ensure loading model works with custom initializer"""

    custom_param_initializer = CustomParamInitializer(
        aconfig.Config({}), "custom_param"
    )
    model_id = random_test_id()
    model = MODEL_MANAGER.load_model(
        model_id=model_id,
        local_model_path=Fixtures.get_good_model_path(),
        model_type=Fixtures.get_good_model_type(),
        initializer=custom_param_initializer,
    ).model()
    assert model
    assert model.custom_param
