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
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
import os
import shutil
import threading
import time

# Third Party
import grpc
import pytest

# Local
from caikit import get_config
from caikit.core.modules import ModuleBase
from caikit.runtime.model_management.loaded_model import LoadedModel
from caikit.runtime.model_management.model_manager import ModelManager
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from caikit.runtime.utils.import_util import get_dynamic_module
from tests.conftest import random_test_id, temp_config
from tests.fixtures import Fixtures

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
    )
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
    )
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


def test_model_manager_loads_local_models_on_init():
    with TemporaryDirectory() as tempdir:
        shutil.copytree(Fixtures.get_good_model_path(), os.path.join(tempdir, "model1"))
        shutil.copy(
            Fixtures.get_good_model_archive_path(),
            os.path.join(tempdir, "model2.zip"),
        )
        ModelManager._ModelManager__instance = None
        with temp_config(
            {"runtime": {"local_models_dir": tempdir}}, merge_strategy="merge"
        ):
            MODEL_MANAGER = ModelManager()

            assert len(MODEL_MANAGER.loaded_models) == 2
            assert "model1" in MODEL_MANAGER.loaded_models.keys()
            assert "model2.zip" in MODEL_MANAGER.loaded_models.keys()
            assert "model-does-not-exist.zip" not in MODEL_MANAGER.loaded_models.keys()


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
    purge_period = 0.001
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
            )
            assert expected_model_size == model_size
            mock_loader.load_model.assert_called_once()
            call_args = mock_loader.load_model.call_args
            assert call_args.args == (
                model_id,
                ANY_MODEL_PATH,
                ANY_MODEL_TYPE,
            )
            assert call_args.kwargs["aborter"] is None
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
            mock_loader.load_model.return_value = (
                LoadedModel.Builder()
                .model_future(model_future)
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
    model_future = pool.submit(slow_loader.load)

    model_id = random_test_id()
    mock_sizer = MagicMock()
    mock_loader = MagicMock()
    with patch.object(MODEL_MANAGER, "model_loader", mock_loader):
        with patch.object(MODEL_MANAGER, "model_sizer", mock_sizer):
            mock_sizer.get_model_size.return_value = 1
            mock_loader.load_model.return_value = (
                LoadedModel.Builder()
                .model_future(model_future)
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
    model_future = pool.submit(slow_loader.load)

    model_id = random_test_id()
    mock_sizer = MagicMock()
    mock_loader = MagicMock()
    with patch.object(MODEL_MANAGER, "model_loader", mock_loader):
        with patch.object(MODEL_MANAGER, "model_sizer", mock_sizer):
            mock_sizer.get_model_size.return_value = 1
            mock_loader.load_model.return_value = (
                LoadedModel.Builder()
                .model_future(model_future)
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
    model_future = pool.submit(slow_loader.load)

    model_id = random_test_id()
    mock_sizer = MagicMock()
    mock_loader = MagicMock()
    special_model_size = 123321
    with patch.object(MODEL_MANAGER, "model_loader", mock_loader):
        with patch.object(MODEL_MANAGER, "model_sizer", mock_sizer):
            mock_sizer.get_model_size.return_value = special_model_size
            mock_loader.load_model.return_value = (
                LoadedModel.Builder()
                .model_future(model_future)
                .id("foo")
                .type("bar")
                .build()
            )
            model_size = MODEL_MANAGER.load_model(
                model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE, wait=False
            )
            assert model_size == special_model_size
            assert (
                MODEL_MANAGER.load_model(
                    model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE, wait=False
                )
                == special_model_size
            )

            # Unblock the load
            assert not slow_loader.done_loading()
            slow_loader.unblock_load()
            model_future.result()
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
