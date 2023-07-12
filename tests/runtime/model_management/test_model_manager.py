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
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
import copy
import os
import shutil

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


def test_load_local_models():
    with TemporaryDirectory() as tempdir:
        shutil.copytree(Fixtures.get_good_model_path(), os.path.join(tempdir, "model1"))
        shutil.copy(
            Fixtures.get_good_model_archive_path(),
            os.path.join(tempdir, "model2.zip"),
        )

        MODEL_MANAGER.load_local_models(tempdir)
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


def test_model_manager_raises_if_all_local_models_fail_to_load():
    with TemporaryDirectory() as tempdir:
        shutil.copy(
            Fixtures.get_bad_model_archive_path(), os.path.join(tempdir, "model1")
        )
        shutil.copy(
            Fixtures.get_invalid_model_archive_path(),
            os.path.join(tempdir, "model2.zip"),
        )
        with pytest.raises(CaikitRuntimeException) as ctx:
            MODEL_MANAGER.load_local_models(tempdir)
        assert grpc.StatusCode.INTERNAL == ctx.value.status_code


def test_load_model_error_response():
    """Test load model's model does not exist when the loader throws"""
    with pytest.raises(CaikitRuntimeException) as context:
        MODEL_MANAGER.load_model(
            model_id=random_test_id(),
            local_model_path=Fixtures().get_invalid_model_archive_path(),
            model_type="categories_esa",
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
            mock_loader.load_model.return_value = LoadedModel.Builder().build()
            mock_sizer.get_model_size.return_value = expected_model_size

            model_size = MODEL_MANAGER.load_model(
                model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE
            )
            assert expected_model_size == model_size
            mock_loader.load_model.assert_called_once_with(
                model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE
            )
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
            mock_loader.load_model.return_value = (
                LoadedModel.Builder().module(expected_module).build()
            )
            MODEL_MANAGER.load_model(model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE)

            model = MODEL_MANAGER.retrieve_model(model_id)
            assert expected_module == model


def test_get_model_size_returns_size_from_model_sizer():
    """Test that loading a model stores the size from the ModelSizer"""
    mock_loader = MagicMock()
    mock_sizer = MagicMock()
    expected_model_size = 1234
    model_id = random_test_id()

    with patch.object(MODEL_MANAGER, "model_loader", mock_loader):
        with patch.object(MODEL_MANAGER, "model_sizer", mock_sizer):
            mock_loader.load_model.return_value = LoadedModel.Builder().build()
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
