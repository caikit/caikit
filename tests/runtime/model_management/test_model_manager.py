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
import os
import shutil
import unittest

# Third Party
import grpc

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


class TestModelManager(unittest.TestCase):
    """This test suite tests the modelmanager class"""

    def setUp(self):
        """This method runs before each test begins to run"""
        self.model_manager = ModelManager.get_instance()

    def tearDown(self):
        """This method runs after each test executes"""
        self.model_manager.unload_all_models()

    # ****************************** Integration Tests ****************************** #
    # These tests do not patch in mocks, the manager will use real instances of its dependencies

    def test_load_model_ok_response(self):
        model_id = "happy_load_test"
        model_size = self.model_manager.load_model(
            model_id=model_id,
            local_model_path=Fixtures.get_good_model_path(),
            model_type=Fixtures.get_good_model_type(),
        )
        self.assertGreater(model_size, 0)

    def test_load_local_models(self):
        with TemporaryDirectory() as tempdir:
            shutil.copytree(
                Fixtures.get_good_model_path(), os.path.join(tempdir, "model1")
            )
            shutil.copy(
                Fixtures.get_good_model_archive_path(),
                os.path.join(tempdir, "model2.zip"),
            )

            self.model_manager.load_local_models(tempdir)
            self.assertEqual(len(self.model_manager.loaded_models), 2)
            self.assertIn("model1", self.model_manager.loaded_models.keys())
            self.assertIn("model2.zip", self.model_manager.loaded_models.keys())
            self.assertNotIn(
                "model-does-not-exist.zip", self.model_manager.loaded_models.keys()
            )

    def test_model_manager_loads_local_models_on_init(self):
        with TemporaryDirectory() as tempdir:
            shutil.copytree(
                Fixtures.get_good_model_path(), os.path.join(tempdir, "model1")
            )
            shutil.copy(
                Fixtures.get_good_model_archive_path(),
                os.path.join(tempdir, "model2.zip"),
            )
            ModelManager._ModelManager__instance = None
            with temp_config(
                {"runtime": {"local_models_dir": tempdir}}, merge_strategy="merge"
            ):
                self.model_manager = ModelManager()

                self.assertEqual(len(self.model_manager.loaded_models), 2)
                self.assertIn("model1", self.model_manager.loaded_models.keys())
                self.assertIn("model2.zip", self.model_manager.loaded_models.keys())
                self.assertNotIn(
                    "model-does-not-exist.zip", self.model_manager.loaded_models.keys()
                )

    def test_model_manager_raises_if_all_local_models_fail_to_load(self):
        with TemporaryDirectory() as tempdir:
            shutil.copy(
                Fixtures.get_bad_model_archive_path(), os.path.join(tempdir, "model1")
            )
            shutil.copy(
                Fixtures.get_invalid_model_archive_path(),
                os.path.join(tempdir, "model2.zip"),
            )
            with self.assertRaises(CaikitRuntimeException) as ctx:
                self.model_manager.load_local_models(tempdir)
            self.assertEqual(grpc.StatusCode.INTERNAL, ctx.exception.status_code)

    def test_load_model_error_response(self):
        """Test load model's model does not exist when the loader throws"""
        with self.assertRaises(CaikitRuntimeException) as context:
            self.model_manager.load_model(
                model_id=random_test_id(),
                local_model_path=Fixtures().get_invalid_model_archive_path(),
                model_type="categories_esa",
            )

        self.assertEqual(context.exception.status_code, grpc.StatusCode.NOT_FOUND)
        self.assertEqual(len(self.model_manager.loaded_models), 0)

    def test_load_model_map_insertion(self):
        """Test if loaded model is correctly added to map storing model data"""
        model = random_test_id()
        self.model_manager.load_model(
            model_id=model,
            local_model_path=Fixtures.get_good_model_path(),
            model_type=Fixtures.get_good_model_type(),
        )
        self.assertGreater(self.model_manager.loaded_models[model].size(), 0)

    def test_load_model_count(self):
        """Test if multiple loaded models are added to map storing model data"""
        self.model_manager.load_model(
            model_id=random_test_id(),
            local_model_path=Fixtures.get_good_model_path(),
            model_type=Fixtures.get_good_model_type(),
        )
        self.model_manager.load_model(
            model_id=random_test_id(),
            local_model_path=Fixtures.get_good_model_path(),
            model_type=Fixtures.get_good_model_type(),
        )
        self.assertEqual(len(self.model_manager.loaded_models), 2)

    def test_unload_model_ok_response(self):
        """Test to make sure that given a loaded model ID, the model manager is able to correctly
        unload a model, giving a nonzero model size back."""
        model_id = "happy_unload_test"
        self.model_manager.load_model(
            model_id=model_id,
            local_model_path=Fixtures.get_good_model_path(),
            model_type=Fixtures.get_good_model_type(),
        )
        model_size = self.model_manager.unload_model(model_id=model_id)
        self.assertTrue(model_size >= 0)

    def test_unload_model_count(self):
        """Test if unloaded models are deleted from loaded models map"""
        id_1 = "test"
        id_2 = "test2"
        # Load models from COS
        self.model_manager.load_model(
            model_id=id_1,
            local_model_path=Fixtures.get_good_model_path(),
            model_type=Fixtures.get_good_model_type(),
        )
        self.model_manager.load_model(
            model_id=id_2,
            local_model_path=Fixtures.get_good_model_path(),
            model_type=Fixtures.get_good_model_type(),
        )
        # Unload one of the models, and make sure it was properly removed
        self.model_manager.unload_model(model_id=id_2)
        self.assertEqual(len(self.model_manager.loaded_models), 1)
        self.assertIn(id_1, self.model_manager.loaded_models.keys())

    def test_unload_all_models_count(self):
        """Test if unload all models deletes every model from loaded models map"""
        id_1 = "test"
        id_2 = "test2"
        self.model_manager.load_model(
            model_id=id_1,
            local_model_path=Fixtures.get_good_model_path(),
            model_type=Fixtures.get_good_model_type(),
        )
        self.model_manager.load_model(
            model_id=id_2,
            local_model_path=Fixtures.get_good_model_path(),
            model_type=Fixtures.get_good_model_type(),
        )
        self.model_manager.unload_all_models()
        self.assertEqual(len(self.model_manager.loaded_models), 0)

    def test_unload_model_not_loaded_response(self):
        """Test unload model for model not loaded does NOT throw an error"""
        try:
            self.model_manager.unload_model(model_id=random_test_id())
        except Exception:
            self.fail("Unload for a model that does not exist threw an error!")

    # TODO: If this is refactored to pytest, the loaded_model_id fixture can be used
    def test_retrieve_model_returns_loaded_model(self):
        """Test that a loaded model can be retrieved"""
        model_id = random_test_id()
        Fixtures.load_model(
            model_id=model_id,
            local_model_path=Fixtures.get_good_model_path(),
            model_type=Fixtures.get_good_model_type(),
        )
        model = self.model_manager.retrieve_model(model_id)
        self.assertIsInstance(model, ModuleBase)
        self.assertEqual(len(self.model_manager.loaded_models), 1)

    def test_retrieve_model_raises_error_for_not_found_model(self):
        """Test that gRPC NOT_FOUND exception raised when non-existent model retrieved"""
        with self.assertRaises(CaikitRuntimeException) as context:
            self.model_manager.retrieve_model("not-found")
        self.assertEqual(context.exception.status_code, grpc.StatusCode.NOT_FOUND)

    def test_model_size_ok_response(self):
        """Test if loaded model correctly returns model size"""
        model = random_test_id()
        self.model_manager.load_model(
            model_id=model,
            local_model_path=Fixtures.get_good_model_path(),
            model_type=Fixtures.get_good_model_type(),
        )
        self.assertTrue(self.model_manager.get_model_size(model) > 0)

    def test_model_size_error_not_found_response(self):
        """Test model size's model does not exist error response"""
        with self.assertRaises(CaikitRuntimeException) as context:
            self.model_manager.get_model_size("no_exist_model")
        self.assertEqual(context.exception.status_code, grpc.StatusCode.NOT_FOUND)

    def test_model_size_error_none_not_found_response(self):
        """Test model size's model is None error response"""
        with self.assertRaises(CaikitRuntimeException) as context:
            self.model_manager.get_model_size(None)
        self.assertEqual(context.exception.status_code, grpc.StatusCode.NOT_FOUND)

    def test_estimate_model_size_ok_response_on_loaded_model(self):
        """Test if loaded model correctly returns model size"""
        self.model_manager.load_model(
            model_id=random_test_id(),
            local_model_path=Fixtures.get_good_model_path(),
            model_type=Fixtures.get_good_model_type(),
        )
        self.assertTrue(
            self.model_manager.estimate_model_size(
                "test", Fixtures.get_good_model_path(), Fixtures.get_good_model_type()
            )
            > 0
        )

    def test_estimate_model_size_ok_response_on_nonloaded_model(self):
        """Test if a model that's not loaded, correctly returns predicted model size"""
        self.assertTrue(
            self.model_manager.estimate_model_size(
                "test", Fixtures.get_good_model_path(), Fixtures.get_good_model_type()
            )
            > 0
        )

    def test_estimate_model_size_by_type(self):
        """Test that a model's size is estimated differently based on its type"""
        config = get_config().inference_plugin.model_mesh
        self.assertTrue(Fixtures.get_good_model_type() in config.model_size_multipliers)

        typed_model_size = self.model_manager.estimate_model_size(
            "test", Fixtures.get_good_model_path(), Fixtures.get_good_model_type()
        )
        untyped_model_size = self.model_manager.estimate_model_size(
            "test", Fixtures.get_good_model_path(), "test-not-a-model-type"
        )

        self.assertTrue(typed_model_size > 0)
        self.assertTrue(untyped_model_size > 0)

        self.assertNotAlmostEqual(typed_model_size, untyped_model_size)

    def test_estimate_model_size_error_not_found_response(self):
        """Test if error in predict model size on unknown model path"""
        with self.assertRaises(CaikitRuntimeException) as context:
            self.model_manager.estimate_model_size(
                model_id=random_test_id(),
                local_model_path="no_exist.zip",
                model_type="categories_esa",
            )
        self.assertEqual(context.exception.status_code, grpc.StatusCode.NOT_FOUND)

    # ****************************** Unit Tests ****************************** #
    # These tests patch in mocks for the manager's dependencies, to test its code in isolation

    def test_load_model(self):
        """Test to make sure that given valid input, the model manager gives a happy response
        when we tried to load in a model (model size > 0 or 0 if the model size will be computed
        at a later time)."""
        mock_loader = MagicMock()
        mock_sizer = MagicMock()
        model_id = random_test_id()
        expected_model_size = 1234

        with patch.object(self.model_manager, "model_loader", mock_loader):
            with patch.object(self.model_manager, "model_sizer", mock_sizer):
                mock_loader.load_model.return_value = LoadedModel.Builder().build()
                mock_sizer.get_model_size.return_value = expected_model_size

                model_size = self.model_manager.load_model(
                    model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE
                )
                self.assertEqual(expected_model_size, model_size)
                mock_loader.load_model.assert_called_once_with(
                    model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE
                )
                mock_sizer.get_model_size.assert_called_once_with(
                    model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE
                )

    def test_load_model_throws_if_the_model_loader_throws(self):
        """Test load model's model does not exist when the loader throws"""
        mock_loader = MagicMock()
        with patch.object(self.model_manager, "model_loader", mock_loader):
            mock_loader.load_model.side_effect = CaikitRuntimeException(
                grpc.StatusCode.NOT_FOUND, "test any not found exception"
            )

            with self.assertRaises(CaikitRuntimeException) as context:
                self.model_manager.load_model(
                    random_test_id(), ANY_MODEL_PATH, ANY_MODEL_TYPE
                )

            self.assertEqual(context.exception.status_code, grpc.StatusCode.NOT_FOUND)
            self.assertEqual(len(self.model_manager.loaded_models), 0)

    def test_retrieve_model_returns_the_module_from_the_model_loader(self):
        """Test that a loaded model can be retrieved"""
        model_id = random_test_id()
        expected_module = "this is definitely a stub module"
        mock_sizer = MagicMock()
        mock_loader = MagicMock()

        with patch.object(self.model_manager, "model_loader", mock_loader):
            with patch.object(self.model_manager, "model_sizer", mock_sizer):
                mock_sizer.get_model_size.return_value = 1
                mock_loader.load_model.return_value = (
                    LoadedModel.Builder().module(expected_module).build()
                )
                self.model_manager.load_model(model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE)

                model = self.model_manager.retrieve_model(model_id)
                self.assertEqual(expected_module, model)

    def test_get_model_size_returns_size_from_model_sizer(self):
        """Test that loading a model stores the size from the ModelSizer"""
        mock_loader = MagicMock()
        mock_sizer = MagicMock()
        expected_model_size = 1234
        model_id = random_test_id()

        with patch.object(self.model_manager, "model_loader", mock_loader):
            with patch.object(self.model_manager, "model_sizer", mock_sizer):
                mock_loader.load_model.return_value = LoadedModel.Builder().build()
                mock_sizer.get_model_size.return_value = expected_model_size

                self.model_manager.load_model(model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE)

                model_size = self.model_manager.get_model_size(model_id)
                self.assertEqual(expected_model_size, model_size)

    def test_estimate_model_size_returns_size_from_model_sizer(self):
        """Test that estimating a model size uses the ModelSizer"""
        mock_sizer = MagicMock()
        expected_model_size = 5678
        model_id = random_test_id()

        with patch.object(self.model_manager, "model_sizer", mock_sizer):
            mock_sizer.get_model_size.return_value = expected_model_size
            model_size = self.model_manager.estimate_model_size(
                model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE
            )
            self.assertEqual(expected_model_size, model_size)

    def test_estimate_model_size_throws_if_model_sizer_throws(self):
        """Test that estimating a model size uses the ModelSizer"""
        mock_sizer = MagicMock()
        model_id = random_test_id()

        with patch.object(self.model_manager, "model_sizer", mock_sizer):
            mock_sizer.get_model_size.side_effect = CaikitRuntimeException(
                grpc.StatusCode.UNAVAILABLE, "test-any-exception"
            )
            with self.assertRaises(CaikitRuntimeException) as context:
                self.model_manager.estimate_model_size(
                    model_id, ANY_MODEL_PATH, ANY_MODEL_TYPE
                )
            self.assertEqual(context.exception.status_code, grpc.StatusCode.UNAVAILABLE)
