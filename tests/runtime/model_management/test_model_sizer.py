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
import os
import unittest
import uuid

# Third Party
import grpc

# Local
from caikit import get_config
from caikit.runtime.model_management.model_sizer import ModelSizer
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
from tests.conftest import random_test_id, temp_config
from tests.fixtures import Fixtures


def _random_test_model_type() -> str:
    return "test-any-type-" + str(uuid.uuid4())


class TestModelSizer(unittest.TestCase):
    """This test suite tests the model sizer class"""

    def setUp(self):
        """This method runs before each test begins to run"""
        self.model_sizer = ModelSizer.get_instance()

    @staticmethod
    def _add_file(path, charsize) -> int:
        with open(path, "w") as f:
            content = ""
            for i in range(charsize):
                content += str(i)
            f.write(content)
        return os.path.getsize(path)

    def test_it_can_size_a_model_folder(self):
        """Get local model directory size"""
        with TemporaryDirectory() as d:
            total_size = TestModelSizer._add_file(os.path.join(d, "some_file"), 256)

            # add a nested file as well
            subdir = os.path.join(d, "some_dir")
            os.makedirs(subdir)
            total_size += TestModelSizer._add_file(
                os.path.join(subdir, "some_file"), 512
            )
            model_type = _random_test_model_type()
            mult = 7
            with temp_config(
                {
                    "inference_plugin": {
                        "model_mesh": {"model_size_multipliers": {model_type: mult}}
                    }
                }
            ):
                expected_size = total_size * mult
                size = self.model_sizer.get_model_size(
                    model_id=random_test_id(),
                    local_model_path=d,
                    model_type=model_type,
                )
                self.assertEqual(size, expected_size)

    def test_it_can_size_a_model_archive(self):
        """Get local model archive file size"""
        model_type = _random_test_model_type()
        mult = 42
        with temp_config(
            {
                "inference_plugin": {
                    "model_mesh": {"model_size_multipliers": {model_type: mult}}
                }
            }
        ):
            expected_size = (
                os.path.getsize(Fixtures.get_good_model_archive_path()) * mult
            )

            size = self.model_sizer.get_model_size(
                model_id=random_test_id(),
                local_model_path=Fixtures.get_good_model_archive_path(),
                model_type=model_type,
            )
            self.assertEqual(size, expected_size)

    def test_it_uses_the_default_multiplier_for_unknown_model_types(self):
        model_type = "definitely not a real type"
        expected_size = (
            os.path.getsize(Fixtures.get_good_model_archive_path())
            * get_config().inference_plugin.model_mesh.default_model_size_multiplier
        )

        size = self.model_sizer.get_model_size(
            model_id=random_test_id(),
            local_model_path=Fixtures.get_good_model_archive_path(),
            model_type=model_type,
        )
        self.assertEqual(size, expected_size)

    def test_it_throws_not_found_if_file_does_not_exist(self):
        with self.assertRaises(CaikitRuntimeException) as context:
            self.model_sizer.get_model_size(
                model_id=random_test_id(),
                local_model_path="not/a/path/to/anything",
                model_type=_random_test_model_type(),
            )
        self.assertEqual(grpc.StatusCode.NOT_FOUND, context.exception.status_code)

    def test_it_caches_model_archive_sizes(self):
        """Check that we avoid unnecessary disk access"""
        with TemporaryDirectory() as d:
            TestModelSizer._add_file(os.path.join(d, "some_file"), 256)

            model_type = _random_test_model_type()
            model_id = random_test_id()
            original_size = self.model_sizer.get_model_size(
                model_id=model_id,
                local_model_path=d,
                model_type=model_type,
            )

            TestModelSizer._add_file(os.path.join(d, "some_new_file"), 256)
            # New file is not accounted for in size, because we cached
            new_size = self.model_sizer.get_model_size(
                model_id=model_id,
                local_model_path=d,
                model_type=model_type,
            )
            self.assertEqual(original_size, new_size)


if __name__ == "__main__":
    unittest.main()
