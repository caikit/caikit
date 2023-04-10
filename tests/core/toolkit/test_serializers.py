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
import json
import os
import tempfile

# Third Party
import pytest
import yaml

# Local
# Unit Test Infrastructure
from tests.base import TestCaseBase
import caikit.core


class TestCoreSerializers(TestCaseBase):
    def test_json_serializer(self):
        with tempfile.TemporaryDirectory() as tempdir:
            input_object = {"foo": "bar"}

            serializer = caikit.core.toolkit.serializers.JSONSerializer()
            save_path = os.path.join(tempdir, "test.json")
            serializer.serialize(input_object, save_path)

            self.assertTrue(os.path.exists(save_path))
            with open(save_path, "r") as handle:
                self.assertEqual(input_object, json.load(handle))

    def test_text_serializer(self):
        with tempfile.TemporaryDirectory() as tempdir:
            input_list = ["foo", "bar"]
            serializer = caikit.core.toolkit.serializers.TextSerializer()
            save_path = os.path.join(tempdir, "test.txt")
            serializer.serialize(input_list, save_path)

            self.assertTrue(os.path.exists(save_path))
            with open(save_path, mode="r", encoding="utf8") as fh:
                output_list = list(map(str.strip, fh.readlines()))
            self.assertEqual(input_list, output_list)

    def test_yaml_serializer(self):
        with tempfile.TemporaryDirectory() as tempdir:
            input_object = {"foo": "bar"}
            serializer = caikit.core.toolkit.serializers.YAMLSerializer()
            save_path = os.path.join(tempdir, "test.yaml")
            serializer.serialize(input_object, save_path)

            self.assertTrue(os.path.exists(save_path))
            with open(save_path, "r") as handle:
                self.assertEqual(input_object, yaml.safe_load(handle))

    def test_not_implemented_error(self):
        """Make sure that a NotImplementedError is raised if the base class is
        invoked directly
        """
        with pytest.raises(TypeError):
            caikit.core.toolkit.serializers.ObjectSerializer()
