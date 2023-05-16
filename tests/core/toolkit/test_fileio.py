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
import os
import shutil
import tempfile

# Local
from caikit.core import toolkit

# Unit Test Infrastructure
from tests.base import TestCaseBase


class TestFeatureExtractor(TestCaseBase):
    def setUp(self):
        self.yaml_filename = os.path.join(
            self.fixtures_dir, "dummy_module", "config.yml"
        )
        self.json_filename = os.path.join(
            self.fixtures_dir, "dummy_module", "data.json"
        )
        self.pickle_filename = os.path.join(
            self.fixtures_dir, "dummy_module", "data.pkl"
        )
        self.csv_filename = os.path.join(self.fixtures_dir, "sample.csv")

    def test_load_yaml(self):
        yaml_dict = toolkit.load_yaml(self.yaml_filename)
        self.assertIsInstance(yaml_dict, dict)
        self.assertEqual(yaml_dict["train"]["batch_size"], 64)
        self.assertEqual(yaml_dict["utf_text"], "人工智能")

    def test_save_yaml(self):
        tempdir = tempfile.mkdtemp()
        tempyaml = os.path.join(tempdir, "temp.yml")

        try:
            toolkit.save_yaml(toolkit.load_yaml(self.yaml_filename), tempyaml)
            yaml_dict = toolkit.load_yaml(tempyaml)

        finally:
            shutil.rmtree(tempdir, ignore_errors=True)

        self.assertIsInstance(yaml_dict, dict)
        self.assertEqual(yaml_dict["train"]["batch_size"], 64)
        self.assertEqual(yaml_dict["utf_text"], "人工智能")

    def test_load_json(self):
        json_dict = toolkit.load_json(self.json_filename)
        self.assertIsInstance(json_dict, dict)
        self.assertEqual(json_dict["tokens"][0]["begin"], 0)
        self.assertEqual(json_dict["utf_text"], "人工智能")

    def test_save_json(self):
        tempdir = tempfile.mkdtemp()
        tempjson = os.path.join(tempdir, "temp.json")

        try:
            toolkit.save_json(toolkit.load_json(self.json_filename), tempjson)
            json_dict = toolkit.load_json(tempjson)

        finally:
            shutil.rmtree(tempdir, ignore_errors=True)

        self.assertEqual(json_dict["tokens"][0]["begin"], 0)
        self.assertEqual(json_dict["utf_text"], "人工智能")

    def test_load_csv(self):
        csv_list = toolkit.load_csv(self.csv_filename)
        self.assertEqual(csv_list[0], ["name", "age"])
        self.assertEqual(csv_list[1], ["Tom", "45"])
        self.assertEqual(csv_list[2], ["Jack", "24"])

    def test_save_csv(self):
        tempdir = tempfile.mkdtemp()
        tempcsv = os.path.join(tempdir, "temp.csv")
        try:
            toolkit.save_csv(toolkit.load_csv(self.csv_filename), tempcsv)
            csv_list = toolkit.load_csv(tempcsv)

        finally:
            shutil.rmtree(tempdir, ignore_errors=True)

        self.assertEqual(csv_list[0], ["name", "age"])
        self.assertEqual(csv_list[1], ["Tom", "45"])
        self.assertEqual(csv_list[2], ["Jack", "24"])

    def test_load_dict_csv(self):
        csv_dict = toolkit.load_dict_csv(self.csv_filename)
        self.assertEqual(csv_dict[0]["name"], "Tom")
        self.assertEqual(csv_dict[0]["age"], "45")
        self.assertEqual(csv_dict[1]["name"], "Jack")
        self.assertEqual(csv_dict[1]["age"], "24")

    def test_save_dict_csv(self):
        tempdir = tempfile.mkdtemp()
        tempcsv = os.path.join(tempdir, "temp.csv")
        try:
            toolkit.save_dict_csv(
                toolkit.load_dict_csv(self.csv_filename), tempcsv, "w"
            )
            toolkit.save_dict_csv(
                toolkit.load_dict_csv(self.csv_filename), tempcsv, "a"
            )
            csv_dict = toolkit.load_dict_csv(tempcsv)

        finally:
            shutil.rmtree(tempdir, ignore_errors=True)

        self.assertEqual(csv_dict[0]["name"], "Tom")
        self.assertEqual(csv_dict[0]["age"], "45")
        self.assertEqual(csv_dict[1]["name"], "Jack")
        self.assertEqual(csv_dict[1]["age"], "24")

    def test_load_pickle(self):
        data = toolkit.load_pickle(self.pickle_filename)
        self.assertEqual(data, [1, 2, 3])

    def test_save_pickle(self):
        tempdir = tempfile.mkdtemp()
        temppickle = os.path.join(tempdir, "temp.pkl")

        try:
            toolkit.save_pickle(toolkit.load_pickle(self.pickle_filename), temppickle)
            data = toolkit.load_pickle(temppickle)

        finally:
            shutil.rmtree(tempdir, ignore_errors=True)

        self.assertEqual(data, [1, 2, 3])
