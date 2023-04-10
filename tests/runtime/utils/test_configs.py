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

# Third Party
import yaml

# First Party
import alog

# Local
from caikit.runtime.utils.config_parser import ConfigParser

log = alog.use_channel("TEST-CONFIGS")


def delete_config_singleton():
    ConfigParser._ConfigParser__instance = None


class TestConfigs(unittest.TestCase):
    """This test suite tests the throughput metric class"""

    LABEL = "test_label"

    @classmethod
    def tearDownClass(cls) -> None:
        delete_config_singleton()
        os.unsetenv("CONFIG_FILES")
        # reset with real stuff
        ConfigParser.get_instance()

    def tearDown(self) -> None:
        # make sure no tests leak an internally set CONFIG_FILES
        os.environ["CONFIG_FILES"] = ""

    def test_it_loads_configs_fine(self):
        delete_config_singleton()
        c = ConfigParser()
        self.assertIsInstance(c, ConfigParser)

    def test_it_overrides_deployment_environment_settings(self):
        delete_config_singleton()
        old_deploy_env = os.getenv("ENVIRONMENT")
        try:
            # PROD should set the grpc sleep setting up to 45
            os.environ["ENVIRONMENT"] = "PROD"
            c = ConfigParser()
            self.assertEqual(45, c.grpc_server_sleep_interval)
        finally:
            # Try to make sure we re-set this to not bork other tests
            os.environ["ENVIRONMENT"] = old_deploy_env

    def test_it_can_merge_in_a_config_file(self):
        delete_config_singleton()
        try:
            with TemporaryDirectory() as tempdir:
                cfg = {"grpc_server_sleep_interval": 7, "new_key": "new_value"}
                path = os.path.join(tempdir, "new_config.yml")
                with open(path, "w") as f:
                    yaml.dump(cfg, f)

                os.environ["CONFIG_FILES"] = path
                c = ConfigParser()

                self.assertEqual(7, c.grpc_server_sleep_interval)
                self.assertEqual("new_value", c.new_key)
        finally:
            os.environ["CONFIG_FILES"] = ""

    def test_it_can_merge_in_a_dictionary_without_deleting_all_keys(self):
        delete_config_singleton()
        try:
            with TemporaryDirectory() as tempdir:
                cfg = {"new_dict": {"key_one": "val_one", "key_three": "val_three"}}
                path = os.path.join(tempdir, "new_config.yml")
                with open(path, "w") as f:
                    yaml.dump(cfg, f)

                # When this is merged in, we should
                # - retain `key_one` from before
                # - add 'key_two'
                # - and update the value of 'key_three'
                cfg2 = {"new_dict": {"key_two": "val_two", "key_three": "val_four"}}
                path2 = os.path.join(tempdir, "new_config_two.yml")
                with open(path2, "w") as f:
                    yaml.dump(cfg2, f)

                os.environ["CONFIG_FILES"] = f"{path},{path2}"
                c = ConfigParser()

                self.assertEqual("val_one", c.new_dict.key_one)
                self.assertEqual("val_two", c.new_dict.key_two)
                self.assertEqual("val_four", c.new_dict.key_three)
        finally:
            os.environ["CONFIG_FILES"] = ""


if __name__ == "__main__":
    unittest.main()
