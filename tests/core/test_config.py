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
from unittest.mock import patch
import sys

# Local
from caikit.config import get_config

# Unit Test Infrastructure
from tests.base import TestCaseBase

## Helpers #####################################################################

SAMPLE_RESOURCES = [
    "newres_v0-0-3_relations_sire_lang_de_template_2019-12-09-100000.zip",
    "res_newres_v0-0-3_relations_sire_lang_de_template_2019-12-09-100000.zip",
    "file_path_v0-0-3_relations_sire_lang_de_stock_2019-12-09-100000.zip",
    "file_path_v0-0-2_relations_sire_lang_de_stock_2019-12-09-100000.zip",
    "file_path_v0-0-1_relations_sire_lang_de_stock_2019-12-09-100000.zip",
    "file_path_v0-0-1_entity-mentions_sire_lang_en_stock_2019-12-09-100000.zip",
    "file_path_v0-0-1_entity-mentions_sire_lang_de_stock_2019-12-09-100000.zip",
    "file_path_v0-0-1_entity-mentions_bilstm_lang_de_stock_2019-12-09-100000.zip",
    "file_path_v0-0-1_bilstm_lang_de_stock_2019-12-09-100000.zip",
    "file_path_v0-0-1_entity-mentions_lang_de_stock_2019-12-09-100000.zip",
    "file_path_v0-0-1_entity-mentions_bilstm_stock_2019-12-09-100000.zip",
    "file_path_v0-0-1_stock_2019-12-09-100000.zip",
    "v0-0-1_stock_2019-12-09-100000.zip",
    "stock_2019-12-09-100000.zip",
    "file_path_stock_2019-12-09-100000.zip",
    "file_path_v0-0-1_stock.zip",
]

## Tests #######################################################################


class TestConfig(TestCaseBase):
    @classmethod
    def setUpClass(cls):
        # 🌶️🌶️🌶️ Set the version to 1.0.0 for the tests in this class to work properly
        # (The version is otherwise reported as 0.0.0)
        get_config().library_version = "1.0.0"

        # pylint: disable=use-dict-literal
        cls.resource_dict = dict(
            **{name.rsplit(".", 1)[0]: "file.zip" for name in SAMPLE_RESOURCES},
            **{
                "rbr": "rbr",
                "11": True,
            }
        )

    # TODO: ARGGGGGGHHH
    @patch("importlib.metadata.version")
    def test_library_version(self, mock_get_distribution):
        version = "1.2.3"
        mock_get_distribution.return_value = version
        # Delete caikit.core.config from the modules cache so that re-importing runs the config
        # setup code again
        sys.modules.pop("caikit.core.config")
        # pylint: disable=redefined-outer-name,reimported,import-outside-toplevel
        # Local
        from caikit.config import get_config

        self.assertEqual(get_config().library_version, version)
