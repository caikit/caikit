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

"""
Tests for utils in config.utils
"""

# First Party
import aconfig

# Local
from caikit.core import config_utils


def test_merge_configs_aconfig():
    """Make sure merge_configs works as expected with aconfig.Config objects"""
    cfg1 = aconfig.Config({"foo": 1, "bar": {"baz": 2}, "bat": [1, 2, 3]})
    cfg2 = aconfig.Config({"foo": 11, "bar": {"buz": 22}, "bat": [5, 6, 7]})
    merged = config_utils.merge_configs(cfg1, cfg2)
    assert merged == aconfig.Config(
        {
            "foo": 11,
            "bar": {"baz": 2, "buz": 22},
            "bat": [5, 6, 7],
        }
    )


def test_merge_configs_dicts():
    """Make sure merge_configs works as expected with plain old dicts objects"""
    cfg1 = {"foo": 1, "bar": {"baz": 2}, "bat": [1, 2, 3]}
    cfg2 = {"foo": 11, "bar": {"buz": 22}, "bat": [5, 6, 7]}
    merged = config_utils.merge_configs(cfg1, cfg2)
    assert merged == {
        "foo": 11,
        "bar": {"baz": 2, "buz": 22},
        "bat": [5, 6, 7],
    }


def test_merge_configs_none_args():
    """Test that None args are handled correctly"""
    cfg1 = {"foo": 1, "bar": {"baz": 2}, "bat": [1, 2, 3]}
    cfg2 = {"foo": 11, "bar": {"buz": 22}, "bat": [5, 6, 7]}
    assert config_utils.merge_configs(cfg1, None) == cfg1
    assert config_utils.merge_configs(None, cfg2) == cfg2
    assert config_utils.merge_configs(None, None) == {}
