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
import os

# Third Party
import pytest
import yaml

# First Party
import aconfig

# Local
from tests.conftest import temp_config
import caikit


# Let all of these tests call caikit.configure() without mucking the real config up
@pytest.fixture(autouse=True)
def patched_config():
    with temp_config({}):
        yield


CFG_1 = {"foo": 1, "combined": {"one": "bar"}}

CFG_2 = {"foo": 2, "combined": {"two": "baz"}}

CFG_3 = {"foo": 3, "combined": {"three": "buz"}}


def _dump_yml(cfg_dict: dict, path: str):
    with open(path, "w") as f:
        yaml.safe_dump(cfg_dict, f)
        f.flush()


def test_configure_raises_on_no_input():
    with pytest.raises(ValueError):
        caikit.configure()


def test_config_is_immutable():
    assert isinstance(caikit.get_config(), aconfig.ImmutableConfig)
    with pytest.raises(AttributeError):
        caikit.get_config().config_files = "foo"


def test_configure_reads_a_config_yml(tmp_path):
    path = os.path.join(os.path.join(tmp_path, "one.yml"))
    _dump_yml(CFG_1, path)
    caikit.configure(path)

    for k, v in CFG_1.items():
        assert caikit.get_config()[k] == v


def test_configure_reads_a_config_dict():
    caikit.configure(config_dict=CFG_1)

    for k, v in CFG_1.items():
        assert caikit.get_config()[k] == v


def test_configure_merges_over_existing_configs():
    caikit.configure(config_dict=CFG_1)

    caikit.configure(config_dict=CFG_2)

    cfg = caikit.get_config()
    # Not just the second one, we merged things in
    assert cfg != CFG_2
    # foo was overriden from two.yml
    assert cfg.foo == 2
    assert "one" in cfg.combined and "two" in cfg.combined


@patch.dict(os.environ, {"FOO": "42"})
def test_configure_picks_up_env_vars():
    caikit.configure(config_dict=CFG_1)

    # the FOO=42 env var is picked up
    assert caikit.get_config().foo == 42


def test_configure_adds_more_user_specified_files_from_env(tmp_path):
    path_2 = os.path.join(os.path.join(tmp_path, "two.yml"))
    _dump_yml(CFG_2, path_2)

    path_3 = os.path.join(os.path.join(tmp_path, "three.yml"))
    _dump_yml(CFG_3, path_3)

    with patch.dict(os.environ, {"CONFIG_FILES": f"{path_2},{path_3}"}):
        caikit.configure(config_dict=CFG_1)

    cfg = caikit.get_config()
    # all three configs applied, with CFG_3 applied last
    assert cfg.foo == 3
    assert "one" in cfg.combined and "two" in cfg.combined and "three" in cfg.combined


def test_configure_merges_lists():
    cfg1 = {"foo_list": [1, 2, 3]}
    cfg2 = {"foo_list": [4, 5, 6]}

    # Second list value is prepended to the first
    caikit.configure(config_dict=cfg1)
    caikit.configure(config_dict=cfg2)
    assert caikit.get_config().foo_list == [4, 5, 6, 1, 2, 3]

    # If values already existed in the list, they are popped then prepended
    cfg3 = {"foo_list": [1, 2]}
    caikit.configure(config_dict=cfg3)
    assert caikit.get_config().foo_list == [1, 2, 4, 5, 6, 1, 2, 3]


def test_merge_strategy():
    # If merge_strategy == 'override', then lists and dicts are not merged
    cfg1 = {"foo_list": [1, 2, 3], "foo_dict": {"bar": "baz"}}
    cfg2 = {
        "foo_list": [4, 5, 6],
        "foo_dict": {"baz": "bar"},
        "merge_strategy": "override",
    }

    caikit.configure(config_dict=cfg1)
    caikit.configure(config_dict=cfg2)
    assert caikit.get_config().foo_list == [4, 5, 6]
    assert caikit.get_config().foo_dict == {"baz": "bar"}


def test_merge_configs_aconfig():
    """Make sure merge_configs works as expected with aconfig.Config objects"""
    cfg1 = aconfig.Config({"foo": 1, "bar": {"baz": 2}, "bat": [1, 2, 3]})
    cfg2 = aconfig.Config({"foo": 11, "bar": {"buz": 22}, "bat": [5, 6, 7]})
    merged = caikit.config.config.merge_configs(cfg1, cfg2)
    assert merged == aconfig.Config(
        {
            "foo": 11,
            "bar": {"baz": 2, "buz": 22},
            "bat": [5, 6, 7, 1, 2, 3],
        }
    )


def test_merge_configs_dicts():
    """Make sure merge_configs works as expected with plain old dicts objects"""
    cfg1 = {"foo": 1, "bar": {"baz": 2}, "bat": [1, 2, 3]}
    cfg2 = {"foo": 11, "bar": {"buz": 22}, "bat": [5, 6, 7]}
    merged = caikit.config.config.merge_configs(cfg1, cfg2)
    assert merged == {
        "foo": 11,
        "bar": {"baz": 2, "buz": 22},
        "bat": [5, 6, 7, 1, 2, 3],
    }


def test_merge_configs_none_args():
    """Test that None args are handled correctly"""
    cfg1 = {"foo": 1, "bar": {"baz": 2}, "bat": [1, 2, 3]}
    cfg2 = {"foo": 11, "bar": {"buz": 22}, "bat": [5, 6, 7]}
    assert caikit.config.config.merge_configs(cfg1, None) == cfg1
    assert caikit.config.config.merge_configs(None, cfg2) == cfg2
    assert caikit.config.config.merge_configs(None, None) == {}
