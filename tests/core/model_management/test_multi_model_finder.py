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
Tests for the MultiModelFinder
"""
# Standard
from contextlib import contextmanager

# Third Party
import pytest

# First Party
import aconfig

# Local
from caikit.core import ModelManager
from caikit.core.model_management.multi_model_finder import MultiModelFinder
from tests.conftest import temp_config
from tests.core.helpers import TestFinder
import caikit

## Helpers #####################################################################


@contextmanager
def temp_config_setup(
    multi_finder_name="default",
    multi_finder_cfg=None,
    config_overrides=None,
):
    config_overrides = config_overrides or {}
    finders = config_overrides.setdefault("model_management", {},).setdefault(
        "finders",
        {},
    )
    # Add a local one to proxy to
    if not finders:
        finders["local"] = {"type": "LOCAL"}
    multi_finder_cfg = multi_finder_cfg or {"finder_priority": ["local"]}

    multi_model_section = {
        "type": MultiModelFinder.name,
        "config": multi_finder_cfg,
    }
    finders.setdefault(multi_finder_name, multi_model_section)
    with temp_config(config_overrides, "merge"):
        config_overrides = aconfig.Config(config_overrides, override_env_vars=False)
        yield config_overrides.model_management.finders[multi_finder_name].config


@contextmanager
def temp_config_finder(
    multi_finder_name="default",
    multi_finder_cfg=None,
    config_overrides=None,
):
    with temp_config_setup(
        multi_finder_name=multi_finder_name,
        multi_finder_cfg=multi_finder_cfg,
        config_overrides=config_overrides,
    ) as mf_config:
        mmgr = ModelManager()
        mf_config.model_manager = mmgr
        yield MultiModelFinder(mf_config, multi_finder_name)


## Tests #######################################################################


def test_multi_model_finder_model_found(good_model_path):
    """Make sure that a simple proxy to local works"""
    with temp_config_finder() as finder:
        assert isinstance(finder, MultiModelFinder)
        assert finder.find_model(good_model_path)


@pytest.mark.parametrize(
    "test_finder_config",
    [{"fail_to_find": True}, {"raise_on_find": True}],
)
def test_multi_model_finder_first_not_found(test_finder_config, good_model_path):
    """Make sure that a model can be found if the first finder fails"""
    with temp_config_finder(
        config_overrides={
            "model_management": {
                "finders": {
                    "default": {
                        "type": MultiModelFinder.name,
                        "config": {
                            "finder_priority": ["test-not-found", "local"],
                        },
                    },
                    "test-not-found": {
                        "type": TestFinder.name,
                        "config": test_finder_config,
                    },
                    "local": {"type": "LOCAL"},
                }
            }
        }
    ) as finder:
        assert isinstance(finder, MultiModelFinder)
        assert finder.find_model(good_model_path)


def test_multi_model_finder_not_found(reset_globals):
    """Make sure that a simple proxy to local works"""
    with temp_config_finder() as finder:
        assert isinstance(finder, MultiModelFinder)
        assert finder.find_model("not/a/valid/path") is None
        with pytest.raises(ValueError) as e:
            caikit.core.load("bad/path/to/model")
        assert (
            e.value.args[0]
            == "value check failed: Unable to find a ModuleConfig for bad/path/to/model"
        )


@pytest.mark.parametrize(
    "params",
    [
        ({"finder_priority": "local"}, TypeError),
        ({"finder_priority": [0, 1]}, TypeError),
        ({"finder_priority": []}, ValueError),
        ({"finder_priority": ["invalid-name"]}, ValueError),
        ({"finder_priority": ["default"]}, ValueError),
    ],
)
def test_multi_model_finder_invalid_config(params):
    """Validate all forms of bad config"""
    multi_finder_cfg, error_type = params
    with temp_config_setup(multi_finder_cfg=multi_finder_cfg) as mf_config:
        with pytest.raises(error_type):
            MultiModelFinder(mf_config, "default")
