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
from caikit.config import get_config
from caikit.core.model_management.factories import model_initializer_factory
from caikit.core.model_management.local_model_finder import LocalModelFinder
from caikit.core.model_management.model_initializer_base import ModelInitializerBase
from caikit.core.modules import ModuleConfig
from tests.conftest import temp_config

## Helpers #####################################################################

# Add bad initializer to model factory
class BadModelInitializer(ModelInitializerBase):
    name = "BAD"

    def __init__(self, config: aconfig.Config, instance_name: str):
        """A FactoryConstructible object must be constructed with a config
        object that it uses to pull in all configuration
        """
        pass

    def init(
        self,
        model_config,
        **kwargs,
    ):
        raise ValueError("Bad Model Initializer")


model_initializer_factory.register(BadModelInitializer)


@contextmanager
def construct_mm_initializer(multi_model_config, config_override={}):
    config_override = config_override or {
        "model_management": {
            "initializers": {
                "default": {
                    "type": "MULTI",
                    "config": multi_model_config,
                },
                "local": {
                    "type": "LOCAL",
                },
                "bad": {"type": "BAD"},
            }
        }
    }

    with temp_config(config_override, "merge"):
        yield model_initializer_factory.construct(
            get_config().model_management.initializers.default, "default"
        )


## Tests #######################################################################


@pytest.mark.parametrize(
    ["initializers", "load_successful"],
    [[["local"], True], [["bad", "local"], True], [["bad"], False]],
)
def test_multi_model_initializer(good_model_path, initializers, load_successful):
    finder = LocalModelFinder(aconfig.Config({}), "local")
    config = finder.find_model(good_model_path)
    with construct_mm_initializer(
        {"initializer_priority": initializers}
    ) as initializer:
        if load_successful:
            assert initializer.init(config)
        else:
            assert not initializer.init(config)


def test_multi_model_initializer_bad_config():
    config = ModuleConfig({"module_id": "bad"})
    with construct_mm_initializer(
        {"initializer_priority": ["bad", "local"]}
    ) as initializer:
        assert not initializer.init(config)
