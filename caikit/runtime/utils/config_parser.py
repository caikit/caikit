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
This module parses multiple config files to create a single application config
"""
# Standard
import os

# First Party
import aconfig
import alog

CONFIG_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "config", "config.yml")
)

log = alog.use_channel("CONFIG_PARSER")


class ConfigParser:
    """This classes manages configuration settings for the caikit-runtime server. It draws
    values first out of config/config.yml, and may be configured to pull different values for
    different environments. This is especially helpful when distinguishing between testing configs
    and production configs.

    This will also merge in any configuration from config yaml files specified in a comma-separated
    list in the environment variable 'CONFIG_FILES', going from left to right and overwriting on
    merge. (Last takes precedence)
    """

    __instance = None

    def __init__(self):
        # Re-instantiating this is a programming error
        assert self.__class__.__instance is None, "This class is a singleton!"
        ConfigParser.__instance = self

        config = aconfig.Config.from_yaml(CONFIG_PATH, override_env_vars=True)
        self._set_attrs_from_config(config)

        # Merge in config from any other user-provided config files
        if config.config_files:
            extra_config_files = [
                s.strip() for s in str(config.config_files).split(",")
            ]

            for file in extra_config_files:
                log.info(
                    {
                        "log_code": "<RUN17612094I>",
                        "message": "Loading config file '%s'" % file,
                    }
                )
                cfg = aconfig.Config.from_yaml(file, override_env_vars=True)
                self._set_attrs_from_config(cfg)

    def _set_attrs_from_config(self, config: aconfig.Config):
        """
        Sets all attrs from `config` onto `self`.

        Includes special handling to promote config values for the "current environment"

        ```yaml
        environment: prod

        prod:
            key1: value1
        test:
            key1: value2
        ```

        Will produce a config with `key1: value1` at the top level.
        """
        for key, val in config.items():
            self.merge_attr(key, val)

        # 🌶️ These are not environment variables, this is the section of config nested under e.g.
        # `prod` or `test`.
        if config.environment:
            environment_config = config[config.environment.lower()]
            for key, val in environment_config.items():
                self.merge_attr(key, val)

    def merge_attr(self, key, val):
        """Safely merge in top-level dictionary items in the config.

        This is a shallow merge only. Potential TODO: make it a recursive deep merge
        """
        if isinstance(val, dict):
            if hasattr(self, key) and isinstance(getattr(self, key), dict):
                # Both the existing config item and the new one being merged in are dictionaries
                # Use old.update(new) to shallow merge them in
                dictionary_config_item = getattr(self, key)
                dictionary_config_item.update(val)
                setattr(self, key, dictionary_config_item)
            else:
                setattr(self, key, val)
        else:
            setattr(self, key, val)

    @classmethod
    def get_instance(cls) -> "ConfigParser":
        """This method returns the instance of Config Parser"""
        if not cls.__instance:
            cls.__instance = ConfigParser()
        return cls.__instance
