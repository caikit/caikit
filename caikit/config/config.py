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


"""Config methods for the `caikit` library. Mainly interacts with `config.yml`.
"""

# Standard
from importlib import metadata
from typing import Optional
import os
import threading

# Third Party
import semver

# First Party
import aconfig
import alog

# Local
from caikit.core.toolkit.errors import error_handler

log = alog.use_channel("CONFIG")
error = error_handler.get(log)


# restrict functions that are imported so we don't pollute the base module namespce
__all__ = ["Config", "compare_versions", "parse_config"]

BASE_CONFIG_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "config.yml")
)

_CONFIG: aconfig.Config = aconfig.Config({})
_CONFIG_LOCK: threading.Lock = threading.Lock()


def get_config() -> aconfig.Config:
    """Get the caikit configuration"""
    # TODO: update aconfig to allow immutable configs and return an immutable config instead
    return _CONFIG


def configure(config_yml_path: Optional[str] = None):
    """Configure caikit for your usage!
    Sets the internal config to an aconfig.Config object with overrides from multiple sources.

    Sources, last takes precedence:
        1. caikit's base config.yml file baked into this repo
        2. all `config_yml_path`s provided to this function, in the order the calls are made
        3. The config files specified in the `config_files` configuration
            (NB: This may be set by the `CONFIG_FILES` environment variable)
        4. Environment variables, in ALL_CAPS_SNAKE_FORMAT

    Args:
        config_yml_path (Optional[str]): The path to the base configuration yaml
            with overrides for your library. If omitted, only the base caikit config is used.

    Returns: None
        This only sets the config object that is returned by `caikit.get_config()`
    """

    if not config_yml_path and not _CONFIG:
        # If nothing is passed and we currently have no config, use the base config
        config_yml_path = BASE_CONFIG_PATH
    elif not config_yml_path:
        # If we already have config and no more was specified, do nothing
        return

    cfg = parse_config(config_yml_path)

    # Update the config by merging the new updates over the existing config
    with _CONFIG_LOCK:
        merge_configs(_CONFIG, cfg)

    # TODO: hook into any inner `configure()` calls that need to happen (fill this section in)
    error_handler.ENABLE_ERROR_CHECKS = get_config().enable_error_checks
    error_handler.MAX_EXCEPTION_LOG_MESSAGES = get_config().max_exception_log_messages


def parse_config(
    config_file: str
) -> aconfig.Config:
    """This function parses a configuration file used to manage configuration settings for caikit.

    It first parses the config in the specified file, then looks for extra config file paths specified
    as a comma separated list either in this file (key: `config_files`) or in the environment variable
    `CONFIG_FILES`. (Environment variable taking precedence).

    Those extra files are then parsed and merged in from left to right, last taking precedence.

    Args:
        config_file (str): path to a config.yml file

    Returns: aconfig.Config
        The merged configuration
    """
    # Start with the given file
    config = aconfig.Config.from_yaml(config_file, override_env_vars=True)

    # Merge in config from any other user-specified config files
    if config.config_files:
        extra_config_files = [s.strip() for s in str(config.config_files).split(",")]
        for file in extra_config_files:
            log.info(
                {
                    "log_code": "<RUN17612094I>",
                    "message": "Loading config file '%s'" % file,
                }
            )
            new_overrides = aconfig.Config.from_yaml(file, override_env_vars=True)
            config = merge_configs(config, new_overrides)
    return config


def merge_configs(base: Optional[dict], overrides: Optional[dict]) -> dict:
    """Helper to perform a deep merge of the overrides into the base. The merge
    is done in place, but the resulting dict is also returned for convenience.
    The merge logic is quite simple: If both the base and overrides have a key
    and the type of the key for both is a dict, recursively merge, otherwise
    set the base value to the override value.
    Args:
        base: Optional[dict]
            The base config that will be updated with the overrides
        overrides: Optional[dict]
            The override config
    Returns:
        merged: dict
            The merged results of overrides merged onto base
    """
    # Handle none args
    if base is None:
        return overrides or {}
    if overrides is None:
        return base or {}

    # Do the deep merge
    for key, value in overrides.items():
        if (
            key not in base
            or not isinstance(base[key], dict)
            or not isinstance(value, dict)
        ):
            base[key] = value
        else:
            base[key] = merge_configs(base[key], value)

    return base
