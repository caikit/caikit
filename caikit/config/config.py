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
from typing import Any, Dict, Optional, Union
import os
import threading

# First Party
import aconfig
import alog

log = alog.use_channel("CONFIG")

BASE_CONFIG_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "config.yml")
)

# The core config object that is continually merged into
_CONFIG: aconfig.Config = aconfig.Config({})
# An immutable view into the core config object, to be passed to callers
_IMMUTABLE_CONFIG: aconfig.ImmutableConfig = aconfig.ImmutableConfig({})
_CONFIG_LOCK: threading.Lock = threading.Lock()
# Little helper type for signatures
_CONFIG_TYPE = Union[dict, aconfig.Config]


def get_config() -> aconfig.Config:
    """Get the caikit configuration"""
    return _IMMUTABLE_CONFIG


def configure(
    config_yml_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None
):
    """Configure caikit for your usage!
    Merges into the internal aconfig.Config object with overrides from multiple sources.

    Sources, last takes precedence:
        1. The existing configuration from calls to `caikit.configure()`
        2. The config from `config_yml_path`
        3. The config from `config_dict`
        4. The config files specified in the `config_files` configuration
            (NB: This may be set by the `CONFIG_FILES` environment variable)
        5. Environment variables, in ALL_CAPS_SNAKE_FORMAT

    Args:
        config_yml_path (Optional[str]): The path to the base configuration yaml
            with overrides for your usage.
        config_dict (Optional[Dict]): Config overrides in dictionary form

    Returns: None
        This only sets the config object that is returned by `caikit.get_config()`
    """
    if not config_yml_path and not config_dict:
        log.error("<CFG43273054E>", "No config_file or config_dict provided")
        raise ValueError("No config_file or config_dict provided")

    cfg = aconfig.Config(_CONFIG)
    if config_yml_path:
        new_config = aconfig.Config.from_yaml(config_yml_path)
    else:
        new_config = aconfig.Config(config_dict)

    cfg = merge_configs(cfg, new_config, _get_merge_strategy(new_config))

    cfg = _merge_extra_files(cfg)
    _update_global_config(cfg)


def _update_global_config(cfg: aconfig.Config):
    """Updates the caikit config and creates a new immutable view of it to be shared via
    get_config().
    Locked because who the heck knows if merge_configs() is threadsafe.
    """
    # pylint: disable=global-statement
    global _IMMUTABLE_CONFIG
    # Update the config by merging the new updates over the existing config
    with _CONFIG_LOCK:
        # Locked just in case `configure()` is called concurrently for any reason
        _CONFIG.update(cfg)
        _IMMUTABLE_CONFIG = aconfig.ImmutableConfig(_CONFIG, override_env_vars=False)


def _merge_extra_files(config: aconfig.Config) -> aconfig.Config:
    """Looks at the `config_files` configuration item and merges those files into the config,
    left to right"""
    if config.config_files:
        extra_config_files = [
            s.strip()
            for s in str(config.config_files or os.environ.get("CONFIG_FILES")).split(
                ","
            )
        ]
        for file in extra_config_files:
            log.info(
                {
                    "log_code": "<RUN17612094I>",
                    "message": "Loading config file '%s'" % file,
                }
            )
            new_overrides = aconfig.Config.from_yaml(file, override_env_vars=True)
            config = merge_configs(
                config, new_overrides, _get_merge_strategy(new_overrides)
            )
    return config


def merge_configs(
    base: Optional[_CONFIG_TYPE],
    overrides: Optional[_CONFIG_TYPE],
    merge_strategy: str = "merge",
) -> _CONFIG_TYPE:
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
        merge_strategy: str
            The merging strategy, either `merge` or `override`
            `override` will replace values in base with those from overrides
            `merge` will deep-merge dictionaries and prepend-merge lists
    Returns:
        merged: dict
            The merged results of overrides merged onto base
    """
    # Handle none args
    if base is None:
        return overrides or {}
    if overrides is None:
        return base or {}

    if merge_strategy == "override":
        base.update(overrides)
        return base

    # Do the deep merge
    for key, value in overrides.items():
        if (
            key not in base
            or not isinstance(base[key], (dict, list))
            or not isinstance(value, (dict, list))
        ):
            base[key] = value
        elif isinstance(value, list):
            base[key] = merge_list(base[key], value)
        else:
            base[key] = merge_configs(base[key], value, merge_strategy)

    return base


def merge_list(base_list: list, new_list: list) -> list:
    for val in new_list:
        if val in base_list:
            base_list.remove(val)
    return new_list + base_list


def _get_merge_strategy(cfg: _CONFIG_TYPE) -> str:
    return cfg.get("merge_strategy", "merge")


# Run initial configuration with the base config
configure(BASE_CONFIG_PATH)
