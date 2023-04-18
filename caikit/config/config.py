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


"""Config methods for the `caikit.core` library. Mainly interacts with `config.yml`.
"""

# Standard
from importlib import metadata
from typing import Optional
import os

# Third Party
import semver

# First Party
import aconfig
import alog

# Local
from caikit.core.toolkit.config_utils import merge_configs
from caikit.core.toolkit.errors import error_handler

log = alog.use_channel("CONFIG")
error = error_handler.get(log)


# restrict functions that are imported so we don't pollute the base module namespce
__all__ = ["Config", "compare_versions", "parse_config"]

BASE_CONFIG_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "config.yml")
)


def configure(library_config_yml_path: Optional[str] = None):
    """Configure caikit for your usage!
    Sets `caikit.config` to an aconfig.Config object with overrides from multiple sources.

    Sources, last takes precedence:
        1. caikit's base config.yml file baked into this repo
        2. `library_config_yml_path`
        3. The config files specified in the `config_files` configuration
            (NB: This may be set by the `CONFIG_FILES` environment variable)
        4. Environment variables, in ALL_CAPS_SNAKE_FORMAT

    Args:
        library_config_yml_path (Optional[str]): The path to the base configuration yaml
            with overrides for your library. If omitted, only the base caikit config is used.

    Returns: None
        This only sets the `caikit.config` attribute.
    """

    cfg = parse_config(library_config_yml_path)
    # Assign this new config back to the top-level `caikit.config` attribute
    # Local
    import caikit

    caikit.config = cfg

    # TODO: hook into any inner `configure()` calls that need to happen (fill this section in)
    error_handler.ENABLE_ERROR_CHECKS = caikit.config.enable_error_checks
    error_handler.MAX_EXCEPTION_LOG_MESSAGES = caikit.config.max_exception_log_messages

    # TODO: Or think about having those pull config dynamically?


def parse_config(extra_config_yml: Optional[str] = None) -> aconfig.Config:
    """This function parses configuration files used to manage configuration settings for caikit.
    It draws values first out of the base config.yml file packaged within this repo.
    It then merges in an optional extra config
    It will also merge in any configuration from config yaml files specified in a comma-separated
    list in the environment variable 'CONFIG_FILES', going from left to right and overwriting on
    merge. (Last takes precedence)
    """
    # Start with the base config
    config = aconfig.Config.from_yaml(BASE_CONFIG_PATH, override_env_vars=True)

    # Merge in the supplied config file
    if extra_config_yml:
        new_overrides = aconfig.Config.from_yaml(
            extra_config_yml, override_env_vars=True
        )
        config = merge_configs(config, new_overrides)

    # Merge in config from any other user-provided config files
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


class ConfigParser:
    """This classes manages configuration settings for caikit.
    It draws values first out of the base config.yml file packaged within this repo.
    It will also merge in any configuration from config yaml files specified in a comma-separated
    list in the environment variable 'CONFIG_FILES', going from left to right and overwriting on
    merge. (Last takes precedence)
    """

    @staticmethod
    def get_config(config_path: str = BASE_CONFIG_PATH) -> aconfig.Config:
        config = aconfig.Config.from_yaml(config_path, override_env_vars=True)
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
                new_overrides = aconfig.Config.from_yaml(file, override_env_vars=True)
                config = merge_configs(config, new_overrides)
        return config


class Config(aconfig.Config):
    @classmethod
    def get_config(cls, library_name, config_path, use_legacy_versioning=False):
        """Get top-level configuration for `caikit.core` library and extensions. Generally only
        used by internal functions.
        """
        out_config = cls.from_yaml(config_path)
        # useful variables to have
        out_config.library_name = library_name
        # If we enable legacy versioning, use <libname>_version from the config
        if use_legacy_versioning:
            out_config.library_version_key = "{0}_version".format(
                out_config.library_name
            )
            out_config.library_version = out_config[out_config.library_version_key]
        else:
            try:
                out_config.library_version = metadata.version(library_name)
            except metadata.PackageNotFoundError:
                log.debug("<COR25991305D>", "No library version found")
                out_config.library_version = "0.0.0"

        return out_config


def compare_versions(v1, v2):
    """Compare a given version against the other. Used for comparing model and library versions.

    Args:
        v1:  str
            SemVer version to compare.
        v2:  str
            SemVer version to compare.

    Returns:
        int
            -1 if `v1` version is less than `v2`, 0 if equal and 1 if greater
    """
    return semver.VersionInfo.parse(v1).compare(v2)
