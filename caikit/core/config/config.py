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
import os
from importlib import metadata

# Third Party
import semver

# First Party
import aconfig
import alog

# Local
from ..toolkit.config_utils import merge_configs
from ..toolkit.errors import error_handler

log = alog.use_channel("CONFIG")
error = error_handler.get(log)


# restrict functions that are imported so we don't pollute the base module namespce
__all__ = [
    "Config",
    "compare_versions",
    "ConfigParser"
]

log = alog.use_channel("CKCCNFG")
error = error_handler.get(log)

BASE_CONFIG_PATH = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "config.yml")
)


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
