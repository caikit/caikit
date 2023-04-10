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

# Third Party
import semver

# First Party
import aconfig
import alog

# Local
from ..toolkit.errors import error_handler

log = alog.use_channel("CONFIG")
error = error_handler.get(log)


# restrict functions that are imported so we don't pollute the base module namespce
__all__ = [
    "Config",
    "compare_versions",
]

log = alog.use_channel("CKCCNFG")
error = error_handler.get(log)


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
