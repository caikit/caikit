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
The LocalModelFinder locates models locally on disk that contain a caikit-native
config.yml file.

Configuration for LocalModelFinder lives under the config as follows:

model_management:
    finders:
        <finder name>:
            type: LOCAL
            config:
                # Path to a local directory that holds models for relative paths
                load_path: <null or str>
"""
# Standard
from typing import Optional
import os

# First Party
import aconfig
import alog

# Local
from ..modules import ModuleConfig
from ..toolkit.errors import error_handler
from .model_finder_base import ModelFinderBase

log = alog.use_channel("LFIND")
error = error_handler.get(log)


class LocalModelFinder(ModelFinderBase):
    __doc__ = __doc__

    name = "LOCAL"

    def __init__(self, config: aconfig.Config, instance_name: str):
        """Initialize with an optional path prefix"""
        self._instance_name = instance_name
        self._load_path = config.load_path

    def find_model(
        self,
        model_path: str,
        **__,
    ) -> Optional[ModuleConfig]:
        """Find a model at the local path or with the configured prefix"""
        full_model_path = model_path
        if not os.path.exists(model_path) and self._load_path:
            full_model_path = os.path.join(self._load_path, model_path)
            log.debug2(
                "Looking for %s in %s using %s",
                model_path,
                full_model_path,
                self._instance_name,
            )
        full_model_path = os.path.normpath(full_model_path)
        return ModuleConfig.load(full_model_path)
