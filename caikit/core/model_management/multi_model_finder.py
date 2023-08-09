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
The MultiModelFinder configures a set of other model finders that will be used
in sequence to try loading models.

Configuration for MultiModelFinder lives under the config as follows:

model_management:
    finders:
        <finder name>:
            type: MULTI
            config:
                # Sequence of other finder names to use in priority order
                finder_sequence:
                    - other_finder1
                    - other_finder2
"""
# Standard
from typing import Optional

# First Party
import aconfig
import alog

# Local
from ...config import get_config
from ..exceptions import error_handler
from ..modules import ModuleConfig
from .model_finder_base import ModelFinderBase

# NOTE: Top-level import done so that global MODEL_MANAGER can be used at
#   construction time without incurring a circular dependency
import caikit.core

log = alog.use_channel("MFIND")
error = error_handler.get(log)


class MultiModelFinder(ModelFinderBase):
    __doc__ = __doc__

    name = "MULTI"

    def __init__(self, config: aconfig.Config, instance_name: str):
        """Initialize with the sequence of finders to use"""
        self._instance_name = instance_name
        finder_priority = config.finder_priority
        error.type_check(
            "<COR47518221E>",
            list,
            finder_priority=finder_priority,
        )
        error.type_check_all(
            "<COR47518222E>",
            str,
            finder_priority=finder_priority,
        )
        error.value_check(
            "<COR05042343E>",
            finder_priority,
            "Must provide at least one valid finder",
        )
        config_finders = get_config().model_management.finders
        invalid_finders = [
            finder for finder in finder_priority if finder not in config_finders
        ]
        error.value_check(
            "<COR68252034E>",
            not invalid_finders,
            "Invalid finders given in finder_priority: {}",
            invalid_finders,
        )
        error.value_check(
            "<COR54613971E>",
            self._instance_name not in finder_priority,
            "Cannot include self in multi finder priority",
        )
        model_manager = config.model_manager or caikit.core.MODEL_MANAGER
        log.debug2("Setting up %s with finder priority: %s", self.name, finder_priority)
        self._finders = [model_manager.get_finder(finder) for finder in finder_priority]

    def find_model(
        self,
        model_path: str,
        **kwargs,
    ) -> Optional[ModuleConfig]:
        """Iterate through the sequence of finders and return the first one that
        succeeds
        """
        for idx, finder in enumerate(self._finders):
            log.debug2(
                "Trying to find %s with finder %d of type %s",
                model_path,
                idx,
                finder.name,
            )
            try:
                module_config = finder.find_model(model_path, **kwargs)
                if module_config:
                    log.debug(
                        "Found %s with finder %d of type %s",
                        model_path,
                        idx,
                        finder.name,
                    )
                    return module_config
                log.debug2(
                    "Finder %d of type %s unable to find %s",
                    idx,
                    finder.name,
                    model_path,
                )
            except Exception as err:  # pylint: disable=broad-exception-caught
                log.debug2(
                    "Finder %d of type %s failed to load %s: %s",
                    idx,
                    finder.name,
                    model_path,
                    err,
                )
                log.debug4("Finder error", exc_info=True)

        # No finder succeeded
        log.warning("Unable to find %s with any finder", model_path)
        return None
