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
A ModelFinder is responsible for finding models based on a unique input string
which may be a path, id, or other identifier.

Configuration for ModelFinders lives under the config as follows:

model_management:
    finders:
        <finder name>:
            type: <finder type name>
            config:
                <config option>: <value>
"""

# Standard
from typing import Optional
import abc

# Local
from ..modules import ModuleConfig
from ..toolkit.factory import FactoryConstructible


class ModelFinderBase(FactoryConstructible):
    __doc__ = __doc__

    @abc.abstractmethod
    def find_model(
        self,
        model_path: str,
        **kwargs,
    ) -> Optional[ModuleConfig]:
        """Find any model that can be uniquely identified by the given (logical)
        path. If found, return the in-memory ModuleConfig.

        Args:
            model_path (str): The logical path to the model (name, id, file
                path, etc...)
            **kwargs: All finders must allow additional kwargs through so that
                specific finders and loaders can support additional optional
                arguments.

        Returns:
            result (Optional[ModuleConfig]): If found, the in-memory config
                object for the model. If the model cannot be found, an exception
                may be raised or None returned.
        """
