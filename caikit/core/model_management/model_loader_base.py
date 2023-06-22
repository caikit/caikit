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
A ModelLoader is responsible for taking an in-memory ModuleConfig and producing
a usable Module instance.
"""

# Standard
from typing import Optional
import abc

# Local
from ..modules import ModuleBase, ModuleConfig


class ModelLoaderBase(abc.ABC):
    __doc__ = __doc__

    @abc.abstractmethod
    def load_model(model_config: ModuleConfig) -> Optional[ModuleBase]:
        """Given a ModelConfig, attempt to load it into memory

        Args:
            model_config (ModuleConfig): The in-memory model config object for
                the model to be loaded

        Returns:
            model (Optional[ModuleBase]): The in-memory ModuleBase instance that
                is ready to run
        """
