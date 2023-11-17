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
"""Local implementation of a model saver"""
# Standard
from typing import Optional
import functools

# First Party
import aconfig

# Local
from ..modules import ModuleBase
from .model_saver_base import ModelSaveFunctor, ModelSaverBase


class LocalModelSaver(ModelSaverBase[str]):
    """Local implementation of a model saver. Simply saves models to disk"""

    name = "LOCAL"

    def __init__(self, config: aconfig.Config, instance_name: str):
        self._instance_name = instance_name
        self.save_with_id = config.save_with_id

    def save_functor(self, output_target: str) -> ModelSaveFunctor:
        """Returns a ModelSaveFunctor that saves the model under the path `output_target`."""
        return functools.partial(self._save_method, output_target)

    def _save_method(
        self,
        output_target,
        model: ModuleBase,
        model_name: str,
        training_id: Optional[str],
    ) -> str:
        save_path = self._get_save_path(
            target=output_target, model_name=model_name, training_id=training_id
        )
        model.save(model_path=save_path)
        return save_path

    def _get_save_path(
        self, target: str, model_name: str, training_id: Optional[str]
    ) -> str:
        return self._get_save_path_with_id(
            save_path=target,
            save_with_id=self.save_with_id,
            training_id=training_id,
            model_name=model_name,
        )
