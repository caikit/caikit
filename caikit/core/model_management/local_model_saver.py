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

# First Party
import aconfig

# Local
from ..modules import ModuleBase
from .model_saver_base import ModelSaverBase, ModelSaverBuilderBase


class LocalModelSaver(ModelSaverBase[str]):
    """Holds the actual impl for saving the model"""

    def __init__(self, target: str, save_with_id: bool):
        self.target = target
        self.save_with_id = save_with_id

    def save_model(
        self, model: ModuleBase, model_name: str, training_id: Optional[str]
    ) -> str:
        save_path = self._save_path(model_name=model_name, training_id=training_id)
        model.save(model_path=save_path)
        return save_path

    def _save_path(self, model_name: str, training_id: Optional[str]) -> str:
        return self._get_save_path_with_id(
            save_path=self.target,
            save_with_id=self.save_with_id,
            training_id=training_id,
            model_name=model_name,
        )


class LocalModelSaverBuilder(ModelSaverBuilderBase[str]):

    name = "LOCAL"

    def __init__(self, config: aconfig.Config, instance_name: str):
        self._instance_name = instance_name
        self.save_with_id = config.save_with_id

    def build_model_saver(self, output_target: str) -> ModelSaverBase[int]:
        return LocalModelSaver(target=output_target, save_with_id=self.save_with_id)
