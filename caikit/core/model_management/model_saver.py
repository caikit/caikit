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
"""This file defines the abstraction for handling the output of training a model"""
# Standard
from typing import Optional
import abc
import os
import typing

# First Party
from ..modules import ModuleBase
from caikit.interfaces.common.data_model.stream_sources import PathReference

T = typing.TypeVar("T")


class ModelSaver(typing.Generic[T]):
    """Generic-typed, abstract base for model saver"""

    @abc.abstractmethod
    def save_model(
        self, model: ModuleBase, model_name: str, training_id: Optional[str]
    ) -> typing.Any:
        """Save the loaded model, based on this target's configuration.

        Args:
            model (ModuleBase): a loaded model to be saved
            model_name (str): The user-provided name of this model
            training_id (str | None): The globally unique id tracking the training run that
                created this model

        Returns:
            Any: Some representation of where the model was saved. This could be a path on disk.

        Raises:
            Any appropriate exception if saving the model fails
        """

    @abc.abstractmethod
    def save_path(self, model_name: str, training_id: Optional[str]) -> typing.Any:
        """If applicable, return some info about where the model was / will be saved.

        Args:
            model_name (str): The user-provided name of this model
            training_id (str | None): The globally unique id tracking the training run that
                created this model

        Returns:
            Any: Some representation of where the model was saved. This could be a path on disk.
                    None if not applicable
        """

    @classmethod
    def _save_path_with_id(
            cls,
            save_path: Optional[str],
            save_with_id: bool,
            training_id: str,
            model_name: Optional[str],
    ) -> Optional[str]:
        """Shared utility method to inject both the training id and model name
        into a save path.
        """
        if save_path is None:
            return save_path

        final_path_parts = [save_path]
        # If told to save with the ID in the path, inject it before the
        # model name.
        if save_with_id and training_id not in save_path:
            # Don't inject training id if it's already in the path
            final_path_parts.append(training_id)

        if model_name and model_name not in save_path:
            # Don't inject model name if it's already in the path
            final_path_parts.append(model_name)

        return os.path.join(*final_path_parts)


# Extend OutputTarget with a concrete message type
class LocalPathModelSaver(ModelSaver[PathReference]):
    """Holds the actual impl for saving the model"""

    # TODO: Do we even keep the base `training_output_dir`?
    # Could create a `LocalFileModelSaver` with that as the target if none given in api?
    # But also very awkward to have both

    def __init__(self, target: PathReference, save_with_id: bool):
        self.target = target
        self.save_with_id = save_with_id

    def save_model(
        self, model: ModuleBase, model_name: str, training_id: Optional[str]
    ) -> str:
        save_path = self._save_path_with_id(
            save_path=self.target.path,
            save_with_id=self.save_with_id,
            training_id=training_id,
            model_name=model_name
        )

        model.save(model_path=save_path)
        return save_path
