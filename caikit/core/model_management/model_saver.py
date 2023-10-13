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
from caikit.interfaces.common.data_model.stream_sources import File

T = typing.TypeVar("T")


class ModelSaver(typing.Generic[T]):
    """Generic-typed, abstract base for model saver"""

    @abc.abstractmethod
    def save_model(
        self, model: ModuleBase, model_name: str, training_id: Optional[str]
    ):
        """Save the loaded model, based on this target's configuration.

        Args:
            model (ModuleBase): a loaded model to be saved
            model_name (str): The user-provided name of this model
            training_id (str | None): The globally unique id tracking the training run that
                created this model

        Returns:
            None on success

        Raises:
            Any appropriate exception if saving the model fails
        """


# Extend OutputTarget with a concrete message type
class LocalFileModelSaver(ModelSaver[File]):
    """Holds the actual impl for saving the model"""

    # TODO: Do we even keep the base `training_output_dir`?
    # Could create a `LocalFileModelSaver` with that as the target if none given in api?
    # But also very awkward to have both

    def __init__(self, target: File, save_with_id: bool):
        self.target = target
        self.save_with_id = save_with_id

    def save_model(
        self, model: ModuleBase, model_name: str, training_id: Optional[str]
    ):
        base_path = self.target.filename
        if self.save_with_id:
            base_path = os.path.join(base_path, training_id)

        model.save(output_path=os.path.join(base_path, model_name))

