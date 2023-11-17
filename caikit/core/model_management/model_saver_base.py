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
"""This file defines the abstraction for saving the output of training a model"""
# Standard
from typing import Optional
import abc
import os
import typing

# First Party
import alog

# Local
from ..modules import ModuleBase
from ..toolkit.factory import FactoryConstructible
from caikit.core.exceptions import error_handler

log = alog.use_channel("MODEL_SAVER_BASE")

error = error_handler.get(log)

OutputTargetType = typing.TypeVar("OutputTargetType")


class ModelSaveFunctor(typing.Protocol):
    """Interface definition for a `save` function"""

    def __call__(self, model: ModuleBase, model_name: str, training_id: Optional[str]):
        """Saves the `model` with name `model_name` created by training job `training_id`"""


class ModelSaverBase(typing.Generic[OutputTargetType], FactoryConstructible):
    """A ModelSaverBuilder is lifetime-scoped with the application.
    It holds static configuration that is used to construct individual ModelSavers.
    """

    def __new__(cls, *args, **kwargs):
        origins = [typing.get_origin(b) for b in cls.__orig_bases__]
        error.value_check(
            "<COR16695051E>",
            ModelSaverBase in origins,
            "Missing generic type on class {}",
            cls,
        )
        instance = super().__new__(cls)
        return instance

    @abc.abstractmethod
    def save_functor(self, output_target: OutputTargetType) -> ModelSaveFunctor:
        """Construct a new save functor to save a model to the given target

        Args:
            output_target (OutputTargetType):

        Returns:
            ModelSaveFunctor:
                A model save functor for this output target type
        """

    @classmethod
    def output_target_type(cls) -> typing.Type[OutputTargetType]:
        bases = cls.__orig_bases__
        # Guaranteed to exist if __new__ succeeds
        output_target_base = [
            b for b in bases if typing.get_origin(b) == ModelSaverBase
        ][0]
        return typing.get_args(output_target_base)[0]

    @classmethod
    def _get_save_path_with_id(
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
