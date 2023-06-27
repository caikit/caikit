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

# Standard
from pathlib import Path
from typing import Dict
import os

# Third Party
import grpc

# First Party
import alog

# Local
from caikit import get_config
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException

log = alog.use_channel("MODEL-SIZER")


class ModelSizer:
    """Model Loader class. The singleton class contains the core implementation details
    for loading models in from S3."""

    __instance = None

    def __init__(self):
        # Re-instantiating this is a programming error
        assert self.__class__.__instance is None, "This class is a singleton!"
        ModelSizer.__instance = self

        # Cache of archive sizes: cos model path -> archive size in bytes
        self._model_archive_sizes: Dict[str, int] = {}

    def get_model_size(self, model_id, local_model_path, model_type) -> int:
        """
        Returns the estimated memory footprint of a model
        Args:
            model_id: The model identifier, used for informative logging
            cos_model_path: The path to the model archive in S3 storage
            model_type: The type of model, used to adjust the memory estimate
        Returns:
            The estimated size in bytes of memory that would be used by loading this model
        """
        # Cache model's size
        if local_model_path not in self._model_archive_sizes:
            self._model_archive_sizes[local_model_path] = self.__get_archive_size(
                model_id, local_model_path
            )

        return self.__estimate_with_multiplier(
            model_id, model_type, self._model_archive_sizes[local_model_path]
        )

    def __estimate_with_multiplier(self, model_id, model_type, archive_size) -> int:
        if (
            model_type
            in get_config().inference_plugin.model_mesh.model_size_multipliers
        ):
            multiplier = (
                get_config().inference_plugin.model_mesh.model_size_multipliers[
                    model_type
                ]
            )
            log.debug(
                "Using size multiplier '%f' for model '%s' to estimate model size",
                multiplier,
                model_id,
            )
        else:
            multiplier = (
                get_config().inference_plugin.model_mesh.default_model_size_multiplier
            )
            log.info(
                "<RUN62161564I>",
                "No configured model size multiplier found for model type '%s' for model '%s'. "
                "Using default multiplier '%f'",
                model_type,
                model_id,
                multiplier,
            )
        return int(archive_size * multiplier)

    def __get_archive_size(self, model_id, local_model_path) -> int:
        try:
            if os.path.isdir(local_model_path):
                # Walk the directory to size all files
                return sum(
                    file.stat().st_size
                    for file in Path(local_model_path).rglob("*")
                    if file.is_file()
                )

            # Probably just an archive file
            return os.path.getsize(local_model_path)
        except FileNotFoundError as ex:
            message = "Failed to estimate size of model '%s', file '%s' not found" % (
                model_id,
                local_model_path,
            )
            log.error("<RUN62168924E>", message)
            raise CaikitRuntimeException(grpc.StatusCode.NOT_FOUND, message) from ex

    @classmethod
    def get_instance(cls) -> "ModelSizer":
        """This method returns the instance of Model Manager"""
        if not cls.__instance:
            cls.__instance = ModelSizer()
        return cls.__instance
