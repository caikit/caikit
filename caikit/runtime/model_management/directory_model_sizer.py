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
import aconfig
import alog

# Local
from caikit.runtime.model_management.model_sizer_base import ModelSizerBase
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException

log = alog.use_channel("DIRECTORY-SIZER")


class DirectoryModelSizer(ModelSizerBase):
    """DirectoryModelSizer. This class calculates a models size based on the
    size of the files in the model directory

    ! Note: It caches the size of the directory after first sizing which can cause
    race conditions in certain situations.
    """

    name = "DIRECTORY"

    def __init__(self, config: aconfig.Config, instance_name: str):
        super().__init__(config, instance_name)
        # Cache of archive sizes: directory model path -> archive size in bytes
        self.model_directory_size: Dict[str, int] = {}

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
        # Return the cached model size if one exists
        if model_size := self.model_directory_size.get(local_model_path):
            return model_size

        # Calculate the model size and add it to the cache.  This uses last in
        # methodology so that the most recent size is used during parallel access
        dir_size = self.__get_directory_size(model_id, local_model_path)
        self.model_directory_size[local_model_path] = dir_size
        return dir_size

    def __get_directory_size(self, model_id, local_model_path) -> int:
        """Get the size of a directory"""
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
            message = (
                f"Failed to estimate size of model '{model_id}',"
                f"file '{local_model_path}' not found"
            )
            log.error("<RUN62168924E>", message)
            raise CaikitRuntimeException(grpc.StatusCode.NOT_FOUND, message) from ex
