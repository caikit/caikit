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

# First Party
import alog

# Local
from caikit import get_config
from caikit.runtime.model_management.directory_model_sizer import DirectoryModelSizer

log = alog.use_channel("MM-SIZER")


class ModelMeshModelSizer(DirectoryModelSizer):
    """ModelMeshModelSizer. This class estimates a models size based on
    the contents of the directory multiplied by a model specific
    constant"""

    name = "MODEL_MESH"

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
        return int(
            super().get_model_size(model_id, local_model_path, model_type) * multiplier
        )
