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
import abc

# First Party
import alog

# Local
from caikit.core.toolkit.factory import FactoryConstructible

log = alog.use_channel("MODEL-SIZER")


class ModelSizerBase(FactoryConstructible):
    """Model Sizer Base class. This class contains the"""

    @abc.abstractmethod
    def get_model_size(
        self, model_id: str, local_model_path: str, model_type: str
    ) -> int:
        """
        Returns the estimated memory footprint of a model
        Args:
            model_id: The model identifier, used for informative logging
            cos_model_path: The path to the model archive in S3 storage
            model_type: The type of model, used to adjust the memory estimate
        Returns:
            The estimated size in bytes of memory that would be used by loading this model
        """
