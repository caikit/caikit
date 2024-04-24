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

# Local
from .http_server import (
    HEALTH_ENDPOINT,
    MODEL_MANAGEMENT_ENDPOINT,
    MODELS_INFO_ENDPOINT,
    RUNTIME_INFO_ENDPOINT,
    TRAINING_MANAGEMENT_ENDPOINT,
    RuntimeHTTPServer,
)
from .pydantic_wrapper import dataobject_to_pydantic, pydantic_to_dataobject
