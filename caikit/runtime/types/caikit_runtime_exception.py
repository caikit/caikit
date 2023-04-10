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
import uuid


class CaikitRuntimeException(Exception):
    """Exception raised when something went wrong while running the Caikit Runtime.

    Args:
        status_code (grpc.StatusCode): gRPC status code for the exception
        message (str): relevant information regarding what went wrong
        metadata (dict): defaults to {}, but contains any other objects/values relevant to exception

    Attributes:
        Same as Args. See above.
    """

    def __init__(self, status_code, message, metadata=None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.id = uuid.uuid4().hex
        # in order to prevent errors in get_error_log .update call
        if metadata and isinstance(metadata, dict):
            self.metadata = metadata
            self.metadata.update({"error_id": self.id})
        else:
            # metadata is of None type
            self.metadata = {"error_id": self.id}
