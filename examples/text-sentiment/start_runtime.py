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
from os import path
import sys

# First Party
import alog

sys.path.append(
    path.abspath(path.join(path.dirname(__file__), "../"))
)  # Here we assume that `start_runtime` file is at the same level of the `text_sentiment` package

# Local
import text_sentiment

alog.configure(default_level="debug")

# Local
from caikit.runtime import grpc_server

grpc_server.main()
