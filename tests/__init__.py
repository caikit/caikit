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
import os

# First Party
import alog

# We used to compile protos here. That now lives in a hook in conftest.py!

alog.configure(
    default_level=os.environ.get("LOG_LEVEL", "info").lower(),
    filters=os.environ.get("LOG_FILTERS", ""),
    formatter="json"
    if os.environ.get("LOG_JSON", "false").lower() == "true"
    else "pretty",
    thread_id=os.environ.get("LOG_THREAD_ID", "false").lower() == "true",
)
