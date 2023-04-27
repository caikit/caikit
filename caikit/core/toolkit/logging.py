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
"""Logging top-level configuration for `caikit.core` library.
"""

# First Party
import alog

# Local
from caikit.config import get_config


def configure():
    """Utility function to initialize logging stack components.

    This uses the caikit configuration to determine how logging should be configured.

    This should _only_ be called in the context of executing a __main__ application.
    Configuring the logger from a library context may override log config that has
    already been set by the consuming application.
    """
    caikit_config = get_config()

    # For pretty format, build the formatter with the configured channel width
    if caikit_config.log.formatter == "pretty":
        formatter = alog.AlogPrettyFormatter(caikit_config.log.channel_width)
    else:
        # Otherwise just use the config string
        formatter = caikit_config.log.formatter

    # Initialize the python alog stack
    alog.configure(
        caikit_config.log.level,
        caikit_config.log.filters,
        formatter,
        caikit_config.log.thread_id,
    )
