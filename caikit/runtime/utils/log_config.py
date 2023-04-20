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
# DEBUG -- This needs a hook to configure library-specific logging
from caikit import get_config
from caikit.core.toolkit.logging import configure
# Alog is a wrapper around the standard logging package in Python; there may be logs from
# coming from channels that you don't want to see (e.g., Boto3). To silence these, add them
# to the list of ALOG_FILTERS; each channel in this list will be set to ALOG_FILTER_LEVEL.

def initialize_logging():
    """Initializes the logging framework (wrapper around alog.configure). If a pretty formatter
    is used, construct one explicitly so that we can set the channel width using our environment
    variable override."""
    config = get_config()

    formatter = (
        alog.AlogPrettyFormatter(int(config.alog_channel_width))
        if config.alog_formatter == "pretty"
        else config.alog_formatter
    )
    configure(
        default_level=config.log_level,
        filters=config.alog_filters,
        formatter=formatter,
        thread_id=config.alog_thread_id,
    )
