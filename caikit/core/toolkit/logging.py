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


def configure(default_level, filters="urllib3:off", formatter=None, thread_id=False):
    """Utility function to initialize logging stack components.

    Args:
        default_level:  str
            This is the level that will be enabled for a given channel when a specific level has
            not been set in the filters.
        filters:  str/dict
            This is a mapping from channel name to level that allows levels to be set on a
            per-channel basis. If a string, it is formatted as "CHAN:info,FOO:debug". If a dict,
            it should map from channel string to level string.
        formatter:  str ('pretty' or 'json')/AlogFormatterBase
            The formatter is either the string 'pretty' or 'json' to indicate one of the default
            formatting options or an instance of AlogFormatterBase for a custom formatter
            implementation.
        thread_id  bool
            If true, include thread.
    """
    # Default to using a 12-character channel slot with the pretty printer
    formatter = formatter or alog.AlogPrettyFormatter(12)

    # Initialize the python alog stack
    alog.configure(default_level, filters, formatter, thread_id)
